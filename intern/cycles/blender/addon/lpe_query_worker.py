# SPDX-License-Identifier: GPL-2.0-or-later
"""Private background CPU renderer for the contribution-query panels."""
from collections import defaultdict
import json
import os
from pathlib import Path
import sys
import threading
import time

import bpy
import numpy as np
import PyOpenColorIO as ocio
import _cycles
from mathutils import Matrix


class QueryFilm:
    def __init__(self, job, scene):
        self.job = job
        x, y, self.width, self.height = job['rect']
        self.origin = (x, y)
        self.routes = defaultdict(dict)
        self.counts = np.zeros((2, self.height * self.width), dtype=np.float64)
        self.beauty = np.zeros((2, self.height * self.width, 3), dtype=np.float64)
        self.types = {o.name: o.type for o in scene.objects}
        self.types.update({o.name: o.type for o in bpy.data.objects})
        world = scene.view_layers[job['layer']].world_override or scene.world
        self.world = world.name if world else None
        self.separate = scene.cycles.photon_caustics_separate_sampling
        self.exposure = 2 ** scene.view_settings.exposure
        self.gamma = scene.view_settings.gamma
        path = os.environ.get('OCIO') or bpy.utils.system_resource('DATAFILES', path='colormanagement/config.ocio')
        if not path:
            path = Path(bpy.utils.resource_path('LOCAL')) / 'datafiles/colormanagement/config.ocio'
        config = ocio.Config.CreateFromFile(str(path))
        self.luma = np.array(config.getDefaultLumaCoefs())
        display = scene.display_settings.display_device
        view = scene.view_settings.view_transform
        transform = ocio.DisplayViewTransform(src=ocio.ROLE_SCENE_LINEAR, display=display, view=view)
        group = ocio.GroupTransform()
        look = scene.view_settings.look
        if config.getLook(look) is not None:
            group.appendTransform(ocio.LookTransform(src=ocio.ROLE_SCENE_LINEAR,
                                  dst=ocio.ROLE_SCENE_LINEAR, looks=look))
        group.appendTransform(transform)
        self.processor = config.getProcessor(group).getDefaultCPUProcessor()
        self.version = 0

    def add(self, batch):
        rows, objects, materials, world_index = batch
        for path, photon, entries in rows:
            coords = np.asarray(entries)
            x = coords[:, 0].astype(int) - self.origin[0]
            y = coords[:, 1].astype(int) - self.origin[1]
            flat = y * self.width + x
            if np.any((x < 0) | (x >= self.width) | (y < 0) | (y >= self.height)):
                raise ValueError('Contribution query pixel outside crop')
            if path == ((-1, -1, -1),):
                self.counts[photon, flat] += coords[:, 2]
                continue
            if not path:
                self.beauty[photon, flat] += coords[:, 2:]
                continue
            chain = []
            for index, shader, event in reversed(path):
                name = objects.get(index, '')
                kind = self.types.get(name, 'OBJECT')
                if not chain and event == 0 and self.world is not None and index == world_index:
                    kind, name = 'WORLD', self.world
                material = '' if kind in {'LIGHT', 'WORLD'} else materials.get(shader, '')
                chain.append((kind, name, material, event, index))
            values = self.routes[tuple(chain), photon]
            for pixel, rgb in zip(flat, coords[:, 2:]):
                if pixel in values:
                    values[pixel] += rgb
                else:
                    values[pixel] = rgb.copy()
        return bool(rows)

    def preview(self, pixels):
        rgb = np.ascontiguousarray(pixels.reshape(self.height, self.width, 3) * self.exposure, dtype=np.float32)
        self.processor.applyRGB(rgb)
        if self.gamma != 1:
            np.power(np.maximum(rgb, 0), 1 / self.gamma, out=rgb)
        rgba = np.empty((self.height, self.width, 4), dtype=np.uint8)
        rgba[:, :, :3] = np.clip(np.nan_to_num(rgb) * 255 + .5, 0, 255)
        rgba[:, :, 3] = 255
        return rgba

    def packet(self, done=False):
        denom = np.maximum(self.counts, 1)
        # The estimators can have different sample counts during a motion-blurred render.
        # Normalize each family first. Rank and expand only the requested routes.
        total_light = self.job.get('total_light', False)
        energies = defaultdict(float)
        for (chain, photon), values in self.routes.items():
            indices = np.fromiter(values, dtype=np.intp)
            rgb = np.asarray(list(values.values()))
            divisor = denom[int(bool(photon) and self.separate), indices]
            energy = float(((rgb @ self.luma) / divisor).sum())
            energies[chain, False if total_light else photon] += energy
        ranking = [(energy, chain, photon) for (chain, photon), energy in energies.items()]
        ranking.sort(key=lambda v: (-v[0], v[1], v[2]))
        total = sum(max(row[0], 0) for row in ranking)
        images = [self.preview((self.beauty / denom[:, :, None]).sum(axis=0))]
        results = []
        for energy, chain, photon in ranking[:self.job['limit']]:
            if energy <= 0:
                break
            image = np.zeros_like(self.beauty[0])
            for family in ((False, True) if total_light else (photon,)):
                values = self.routes.get((chain, family))
                if values:
                    indices = np.fromiter(values, dtype=np.intp)
                    rgb = np.asarray(list(values.values()))
                    image[indices] += rgb / denom[int(bool(family) and self.separate), indices, None]
            images.append(self.preview(image))
            results.append({'path': chain, 'photon': photon, 'total_light': total_light, 'share': energy / total})
        self.version += 1
        meta = dict(version=self.version, done=done, rows=results, candidates=len(ranking),
                    samples=self.counts.mean(axis=1).tolist())
        return meta, np.stack(images)

    def publish(self, done=False):
        meta, images = self.packet(done)
        target = Path(self.job['output'])
        temp = target.with_suffix('.tmp')
        with temp.open('wb') as stream:
            np.savez(stream, meta=json.dumps(meta), images=images)
        # Windows can briefly deny replacement while the UI reads the previous batch.
        for attempt in range(6):
            try:
                os.replace(temp, target)
                break
            except PermissionError:
                if attempt == 5:
                    if done:
                        # A reader may still hold the previous packet. This job's final
                        # filename is new, so it can be published without replacing it.
                        os.replace(temp, target.with_suffix('.done.npz'))
                    else:
                        temp.unlink()
                    return
                time.sleep(.02 * (attempt + 1))


def run(job):
    scene = bpy.data.scenes[job['scene']]
    layer = scene.view_layers[job['layer']]
    bpy.context.window.scene = scene
    bpy.context.window.view_layer = layer
    scene.render.engine = 'CYCLES'
    c = scene.cycles
    c.device = 'CPU'
    c.integrator_method = 'PATH_TRACING'
    c.use_sample_subset = False
    c.samples = job['samples']
    c.use_adaptive_sampling = c.use_denoising = False
    c.time_limit = 0
    c.use_auto_tile = c.use_photon_cache = False
    c.photon_cache_order = 'SIMULTANEOUS'
    c.photon_caustics_sampling_budget = 'SHARED'
    layer.samples = 0
    scene.render.use_compositing = scene.render.use_sequencer = False
    scene.render.use_multiview = False
    scene.render.threads_mode = 'FIXED'
    scene.render.threads = job['threads']
    width, height = job['frame_size']
    scene.render.resolution_x, scene.render.resolution_y = width, height
    scene.render.resolution_percentage = 100
    if 'view' in job:
        v = job['view']
        cam = bpy.data.cameras.new('Contribution query view')
        ob = bpy.data.objects.new(cam.name, cam)
        scene.collection.objects.link(ob)
        ob.matrix_world = Matrix(v['matrix'])
        cam.type = v['type']
        cam.sensor_fit = 'HORIZONTAL'
        cam.sensor_width = 36
        cam.lens = v['lens']
        cam.ortho_scale = v['scale']
        cam.shift_x, cam.shift_y = v['shift']
        cam.clip_start, cam.clip_end = v['clip']
        if v.get('camera_view') and scene.camera:
            src = scene.camera.data.dof
            cam.dof.use_dof = src.use_dof
            cam.dof.focus_object, cam.dof.focus_distance = src.focus_object, src.focus_distance
            cam.dof.aperture_fstop, cam.dof.aperture_blades = src.aperture_fstop, src.aperture_blades
            cam.dof.aperture_rotation, cam.dof.aperture_ratio = src.aperture_rotation, src.aperture_ratio
        scene.camera = ob
        scene.render.pixel_aspect_x = scene.render.pixel_aspect_y = 1
        scene.render.use_motion_blur = False  # Cycles preview synchronization uses a static shutter.
    x, y, w, h = job['rect']
    scene.render.use_border = scene.render.use_crop_to_border = True
    scene.render.border_min_x, scene.render.border_max_x = x / width, (x + w) / width
    scene.render.border_min_y, scene.render.border_max_y = y / height, (y + h) / height
    # Different shutter corrections can require two BVH states for a final-camera query.
    # Alternate short sample subsets in that case. A static IPR uses one joint camera path.
    split_motion = (scene.render.use_motion_blur and c.caustics_method == 'PHOTON' and
                    c.use_relative_motion_blur != c.use_photon_motion_blur_correction)
    c.photon_caustics_separate_sampling = split_motion
    if split_motion:
        c.use_sample_subset = True
        # Recreate the final-camera session for each shutter subset: a persistent render can
        # retain the previous subset length when switching the two estimator trajectories.
        scene.render.use_persistent_data = False
    screen = QueryFilm(job, scene)
    stop = threading.Event()
    errors = []

    def publish_batches():
        try:
            gate = Path(job['output']).with_suffix('.pause')
            last_batch = 0.
            paused = False
            while not stop.wait(.08):
                requested = gate.exists()
                if requested != paused:
                    _cycles.query_pause(requested)
                    paused = requested
                now = time.monotonic()
                if not split_motion and now - last_batch >= .5:
                    if screen.add(_cycles.query_take(False)):
                        screen.publish()
                    last_batch = now
        except Exception as error:
            errors.append(error)
        finally:
            _cycles.query_pause(False)

    _cycles.query_begin(job.get('total_light', False))
    publisher = threading.Thread(target=publish_batches)
    publisher.start()
    try:
        start = 0
        while start < c.samples:
            count = min(1 if start == 0 else 8, c.samples - start) if split_motion else c.samples
            if split_motion:
                c.sample_offset, c.sample_subset_length = start, count
            status = bpy.ops.render.render(scene=scene.name, layer=layer.name)
            if 'FINISHED' not in status:
                raise RuntimeError('CPU contribution render was cancelled')
            start += count
            if split_motion and start < c.samples:
                screen.add(_cycles.query_take(False))
                screen.publish()
    finally:
        stop.set()
        publisher.join()
        screen.add(_cycles.query_take(True))
    if errors:
        raise errors[0]
    if not screen.counts.any():
        raise RuntimeError('CPU contribution render produced no samples')
    screen.publish(done=True)


if __name__ == '__main__':
    run(json.loads(Path(sys.argv[sys.argv.index('--') + 1]).read_text(encoding='utf8')))
