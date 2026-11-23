# SPDX-License-Identifier: GPL-2.0-or-later
"""Progressive contribution query shared by the LPE and Image Editor sidebars."""
import atexit
from concurrent.futures import ThreadPoolExecutor
import itertools
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile
import time

import bpy
import bpy.utils.previews
from bpy.app.handlers import persistent
from bpy.props import BoolProperty, EnumProperty, IntProperty, PointerProperty, StringProperty
from bpy.types import Operator, Panel, PropertyGroup
from bpy.app.translations import pgettext_iface as tr


EVENTS = {1: 'Diffuse', 2: 'SSS', 4: 'Glossy', 8: 'Transmission', 16: 'Transparent', 32: 'Volume'}
_runtime = None
_previews = None
_handlers = []


class Query:
    def __init__(self):
        self.target = None
        self.rect = None
        self.drag = None
        self.selecting = False
        self.selection_token = None
        self.cursor_windows = []
        self.paused = False
        self.stopped = False
        self.signature = None
        self.pending = 0
        self.process = None
        self.folder = None
        self.snapshot = None
        self.scene_uid = None
        self.scene_revision = 0
        self.meta = {}
        self.icons = []
        self.message = ''
        self.extent = (0, 0)
        self.scene = self.layer = ''
        self.stamp = None
        self.serial = 0
        self.gpu = None
        self.gpu_bytes = 0
        self.backend = ''
        self.reason = ''
        self.request = {}

    def stop(self):
        if self.gpu:
            self.gpu.close()
            self.gpu = None
        if self.process is not None:
            if self.process.poll() is None:
                self.process.kill()
            self.process.wait()
            self.process = None

    def end_selection(self):
        self.selecting, self.drag, self.selection_token = False, None, None
        for window in self.cursor_windows:
            try:
                window.cursor_modal_restore()
            except ReferenceError:
                pass  # The window closed while the selection was active.
        self.cursor_windows.clear()

    def clear(self, preserve=True, *, read_latest=True):
        self.end_selection()
        if preserve and read_latest and self.process and self.request:
            self.read_preview()
        self.stop()
        self.target = self.rect = self.drag = self.signature = None
        self.pending = 0
        self.selecting = self.paused = self.stopped = False
        if self.folder:
            self.folder.cleanup()
            self.folder = self.snapshot = None
        self.request = {}
        if preserve and self.meta:
            self.message = 'Completed' if self.meta['done'] else 'Unfinished'
        else:
            self.meta.clear()
            self.icons.clear()
            self.message = ''
            _previews.clear()
        redraw()

    def restart(self):
        self.stop()
        self.stopped = False
        self.pending = time.monotonic() + .3
        self.message = 'Paused' if self.paused else 'Waiting for view'
        self.stamp = None
        self.remove_preview_files()
        self.request = {}
        self.meta = {}
        self.icons.clear()
        _previews.clear()
        redraw()

    def remove_preview_files(self):
        if self.request.get('output'):
            output = Path(self.request['output'])
            output.unlink(missing_ok=True)
            output.with_suffix('.done.npz').unlink(missing_ok=True)

    def launch_cpu(self, context, request):
        if self.folder is None:
            self.folder = tempfile.TemporaryDirectory(prefix='cycles-lpe-query-')
        directory = Path(self.folder.name)
        if self.snapshot is None:
            snapshot = directory / 'query.blend'
            # Replacing the finished copy avoids a .blend1 backup for each lighting edit.
            staging = directory / 'writing.blend'
            if 'FINISHED' not in bpy.ops.wm.save_as_mainfile(
                    filepath=str(staging), copy=True, compress=False, relative_remap=True):
                raise RuntimeError('Could not save contribution query scene')
            staging.replace(snapshot)
            self.snapshot = snapshot
        self.remove_preview_files()
        self.serial += 1
        output = directory / ('preview-%d.npz' % self.serial)
        spec = directory / 'query.json'
        request['output'] = str(output)
        spec.write_text(json.dumps(request), encoding='utf8')
        with (directory / 'worker.log').open('w', encoding='utf8') as log:
            self.process = subprocess.Popen(
                [bpy.app.binary_path, '-b', str(self.snapshot), '-t', str(request['threads']),
                 '--python-exit-code', '1', '-P', str(Path(__file__).with_name('lpe_query_worker.py')),
                 '--', str(spec)], stdout=log, stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        self.scene, self.layer = request['scene'], request['layer']
        self.request = request
        self.pending = 0
        self.message = 'Sampling'
        redraw()

    def launch(self, context, request):
        self.scene_uid = context.scene.session_uid
        owner = compatible_viewport(context, request) if request.get('device') == 'GPU' else None
        self.gpu_bytes = 0
        self.reason = ''
        if owner is None:
            self.backend = 'CPU fallback' if request.get('device') == 'GPU' else 'CPU'
            self.launch_cpu(context, request)
            return
        self.serial += 1
        self.scene, self.layer = request['scene'], request['layer']
        self.request = request
        self.gpu = LiveGPUQuery(owner, request, context.scene)
        self.backend, self.message, self.pending = 'GPU', 'Sampling', 0
        redraw()

    def advance_gpu(self, context, job):
        reason = self.gpu.step(self, context, job)
        if reason:
            started = self.gpu.revision is not None
            self.stop()
            self.reason = reason
            if started or reason == 'Viewport session closed':
                # Preserve the published sample sequence until the artist starts a new query.
                self.stopped = True
                self.pending = 0
                self.paused = False
                self.message = 'Unfinished'
                redraw()
                return
            self.backend = 'CPU fallback'
            self.snapshot = None
            self.launch_cpu(context, job)
        elif self.meta.get('done'):
            self.stop()
            self.message = 'Completed'
            redraw()

    def read_preview(self):
        if 'output' not in self.request:
            return
        output = Path(self.request['output'])
        final = output.with_suffix('.done.npz')
        if final.exists():
            output = final
        import numpy as np
        try:
            stamp = (output, output.stat().st_mtime_ns)
            if self.stamp == stamp:
                return
            with np.load(output, allow_pickle=False) as packet:
                meta = json.loads(str(packet['meta']))
                images = packet['images']
        except FileNotFoundError:
            return
        self.show_packet(meta, images)
        self.stamp = stamp

    def show_packet(self, meta, images):
        import numpy as np
        icons = []
        for i, image in enumerate(images):
            key = str(i)
            preview = _previews[key] if key in _previews else _previews.new(key)
            height, width = image.shape[:2]
            preview.image_size = width, height
            preview.image_pixels = image.view(np.int32).reshape(-1).tolist()
            icons.append(preview.icon_id)
        self.icons, self.meta = icons, meta
        redraw()


def detected_gpu_ipr(context):
    from .engine import _query_viewports
    if context.scene.cycles.preview_pause:
        return False
    active = {(window, area): ready for (window, area, _), (_, uid, layer, ready) in _query_viewports.items()
              if ready and uid == context.scene.session_uid and layer == context.view_layer.name}
    if not active:
        return 0
    return max((active.get((window.as_pointer(), area.as_pointer()), 0)
                for window in context.window_manager.windows for area in window.screen.areas
                if area.type == 'VIEW_3D' and valid_source(area)), default=0)


def compatible_viewport(ctx, job, check_geometry=True):
    from .engine import _query_viewports
    scene = ctx.scene
    if scene.cycles.preview_pause:
        return None
    for key, (owner, uid, layer, _) in tuple(_query_viewports.items()):
        if uid != scene.session_uid or layer != job['layer']:
            continue
        if 'view' in job:
            if key == _runtime.target:
                return owner
            continue
        # Render Result may borrow only a viewport showing the same camera and projection.
        if not scene.camera or scene.render.use_motion_blur or scene.cycles.use_overscan:
            continue
        for window in ctx.window_manager.windows:
            if window.as_pointer() != key[0]:
                continue
            area = next((a for a in window.screen.areas if a.as_pointer() == key[1]), None)
            if not area or not valid_source(area) or area.type != 'VIEW_3D':
                continue
            space = area.spaces.active
            if space.local_view or space.use_local_collections or not (
                    space.shading.use_scene_world_render and space.shading.use_scene_lights_render):
                continue
            rv = space.region_3d
            if rv.view_perspective != 'CAMERA' or scene.camera.data.type not in {'PERSP', 'ORTHO'}:
                continue
            w, h = job['frame_size']
            region = next((r for r in area.regions if r.as_pointer() == key[2]), None)
            if region is None or viewport_resolution(ctx, region) != (w, h):
                continue
            depsgraph = ctx.evaluated_depsgraph_get()
            camera = scene.camera.evaluated_get(depsgraph)
            projection = camera.calc_matrix_camera(depsgraph, x=w, y=h,
                scale_x=scene.render.pixel_aspect_x, scale_y=scene.render.pixel_aspect_y)
            transform = camera.matrix_world.normalized().inverted()
            if (max(abs(rv.view_matrix[i][j] - transform[i][j]) for i in range(4) for j in range(4)) < 1e-5 and
                max(abs(rv.window_matrix[i][j] - projection[i][j]) for i in (0, 1, 3) for j in range(4)) < 1e-5):
                if check_geometry and any(
                        ob.hide_viewport != ob.hide_render or ob.hide_get(view_layer=ctx.view_layer) or
                        any(mod.show_viewport != mod.show_render or
                            (mod.type == 'SUBSURF' and mod.levels != mod.render_levels)
                            for mod in ob.modifiers)
                        for ob in ctx.view_layer.objects):
                    continue
                return owner
    return None


def query_batches(job):
    x0, y0, width, height = job['rect']
    stride = min(width, 32)
    stripe_height = max(1, 256 // stride)
    start = 0
    while start < job['samples']:
        count = min(1 if start == 0 else 8, job['samples'] - start)
        interval = job.get('photon_interval')
        if interval:
            count = min(count, interval - start % interval)
        for y in range(0, height, stripe_height):
            for x in range(0, width, stride):
                yield (x0+x, y0+y, min(stride, width-x), min(stripe_height, height-y), start, count)
        start += count


def split_query_batch(batch):
    x, y, w, h, start, count = batch
    if count > 1:
        n = count // 2
        return (x, y, w, h, start, n), (x, y, w, h, start+n, count-n)
    if w >= h and w > 1:
        n = w // 2
        return (x, y, n, h, start, count), (x+n, y, w-n, h, start, count)
    if h > 1:
        n = h // 2
        return (x, y, w, n, start, count), (x, y+n, w, h-n, start, count)
    return None


class LiveGPUQuery:
    streams = itertools.count(1)

    def __init__(self, owner, job, scene):
        from .lpe_query_worker import QueryFilm
        self.owner = owner
        self.stream = next(self.streams)
        self.photon_builds = self.photon_reuses = 0
        self.film = QueryFilm(job, scene)
        self.ticket = None
        self.batches = query_batches(job)
        self.batch = next(self.batches)
        self.retry = []
        self.revision = None
        self.published = 0.
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='Cycles query results')
        self.fold = None
        self.wait_started = None
        self.next_try = 0.
        self.poll_delay = .01

    def close(self):
        import _cycles
        from .engine import _query_viewports
        self.ticket = None
        # A released viewport session no longer owns any query memory.
        if self.stream and any(entry[0] == self.owner for entry in _query_viewports.values()):
            _cycles.query_gpu_release(self.owner, self.stream)
        self.stream = 0
        self.executor.shutdown(wait=False, cancel_futures=True)

    def accumulate(self, data, publish, done):
        # Plain sample data and an OCIO processor created on the main thread.
        self.film.add(data)
        return self.film.packet(done) if publish else None

    def submit(self, rt, job):
        import _cycles
        if rt.paused or time.monotonic() < self.next_try:
            return None
        x, y, w, h, start, count = self.batch
        rx, ry, rw, rh = job['rect']
        last_slice = (x + w == rx + rw and y + h == ry + rh and start + count == job['samples'])
        self.ticket = _cycles.query_gpu_submit(self.owner, job['frame_size'], (x, y, w, h),
                                              start, count, job.get('total_light', False),
                                              self.stream, last_slice)
        self.poll_delay = .01
        return None if self.ticket is not None else 'Viewport session closed'

    def step(self, rt, context, job):
        import _cycles
        if compatible_viewport(context, job, check_geometry=False) != self.owner:
            return 'Viewport session closed'
        if self.fold is not None:
            if not self.fold.done():
                return None
            preview = self.fold.result()
            self.fold = None
            if preview is not None:
                rt.show_packet(*preview)
                if rt.meta.get('done'):
                    return None
        if self.ticket is None:
            return self.submit(rt, job)
        packet = _cycles.query_gpu_poll(self.ticket, True)
        if packet is None:
            self.poll_delay = min(.03, self.poll_delay * 1.5)
            return None
        self.ticket = None
        error, revision, rt.gpu_bytes, data, builds, reuses = packet
        self.photon_builds += builds
        self.photon_reuses += reuses
        if error in {'Viewport GPU scene is not resident', 'Viewport projection does not match the selection'}:
            # Progressive viewport startup can briefly use a coarser raster.
            now = time.monotonic()
            if self.wait_started is None:
                self.wait_started = now
            if error == 'Viewport GPU scene is not resident' or now - self.wait_started < 2:
                self.next_try = now + .1
                return None
        else:
            self.wait_started = None
        if error == 'GPU query event buffer full':
            split = split_query_batch(self.batch)
            if split:
                self.batch, later = split
                self.retry.append(later)
                return None
        if error or (self.revision is not None and self.revision != revision):
            return error or 'Viewport scene changed'
        self.revision = revision
        self.batch = self.retry.pop() if self.retry else next(self.batches, None)
        done = self.batch is None
        if done:
            self.stream = 0  # The last successful slice already released its workspace.
        now = time.monotonic()
        publish = done or rt.paused or now - self.published >= .4
        if publish:
            self.published = now
        self.fold = self.executor.submit(self.accumulate, data, publish, done)
        # Fold the previous tile on the CPU while the viewport schedules one more short slice.
        return None if done else self.submit(rt, job)


def redraw():
    wm = bpy.context.window_manager
    if wm:
        for window in wm.windows:
            for area in window.screen.areas:
                if area.type in {'VIEW_3D', 'IMAGE_EDITOR', 'NODE_EDITOR'}:
                    area.tag_redraw()


def target_id(window, area, region):
    return window.as_pointer(), area.as_pointer(), region.as_pointer()


def valid_source(area):
    space = area.spaces.active
    return ((area.type == 'VIEW_3D' and space.shading.type == 'RENDERED') or
            (area.type == 'IMAGE_EDITOR' and space.image and space.image.type == 'RENDER_RESULT'))


def target_under(context, event):
    for area in context.window.screen.areas:
        if not valid_source(area):
            continue
        for region in area.regions:
            x, y = event.mouse_x - region.x, event.mouse_y - region.y
            if region.type == 'WINDOW' and 0 <= x < region.width and 0 <= y < region.height:
                return area, region, (x, y)
    return None


def crop_interval(a, b, size):
    lo, hi = sorted((a * size, b * size))
    first = min(size - 1, max(0, math.floor(lo + 1e-6)))
    return first, min(size, max(first + 1, math.ceil(hi - 1e-6)))


def region_coordinate(area, region, point):
    if area.type == 'IMAGE_EDITOR':
        return tuple(region.view2d.region_to_view(*point))
    return point[0] / region.width, point[1] / region.height


def settings(context):
    return context.window_manager.cycles_contribution_query


def viewport_resolution(context, region):
    size = context.scene.render.preview_pixel_size
    if size == 'AUTO':
        divisor = 2 if context.preferences.system.pixel_size > 1.5 else 1
    else:
        divisor = int(size)
    return max(1, region.width // divisor), max(1, region.height // divisor)


def build_request(context, area, region):
    scene = context.scene
    space = area.spaces.active
    controls = settings(context)
    job = dict(scene=scene.name, layer=context.view_layer.name, samples=controls.samples,
               limit=controls.limit, device=controls.device, total_light=controls.mode == 'TOTAL')
    if scene.cycles.caustics_method == 'PHOTON':
        job['photon_interval'] = max(1, scene.cycles.photon_caustics_update_interval)
    threads = getattr(os, 'process_cpu_count', os.cpu_count)() or 1
    if scene.render.threads_mode == 'FIXED':
        threads = min(threads, scene.render.threads)
    job['threads'] = max(1, threads - 2)
    (x1, y1), (x2, y2) = _runtime.rect
    if area.type == 'IMAGE_EDITOR':
        r = scene.render
        width = max(1, r.resolution_x * r.resolution_percentage // 100)
        height = max(1, r.resolution_y * r.resolution_percentage // 100)
        if r.use_border and r.use_crop_to_border:
            bx, by = int(r.border_min_x * width), int(r.border_min_y * height)
            w = int(r.border_max_x * width) - bx
            h = int(r.border_max_y * height) - by
            x1, x2 = (bx + x1 * w) / width, (bx + x2 * w) / width
            y1, y2 = (by + y1 * h) / height, (by + y2 * h) / height
    else:
        rv = context.region_data or space.region_3d
        width, height = viewport_resolution(context, region)
        matrix = rv.window_matrix
        aspect = region.height / region.width
        persp = rv.is_perspective
        job['view'] = dict(matrix=[list(row) for row in rv.view_matrix.inverted()],
                           type='PERSP' if persp else 'ORTHO', lens=18 * matrix[0][0],
                           scale=2 / matrix[0][0],
                           shift=(.5 * matrix[0][2], .5 * matrix[1][2] * aspect) if persp else
                                 (-.5 * matrix[0][3], -.5 * matrix[1][3] * aspect),
                           clip=(space.clip_start, space.clip_end), camera_view=rv.view_perspective == 'CAMERA')
    x, xend = crop_interval(x1, x2, width)
    y, yend = crop_interval(y1, y2, height)
    job.update(frame_size=(width, height), rect=(x, y, xend - x, yend - y))
    return job


def tick():
    rt = _runtime
    if not rt or not rt.target:
        return .15
    try:
        for window in bpy.context.window_manager.windows:
            if window.as_pointer() != rt.target[0]:
                continue
            for area in window.screen.areas:
                if area.as_pointer() != rt.target[1] or not valid_source(area):
                    continue
                region = next((r for r in area.regions if r.as_pointer() == rt.target[2]), None)
                if not region:
                    continue
                with bpy.context.temp_override(window=window, area=area, region=region):
                    if rt.stopped:
                        return .15
                    job = build_request(bpy.context, area, region)
                    sig = (json.dumps(job), bpy.context.scene.frame_current_final, rt.scene_revision)
                    if sig != rt.signature:
                        if rt.gpu and compatible_viewport(bpy.context, job, check_geometry=False) != rt.gpu.owner:
                            rt.advance_gpu(bpy.context, job)
                            return .15
                        if rt.signature and (sig[1] != rt.signature[1] or job['scene'] != rt.scene):
                            rt.snapshot = None
                        rt.signature = sig
                        rt.extent = job['rect'][2:]
                        rt.restart()
                    if rt.pending and not rt.paused and time.monotonic() >= rt.pending:
                        rt.launch(bpy.context, job)
                    if rt.gpu:
                        rt.advance_gpu(bpy.context, job)
                    if rt.process:
                        rt.read_preview()
                        code = rt.process.poll()
                        if code is not None:
                            rt.read_preview()
                            rt.process = None
                            rt.message = 'Completed'
                            if code or not rt.meta.get('done'):
                                lines = (Path(rt.folder.name) / 'worker.log').read_text(
                                    encoding='utf8', errors='replace').splitlines()
                                rt.message = next((line for line in reversed(lines) if 'Error:' in line),
                                                  'CPU query did not complete')
                            redraw()
                return rt.gpu.poll_delay if rt.gpu and not rt.paused else .12
        rt.clear()
    except Exception as exc:
        rt.clear(read_latest=False)
        rt.message = str(exc)
        redraw()
    return .15


class CYCLES_PG_contribution_query(PropertyGroup):
    device: EnumProperty(name='Query Device', items=(('CPU', 'CPU', ''), ('GPU', 'GPU', 'Use a compatible NVIDIA viewport, otherwise use CPU')), default='CPU')
    samples: IntProperty(name='Query Samples', min=1, soft_max=4096, default=256)
    limit: IntProperty(name='Sampling Results', min=1, max=10, default=6)
    mode: EnumProperty(name='Query Mode', items=(
        ('EVENTS', 'Detailed Paths', 'Rank each scattering event sequence separately'),
        ('TOTAL', 'Total Light', 'Group all events on the same ordered object and material route')),
        default='EVENTS')
    detail: EnumProperty(name='Event Detail', items=(('SHORT', 'Compact', ''), ('FULL', 'Detailed', '')),
                         default='SHORT')


class CYCLES_OT_query_select_window(Operator):
    bl_idname = 'cycles.query_select_window'
    bl_label = 'Select Contribution Query Region'
    bl_options = {'INTERNAL'}

    def invoke(self, context, event):
        self.selection_token = _runtime.selection_token
        context.window.cursor_modal_set('CROSSHAIR')
        _runtime.cursor_windows.append(context.window)
        context.window_manager.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def cancel(self, context):
        if _runtime and self.selection_token is _runtime.selection_token:
            _runtime.end_selection()
            redraw()

    def modal(self, context, event):
        rt = _runtime
        if not rt or not rt.selecting or self.selection_token is not rt.selection_token:
            return {'CANCELLED'}
        if event.type in {'ESC', 'RIGHTMOUSE'}:
            rt.end_selection()
            redraw()
            return {'CANCELLED'}
        hit = target_under(context, event)
        if not rt.drag:
            if hit and event.type == 'LEFTMOUSE' and event.value == 'PRESS':
                area, region, point = hit
                rt.drag = [target_id(context.window, area, region), point, point]
                redraw()
                return {'RUNNING_MODAL'}
            return {'PASS_THROUGH'}
        target, first, _ = rt.drag
        if context.window.as_pointer() != target[0]:
            return {'PASS_THROUGH'}
        area = next((a for a in context.window.screen.areas if a.as_pointer() == target[1]), None)
        region = next((r for r in area.regions if r.as_pointer() == target[2]), None) if area else None
        if not region:
            rt.end_selection()
            return {'CANCELLED'}
        point = (min(region.width, max(0, event.mouse_x - region.x)),
                 min(region.height, max(0, event.mouse_y - region.y)))
        rt.drag[2] = point
        if event.type == 'LEFTMOUSE' and event.value == 'RELEASE':
            coords = [region_coordinate(area, region, p) for p in (first, point)]
            if any(max(p[i] for p in coords) < 0 or min(p[i] for p in coords) > 1 for i in (0, 1)):
                rt.end_selection()
                redraw()
                return {'CANCELLED'}
            rt.clear(False)
            rt.target = target
            rt.rect = [tuple(max(0., min(1., v)) for v in p) for p in coords]
            redraw()
            return {'FINISHED'}
        redraw()
        return {'RUNNING_MODAL'}


class CYCLES_OT_query_region(Operator):
    bl_idname = 'cycles.query_region'
    bl_label = 'Sampling Region'

    def invoke(self, context, event):
        if _runtime.selecting:
            _runtime.end_selection()
            redraw()
            return {'FINISHED'}
        targets = [(w, a) for w in context.window_manager.windows for a in w.screen.areas if valid_source(a)]
        if not targets:
            self.report({'INFO'}, tr('Open a rendered 3D View or Render Result to select pixels'))
            return {'CANCELLED'}
        _runtime.selecting, _runtime.drag = True, None
        _runtime.selection_token = object()
        targets.append((context.window, context.area))
        visited = set()
        for window, area in targets:
            if window.as_pointer() in visited:
                continue
            visited.add(window.as_pointer())
            region = next(r for r in area.regions if r.type == 'WINDOW')
            with context.temp_override(window=window, area=area, region=region):
                bpy.ops.cycles.query_select_window('INVOKE_DEFAULT')
        redraw()
        return {'FINISHED'}


class CYCLES_OT_query_play(Operator):
    bl_idname = 'cycles.query_play'
    bl_label = 'Start Contribution Query'

    def invoke(self, context, event):
        rt = _runtime
        if not rt.target:
            return bpy.ops.cycles.query_region('INVOKE_DEFAULT')
        return self.execute(context)

    def execute(self, context):
        rt = _runtime
        if not rt.target:
            return {'CANCELLED'}
        if rt.meta.get('done') or not (rt.pending or rt.process or rt.gpu):
            rt.paused = False
            rt.snapshot = None
            rt.restart()
        else:
            rt.paused = not rt.paused
            if rt.process and rt.request:
                gate = Path(rt.request['output']).with_suffix('.pause')
                if rt.paused:
                    gate.touch()
                else:
                    gate.unlink(missing_ok=True)
            rt.message = 'Paused' if rt.paused else 'Sampling'
        redraw()
        return {'FINISHED'}


class CYCLES_OT_query_clear(Operator):
    bl_idname = 'cycles.query_clear'
    bl_label = 'Clear Region'

    def execute(self, context):
        _runtime.clear()
        return {'FINISHED'}


def overlay():
    import gpu
    from gpu_extras.batch import batch_for_shader
    ctx = bpy.context
    rt = _runtime
    if not rt or not ctx.region or ctx.region.type != 'WINDOW':
        return
    key = target_id(ctx.window, ctx.area, ctx.region)
    if rt.drag and key == rt.drag[0]:
        (x1, y1), (x2, y2) = rt.drag[1:]
    elif rt.rect and key == rt.target:
        if ctx.area.type == 'IMAGE_EDITOR':
            (x1, y1), (x2, y2) = [ctx.region.view2d.view_to_region(*p, clip=False) for p in rt.rect]
        else:
            (x1, y1), (x2, y2) = [(u * ctx.region.width, v * ctx.region.height) for u, v in rt.rect]
    else:
        return
    pts = []
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    for x in range(max(0, int(x1)), min(ctx.region.width, int(x2)), 10):
        for y in (y1, y2):
            pts.extend(((x, y), (min(x + 6, x2), y)))
    for y in range(max(0, int(y1)), min(ctx.region.height, int(y2)), 10):
        for x in (x1, x2):
            pts.extend(((x, y), (x, min(y + 6, y2))))
    if pts:
        shader = gpu.shader.from_builtin('UNIFORM_COLOR')
        batch = batch_for_shader(shader, 'LINES', {'pos': pts})
        shader.bind()
        shader.uniform_float('color', (.15, 1., .25, 1.))
        batch.draw(shader)

def item_socket(node, index=0):
    identifier = 'Item_%d' % node.path_items[index].identifier
    return next(sock for sock in node.inputs if sock.identifier == identifier)


def resolve(data, name):
    matches = [item for item in data if item.name == name]
    if len(matches) != 1:
        raise ValueError(tr('Missing or ambiguous datablock') + ': ' + name)
    return matches[0]


def unique_output_name(tree, base):
    used = {socket.name for node in tree.nodes if node.bl_idname == 'LPENodeOutput'
            for socket in node.inputs if socket.identifier.startswith('Item_')}
    # bNodeSocket.name stores 63 UTF-8 bytes plus its terminator.
    for index in itertools.count():
        suffix = f'.{index:03d}' if index else ''
        name = base.encode('utf8')[:63 - len(suffix)].decode('utf8', 'ignore') + suffix
        if name not in used:
            return name


def write_route(layer, row, material):
    specs = []
    for i, (kind, obj, mat, event, instance) in enumerate(row['path']):
        caster = bool(event & 256)
        if i == 0 and kind == 'WORLD':
            specs.append(('LPENodeWorld', None, 'Photon' if row['photon'] else 'Total Light'))
        elif i == 0 and kind == 'LIGHT':
            specs.append(('LPENodeLight', resolve(bpy.data.objects, obj),
                          'Photon' if row['photon'] else 'Total Light'))
        else:
            data = resolve(bpy.data.materials, mat) if material else resolve(bpy.data.objects, obj)
            output = ('Emission' if i == 0 else 'Total Light' if row.get('total_light') else
                      'Caustics' if caster else EVENTS[event])
            spec = ('LPENodeMaterial' if material else 'LPENodeObject', data, output)
            # One Caustics selector accepts successive entry/exit interfaces of this caster.
            if not (caster and specs and spec == specs[-1]):
                specs.append(spec)
    tree = layer.lpe_node_tree
    fresh = tree is None
    if fresh:
        tree = bpy.data.node_groups.new('Light Paths', 'LPENodeTree')
        layer.lpe_node_tree = tree
    made = []
    try:
        if not tree.is_editable:
            raise ValueError(tr('The current LPE tree is read-only'))
        y = min((n.location.y - max(n.dimensions.y, 140) for n in tree.nodes), default=0) - 64
        socket = None
        for i, (node_type, data, output) in enumerate(specs):
            node = tree.nodes.new(node_type)
            made.append(node)
            node.location = (i * 215, y)
            if data is not None:
                item_socket(node).default_value = data
            if socket is not None:
                tree.links.new(socket, node.inputs['Light Path'])
            socket = node.outputs[output]
        target = tree.nodes.new('LPENodeOutput')
        made.append(target)
        target.location = (len(specs) * 215, y)
        base = target.name + ' ' + (specs[0][1].name if specs[0][1] else 'World')
        target.path_items[0].name = unique_output_name(tree, base)
        tree.links.new(socket, item_socket(target))
        if tree.state_error or (tree.state_count > tree.state_limit):
            raise ValueError(tree.state_error or tr('LPE state budget exceeded'))
    except Exception:
        if fresh:
            layer.lpe_node_tree = None
            bpy.data.node_groups.remove(tree)
        else:
            for node in reversed(made):
                tree.nodes.remove(node)
        raise


class CYCLES_OT_query_add(Operator):
    bl_idname = 'cycles.query_add'
    bl_label = 'Add Light Path'
    bl_options = {'UNDO'}
    index: IntProperty()
    material: BoolProperty()
    version: IntProperty()
    serial: IntProperty()
    route: StringProperty(options={'HIDDEN', 'SKIP_SAVE'})

    def execute(self, context):
        rt = _runtime
        if (context.scene.name != rt.scene or context.view_layer.name != rt.layer or
                self.serial != rt.serial or not rt.meta or
                (not self.route and (self.version != rt.meta.get('version') or
                                     not 0 <= self.index < len(rt.meta.get('rows', []))))):
            self.report({'INFO'}, tr('Query result changed'))
            return {'CANCELLED'}
        try:
            # The rank and preview version can change between drawing the button and clicking it.
            row = json.loads(self.route) if self.route else rt.meta['rows'][self.index]
            write_route(context.view_layer, row, self.material)
        except (KeyError, RuntimeError, ValueError) as error:
            self.report({'WARNING'}, str(error))
            return {'CANCELLED'}
        return {'FINISHED'}


class QueryPanel:
    bl_region_type = 'UI'
    bl_category = 'Contribution Query'
    bl_label = 'Contribution Query'

    def draw(self, context):
        rt = _runtime
        props = settings(context)
        col = self.layout.column()
        tools = col.row(align=True)
        tools.operator('cycles.query_region', icon='SELECT_SET', depress=rt.selecting)
        tools.operator('cycles.query_clear', icon='X')
        if rt.paused:
            col.operator('cycles.query_play', text='Resume', icon='PLAY')
        elif rt.target and (rt.process or rt.pending or rt.gpu) and not rt.meta.get('done'):
            col.operator('cycles.query_play', text='Pause', icon='PAUSE')
        else:
            col.operator('cycles.query_play', text='Start', icon='PLAY')
        col.row(align=True).prop(props, 'device', expand=True)
        gpu_status = detected_gpu_ipr(context)
        if gpu_status:
            col.label(text='IPR detected. GPU sampling is available in this viewport.', icon='INFO')
            if props.device == 'GPU' and gpu_status == 2:  # Session::GPUQueryStatus::READY_WITH_PHOTONS
                col.label(text='Photon contribution queries can be slower on GPU than CPU.', icon='INFO')
        if rt.backend:
            info = tr(rt.backend)
            if rt.backend == 'GPU' and rt.gpu_bytes:
                info += ' · ' + (tr('Estimated extra GPU memory: %.1f MiB') % (rt.gpu_bytes / 1048576))
            col.label(text=info, translate=False)
            if rt.reason:
                col.label(text=tr(rt.reason), translate=False)
        col.prop(props, 'samples')
        col.prop(props, 'limit')
        col.row(align=True).prop(props, 'mode', expand=True)
        if props.mode == 'EVENTS':
            col.row(align=True).prop(props, 'detail', expand=True)
        if rt.selecting:
            col.label(text='Drag a box in the rendered view or Render Result', icon='EYEDROPPER')
        if rt.target:
            w, h = rt.extent
            col.label(text=f'{w} × {h}  ({w*h})', translate=False)
            if w*h > 1024:
                col.label(text='Over 1024 pixels. CPU evaluation may be slow.', icon='INFO')
        if rt.message:
            count = rt.meta.get('samples')
            suffix = (' · %.1f spp' % count[0]) if count else ''
            if count and count[1]:
                suffix += ' · %.1f %s' % (count[1], tr('Photon Samples'))
            col.label(text=tr(rt.message) + suffix, translate=False)
        if not rt.icons:
            return
        col.label(text='Local Render')
        col.template_icon(icon_value=rt.icons[0], scale=6)
        for i, row in enumerate(rt.meta['rows']):
            box = col.box()
            line = box.split(factor=.5)
            preview = line.column()
            unit = 20 * max(1., context.preferences.system.ui_scale)
            scale = context.region.width / (2 * unit)
            preview.template_icon(icon_value=rt.icons[i+1], scale=max(1, scale - 1))
            preview.label(text='%.2f%%' % (row['share'] * 100), translate=False)
            paths = line.split(factor=.5)
            for use_material in (False, True):
                part = paths.column()
                part.label(text='Material Path' if use_material else 'Object Path')
                for j, (kind, obj, mat, event, instance) in enumerate(row['path']):
                    name = obj if not use_material or kind in {'LIGHT', 'WORLD'} else mat
                    label = name or tr('Unknown')
                    if props.detail == 'FULL' and event:
                        family = tr(EVENTS.get(event & 255, 'Unknown'))
                        if event & 256:
                            family = tr('Caustics') + ' · ' + family
                        label += ' (' + family + ')'
                    part.label(text=label, translate=False)
                op = part.operator('cycles.query_add',
                                   text='Add Material Path' if use_material else 'Add Object Path')
                op.index, op.material = i, use_material
                op.version, op.serial = rt.meta['version'], rt.serial
                op.route = json.dumps(dict(path=row['path'], photon=row['photon'],
                                           total_light=row.get('total_light', False)), ensure_ascii=False)


class CYCLES_PT_query_node(QueryPanel, Panel):
    bl_space_type = 'NODE_EDITOR'

    @classmethod
    def poll(cls, context):
        return context.scene.render.engine == 'CYCLES' and context.space_data.tree_type == 'LPENodeTree'


class CYCLES_PT_query_image(QueryPanel, Panel):
    bl_space_type = 'IMAGE_EDITOR'

    @classmethod
    def poll(cls, context):
        return context.scene.render.engine == 'CYCLES'


classes = (CYCLES_PG_contribution_query, CYCLES_OT_query_select_window, CYCLES_OT_query_region,
           CYCLES_OT_query_play, CYCLES_OT_query_clear, CYCLES_OT_query_add, CYCLES_PT_query_node, CYCLES_PT_query_image)


@persistent
def invalidate_query_scene(scene, depsgraph):
    rt = _runtime
    if (rt is None or not rt.target or scene.original.session_uid != rt.scene_uid or
            depsgraph.mode != 'VIEWPORT'):
        return
    # Selection alone tags the scene without any transform, geometry or shading change.
    # Read the evaluated graph here; saving/stopping/restarting stays on the debounced timer.
    if any(u.is_updated_transform or u.is_updated_geometry or u.is_updated_shading
           for u in depsgraph.updates):
        rt.snapshot = None
        if rt.backend != 'GPU':
            rt.scene_revision += 1


@persistent
def load_pre(_):
    if _runtime:
        _runtime.clear(False)


def register():
    global _runtime, _previews
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.WindowManager.cycles_contribution_query = PointerProperty(type=CYCLES_PG_contribution_query)
    _previews = bpy.utils.previews.new()
    _runtime = Query()
    if not bpy.app.background:
        for space in (bpy.types.SpaceView3D, bpy.types.SpaceImageEditor):
            _handlers.append((space, space.draw_handler_add(overlay, (), 'WINDOW', 'POST_PIXEL')))
        bpy.app.timers.register(tick, first_interval=.15, persistent=True)
        bpy.app.handlers.load_pre.append(load_pre)
        bpy.app.handlers.depsgraph_update_post.append(invalidate_query_scene)
        atexit.register(_runtime.stop)


def unregister():
    global _runtime, _previews
    if bpy.app.timers.is_registered(tick):
        bpy.app.timers.unregister(tick)
    if invalidate_query_scene in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.remove(invalidate_query_scene)
    if load_pre in bpy.app.handlers.load_pre:
        bpy.app.handlers.load_pre.remove(load_pre)
    for space, handler in _handlers:
        space.draw_handler_remove(handler, 'WINDOW')
    _handlers.clear()
    atexit.unregister(_runtime.stop)
    _runtime.clear(False)
    _runtime = None
    bpy.utils.previews.remove(_previews)
    _previews = None
    del bpy.types.WindowManager.cycles_contribution_query
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
