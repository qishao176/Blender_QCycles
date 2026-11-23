# SPDX-License-Identifier: Apache-2.0

"""Viewport-only outline of the exact cone used by World photon emission.

No RNA writes or image downloads occur in drawing. The render engine provides a
non-blocking copy of the host kernel data; final rendering never calls this code.
"""

import math
import bpy
from bpy.app.handlers import persistent

_handler = None
_regions = {}


def _world(context):
    return getattr(context.view_layer, "world_override", None) or context.scene.world


def _key(context, world):
    cw = world.cycles
    return (context.scene.as_pointer(), world.as_pointer(), cw.photon_sun_threshold,
            cw.photon_sun_radius_scale)


def update(engine, context):
    world = _world(context)
    if world is None or context.region is None or not world.cycles.show_photon_sun:
        return
    import _cycles
    region = _cycles.photon_world_region(engine.session)
    if region is not None:
        _regions[context.region.as_pointer()] = (_key(context, world), region)


def cone_points(direction, angle, origin, distance, segments=128):
    """Sample the angular cone boundary, independent of the HDR's mapping nodes."""
    from mathutils import Vector
    axis = Vector(direction).normalized()
    tangent = axis.orthogonal().normalized()
    bitangent = axis.cross(tangent)
    return [origin + distance * (math.cos(angle) * axis + math.sin(angle) *
            (math.cos(i * math.tau / segments) * tangent +
             math.sin(i * math.tau / segments) * bitangent))
            for i in range(segments + 1)]


def draw():
    context = bpy.context
    space = context.space_data
    rv3d = context.region_data
    if (not space or space.type != 'VIEW_3D' or not rv3d or not rv3d.is_perspective or
            space.shading.type != 'RENDERED' or not space.overlay.show_overlays or
            not space.shading.use_scene_world_render or
            context.scene.render.engine != 'CYCLES' or
            context.scene.cycles.caustics_method != 'PHOTON'):
        return
    world = _world(context)
    if not world:
        return
    cw = world.cycles
    if not (cw.show_photon_sun and cw.is_caustics_light and cw.visible_photon):
        return
    cached = _regions.get(context.region.as_pointer())
    if not cached or cached[0] != _key(context, world):
        return
    region = cached[1]
    if region[3] <= 0.0:
        return
    import gpu
    from gpu_extras.batch import batch_for_shader
    origin = rv3d.view_matrix.inverted().translation
    distance = max(space.clip_start * 2.0, min(1000.0, space.clip_end * 0.5))
    positions = cone_points(region[:3], region[3], origin, distance)
    shader = gpu.shader.from_builtin('UNIFORM_COLOR')
    batch = batch_for_shader(shader, 'LINE_STRIP', {"pos": positions})
    depth = gpu.state.depth_test_get()
    try:
        gpu.state.depth_test_set('NONE')
        shader.bind()
        shader.uniform_float("color", (1.0, 0.03, 0.03, 1.0))
        batch.draw(shader)
    finally:
        gpu.state.depth_test_set(depth)


@persistent
def clear(_unused=None):
    _regions.clear()


def register():
    global _handler
    if not bpy.app.background and _handler is None:
        _handler = bpy.types.SpaceView3D.draw_handler_add(draw, (), 'WINDOW', 'POST_VIEW')
    bpy.app.handlers.load_pre.append(clear)


def unregister():
    global _handler
    if _handler is not None:
        bpy.types.SpaceView3D.draw_handler_remove(_handler, 'WINDOW')
        _handler = None
    if clear in bpy.app.handlers.load_pre:
        bpy.app.handlers.load_pre.remove(clear)
    clear()
