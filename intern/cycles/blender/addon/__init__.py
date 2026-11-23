# SPDX-FileCopyrightText: 2011-2022 Blender Foundation
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

bl_info = {
    "name": "Cycles Render Engine",
    "author": "",
    "blender": (2, 80, 0),
    "description": "Cycles renderer integration",
    "warning": "",
    "doc_url": "https://docs.blender.org/manual/en/latest/render/cycles/",
    "tracker_url": "",
    "support": 'OFFICIAL',
    "category": "Render"}

# Support 'reload' case.
if "bpy" in locals():
    import importlib
    if "engine" in locals():
        importlib.reload(engine)
    if "version_update" in locals():
        importlib.reload(version_update)
    if "ui" in locals():
        importlib.reload(ui)
    if "operators" in locals():
        importlib.reload(operators)
    if "properties" in locals():
        importlib.reload(properties)
    if "translations" in locals():
        importlib.reload(translations)
    if "presets" in locals():
        importlib.reload(presets)
    if "maketx" in locals():
        importlib.reload(maketx)

import bpy

from . import (
    engine,
    version_update,
)


class CyclesRender(bpy.types.RenderEngine):
    bl_idname = 'CYCLES'
    bl_label = "Cycles"
    bl_use_eevee_viewport = True
    bl_use_preview = True
    bl_use_exclude_layers = True
    bl_use_spherical_stereo = True
    bl_use_custom_freestyle = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.session = None

    def __del__(self):
        engine.free(self)

    # final render
    def update(self, data, depsgraph):
        scene = getattr(depsgraph, "scene", None)
        if scene is not None:
            for view_layer in scene.view_layers:
                view_layer.update_render_passes()
            engine.prune_stale_lightgroup_split_file_outputs(scene)

        if not self.session:
            if self.is_preview:
                cscene = bpy.context.scene.cycles
                use_osl = cscene.shading_system

                engine.create(self, data, preview_osl=use_osl)
            else:
                engine.create(self, data)

        engine.reset(self, data, depsgraph)

    def render(self, depsgraph):
        engine.render(self, depsgraph)

    def render_frame_finish(self):
        engine.render_frame_finish(self)

    def draw(self, context, depsgraph):
        engine.draw(self, depsgraph, context.space_data)

    def bake(self, depsgraph, obj, pass_type, pass_filter, width, height):
        engine.bake(self, depsgraph, obj, pass_type, pass_filter, width, height)

    # viewport render
    def view_update(self, context, depsgraph):
        if engine.lpe_viewport_error_pending(context):
            return
        if not self.session:
            engine.create(self, context.blend_data,
                          context.region, context.space_data, context.region_data)

        engine.register_query_viewport(self, context)
        from . import viewport
        viewport.sync_viewer(context)
        engine.reset(self, context.blend_data, depsgraph)
        error = engine.sync(self, depsgraph, context.blend_data)
        engine.handle_lpe_viewport_error(self, context, error)

    def view_draw(self, context, depsgraph):
        if engine.lpe_viewport_error_pending(context) or not self.session:
            return
        from . import viewport
        viewport.sync_viewer(context)
        engine.register_query_viewport(self, context)
        error = engine.view_draw(self, depsgraph, context.region, context.space_data, context.region_data)
        engine.handle_lpe_viewport_error(self, context, error)
        from . import world_photon_overlay
        world_photon_overlay.update(self, context)

    def view_pause(self, context):
        engine.view_pause(self, True)

    def view_resume(self, context):
        engine.view_pause(self, False)

    def update_script_node(self, node):
        if engine.with_osl():
            from . import osl
            osl.update_script_node(node, self.report)
        else:
            self.report({'ERROR'}, "OSL support disabled in this build")

    def update_custom_camera(self, cam):
        if engine.with_osl():
            from . import osl
            osl.update_custom_camera_shader(cam, self.report)
        else:
            self.report({'ERROR'}, "OSL support disabled in this build")

    def update_render_passes(self, scene, srl):
        engine.register_passes(self, scene, srl)


def engine_exit():
    engine.exit()


classes = (
    CyclesRender,
)


cli_commands = []


def register():
    from bpy.utils import register_class
    from . import ui
    from . import operators
    from . import properties
    from . import translations
    from . import presets
    from .maketx import maketx_command
    import atexit

    # Make sure we only registered the callback once.
    atexit.unregister(engine_exit)
    atexit.register(engine_exit)

    engine.init()

    translations.register()
    properties.register()
    from . import shader_cache
    shader_cache.configure()
    import _cycles
    for name in ('display_pass', 'lpe_display_output', 'lpe_overlay', 'lpe_overlay_output',
                 'lpe_viewer_signature', 'lpe_previous_display'):
        _cycles.configure_lpe_controls(properties.CyclesView3DShadingSettings.bl_rna.properties[name].as_pointer())
    ui.register()
    operators.register()
    presets.register()
    from . import world_photon_overlay
    world_photon_overlay.register()
    from . import lpe_query
    lpe_query.register()

    for cls in classes:
        register_class(cls)

    bpy.app.handlers.version_update.append(version_update.do_versions)

    cli_commands.append(bpy.utils.register_cli_command("maketx", maketx_command))


def unregister():
    from bpy.utils import unregister_class
    from . import ui
    from . import operators
    from . import properties
    from . import translations
    from . import presets

    bpy.app.handlers.version_update.remove(version_update.do_versions)

    for cmd in cli_commands:
        bpy.utils.unregister_cli_command(cmd)
    cli_commands.clear()

    ui.unregister()
    from . import world_photon_overlay
    world_photon_overlay.unregister()
    from . import lpe_query
    lpe_query.unregister()
    operators.unregister()
    properties.unregister()
    presets.unregister()
    translations.unregister()

    for cls in classes:
        unregister_class(cls)
