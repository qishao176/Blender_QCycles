# SPDX-FileCopyrightText: 2011-2022 Blender Foundation
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations


def _configure_argument_parser():
    import argparse
    # No help because it conflicts with general Python scripts argument parsing
    parser = argparse.ArgumentParser(description="Cycles Addon argument parser",
                                     add_help=False)
    parser.add_argument("--cycles-print-stats",
                        help="Print rendering statistics to stderr",
                        action='store_true')
    parser.add_argument("--cycles-device",
                        help="Set the device to use for Cycles, overriding user preferences and the scene setting."
                             "Valid options are 'CPU', 'CUDA', 'OPTIX', 'HIP', 'ONEAPI', or 'METAL'."
                             "Additionally, you can append '+CPU' to any GPU type for hybrid rendering.",
                        default=None)
    return parser


def _parse_command_line():
    import sys

    argv = sys.argv
    if "--" not in argv:
        return

    parser = _configure_argument_parser()
    args, _ = parser.parse_known_args(argv[argv.index("--") + 1:])

    if args.cycles_print_stats:
        import _cycles
        _cycles.enable_print_stats()

    if args.cycles_device:
        import _cycles
        if not _cycles.set_device_override(args.cycles_device):
            sys.exit(1)


def init():
    import bpy
    import _cycles
    import os.path

    path = os.path.dirname(__file__)
    user_path = os.path.dirname(os.path.abspath(bpy.utils.user_resource('CONFIG', path='')))

    _cycles.init(path, user_path, bpy.app.background)
    _parse_command_line()


def exit():
    import _cycles
    _cycles.exit()


def create(engine, data, region=None, v3d=None, rv3d=None, preview_osl=False):
    import _cycles
    import bpy

    data = data.as_pointer()
    prefs = bpy.context.preferences.as_pointer()
    screen = 0
    if region:
        screen = region.id_data.as_pointer()
        region = region.as_pointer()
    if v3d:
        screen = screen or v3d.id_data.as_pointer()
        v3d = v3d.as_pointer()
    if rv3d:
        screen = screen or rv3d.id_data.as_pointer()
        rv3d = rv3d.as_pointer()

    engine.session = _cycles.create(engine.as_pointer(), prefs, data, screen, region, v3d, rv3d, preview_osl)


_query_viewports = {}
_photon_probes = {}


def register_query_viewport(engine, context):
    if context.window and context.area and context.region and context.region.type == 'WINDOW':
        key = context.window.as_pointer(), context.area.as_pointer(), context.region.as_pointer()
        import _cycles
        probe = _cycles.photon_viewport_probe(engine.session)
        if _photon_probes.get(key) != probe:
            _photon_probes[key] = probe
            for area in context.window.screen.areas:
                if area.type == 'PROPERTIES':
                    area.tag_redraw()
            for region in context.area.regions:
                if region.type == 'UI':
                    region.tag_redraw()
        entry = engine.session, context.scene.session_uid, context.view_layer.name, _cycles.query_gpu_ready(engine.session)
        if _query_viewports.get(key) != entry:
            _query_viewports[key] = entry
            from . import lpe_query
            lpe_query.redraw()


def free(engine):
    if hasattr(engine, "session"):
        if engine.session:
            import _cycles
            for key, entry in tuple(_query_viewports.items()):
                if entry[0] == engine.session:
                    del _query_viewports[key]
                    if _photon_probes.pop(key, None):
                        import bpy
                        wm = bpy.context.window_manager
                        for window in wm.windows if wm else ():
                            if window.as_pointer() == key[0]:
                                for area in window.screen.areas:
                                    if area.type == 'PROPERTIES':
                                        area.tag_redraw()
                    from . import lpe_query
                    lpe_query.redraw()
            _cycles.free(engine.session)
        del engine.session


def render(engine, depsgraph):
    import _cycles
    if hasattr(engine, "session"):
        evaluated = depsgraph.scene_eval
        if use_vcm(evaluated) and not engine.is_preview:
            excluded = [name for name, _, supported in vcm_lpe_outputs(depsgraph.view_layer_eval)
                        if not supported]
            if excluded:
                from bpy.app.translations import pgettext_rpt as rpt_
                engine.report({'WARNING'}, rpt_("Unsupported VCM light paths: %s") % ", ".join(excluded))
        if (evaluated.cycles.integrator_method == 'VCM' and not use_vcm(evaluated) and
                use_vcm(depsgraph.scene)):
            # Blender registers passes before evaluating geometry nodes and layer overrides.
            # Complete the layer's pass list if evaluation revealed a native-only material.
            layer = depsgraph.view_layer_eval
            for name, channels, _ in list_render_passes(evaluated, layer):
                engine.add_pass(name, len(channels), channels, layer=layer.name)
        _cycles.render(engine.session, depsgraph.as_pointer())


def render_frame_finish(engine):
    if not engine.session:
        return

    import _cycles
    _cycles.render_frame_finish(engine.session)


def draw(engine, depsgraph, space_image):
    if not engine.session:
        return

    depsgraph_ptr = depsgraph.as_pointer()
    space_image_ptr = space_image.as_pointer()
    screen_ptr = space_image.id_data.as_pointer()

    import _cycles
    _cycles.draw(engine.session, depsgraph_ptr, screen_ptr, space_image_ptr)


def bake(engine, depsgraph, obj, pass_type, pass_filter, width, height):
    import _cycles
    session = getattr(engine, "session", None)
    if session is not None:
        _cycles.bake(engine.session, depsgraph.as_pointer(), obj.as_pointer(), pass_type, pass_filter, width, height)


def reset(engine, data, depsgraph):
    import _cycles
    import bpy

    prefs = bpy.context.preferences
    if prefs.experimental.use_cycles_debug and prefs.view.show_developer_ui:
        _cycles.debug_flags_update(depsgraph.scene.as_pointer())
    else:
        _cycles.debug_flags_reset()

    data = data.as_pointer()
    depsgraph = depsgraph.as_pointer()
    _cycles.reset(engine.session, data, depsgraph)


def sync(engine, depsgraph, data):
    import _cycles
    return _cycles.sync(engine.session, depsgraph.as_pointer())


def view_draw(engine, depsgraph, region, v3d, rv3d):
    import _cycles
    depsgraph = depsgraph.as_pointer()
    v3d = v3d.as_pointer()
    rv3d = rv3d.as_pointer()

    # draw render image
    return _cycles.view_draw(engine.session, depsgraph, v3d, rv3d)


# Leave render-engine callbacks before changing shading: the RNA setter can
# free the current render engine, including its Cycles session and GPU context.
_lpe_viewport_fallbacks = set()


def _viewport_scope(context):
    return (context.window.as_pointer(), context.area.as_pointer(),
            context.space_data.as_pointer())


def lpe_viewport_error_pending(context):
    return _viewport_scope(context) in _lpe_viewport_fallbacks


def handle_lpe_viewport_error(engine, context, error):
    if not isinstance(error, str) or not error.startswith('LPE:'):
        return
    import bpy
    from bpy.app.translations import pgettext_iface as iface_

    scope = _viewport_scope(context)
    if scope in _lpe_viewport_fallbacks:
        return
    _lpe_viewport_fallbacks.add(scope)
    engine.report({'ERROR'}, error)
    scene_uid = context.scene.session_uid
    layer_name = context.view_layer.name

    def fallback():
        try:
            # Resolve live UI data again: the user may have closed the area,
            # changed scenes/workspaces, or left rendered mode in the meantime.
            # Do not retain the RenderEngine or dereference old RNA pointers.
            for window in bpy.context.window_manager.windows:
                if (window.as_pointer() != scope[0] or window.scene.session_uid != scene_uid or
                        window.view_layer.name != layer_name):
                    continue
                for area in window.screen.areas:
                    if area.as_pointer() != scope[1] or area.type != 'VIEW_3D':
                        continue
                    space = area.spaces.active
                    if space.as_pointer() != scope[2] or space.shading.type != 'RENDERED':
                        continue
                    space.shading.type = 'SOLID'
                    area.tag_redraw()

                    def draw_error(self, _context):
                        import textwrap
                        self.layout.label(text=iface_("LPE preview stopped; switched to Solid shading"), icon='ERROR')
                        for line in textwrap.wrap(error, width=100):
                            self.layout.label(text=line, translate=False)

                    with bpy.context.temp_override(window=window, area=area):
                        bpy.context.window_manager.popup_menu(draw_error, title=iface_("LPE Error"), icon='ERROR')
                    return None
        finally:
            _lpe_viewport_fallbacks.discard(scope)
        return None

    bpy.app.timers.register(fallback, first_interval=0.0)


def view_pause(engine, pause):
    import _cycles
    if engine.session:
        _cycles.view_pause(engine.session, pause)


def available_devices():
    import _cycles
    return _cycles.available_devices()


def with_osl():
    import _cycles
    return _cycles.with_osl


def osl_version():
    import _cycles
    return _cycles.osl_version


def with_path_guiding():
    import _cycles
    return _cycles.with_path_guiding


def system_info():
    import _cycles
    return _cycles.system_info()


_LIGHTGROUP_SPLIT_PASS_DESCRIPTORS = (
    ("use_pass_diffuse_combined", "Diff"),
    ("use_pass_diffuse_direct", "Diff_Dir"),
    ("use_pass_diffuse_indirect", "Diff_Ind"),
    ("use_pass_glossy_combined", "Glos"),
    ("use_pass_glossy_direct", "Glos_Dir"),
    ("use_pass_glossy_indirect", "Glos_Ind"),
    ("use_pass_transmission_combined", "Trans"),
    ("use_pass_transmission_direct", "Trans_Dir"),
    ("use_pass_transmission_indirect", "Trans_Ind"),
    ("use_pass_subsurface_combined", "SSS"),
    ("use_pass_subsurface_direct", "SSS_Dir"),
    ("use_pass_subsurface_indirect", "SSS_Ind"),
    ("use_pass_volume_combined", "Vol"),
    ("use_pass_volume_direct", "Vol_Dir"),
    ("use_pass_volume_indirect", "Vol_Ind"),
    ("use_pass_photon", "Photon"),
)
_LIGHTGROUP_SPLIT_PASS_SUFFIXES = tuple(
    descriptor[1] for descriptor in _LIGHTGROUP_SPLIT_PASS_DESCRIPTORS
)


def _is_lightgroup_split_pass_name(name):
    return name.startswith("lg_") and any("_" + suffix in name for suffix in _LIGHTGROUP_SPLIT_PASS_SUFFIXES)


_LIGHTGROUP_GENERATED_FILE_OUTPUT_ITEMS_PROP = "cycles_lightgroup_split_generated_items"
_RENDER_PASS_NAME_MAX_UTF8_BYTES = 63


def _lightgroup_generated_file_output_item_names(node):
    value = node.get(_LIGHTGROUP_GENERATED_FILE_OUTPUT_ITEMS_PROP, ())
    if isinstance(value, str):
        return {value} if value else set()

    try:
        return {str(name) for name in value if name}
    except TypeError:
        return set()


def _set_lightgroup_generated_file_output_item_names(node, item_names):
    if item_names:
        node[_LIGHTGROUP_GENERATED_FILE_OUTPUT_ITEMS_PROP] = sorted(item_names)
    elif _LIGHTGROUP_GENERATED_FILE_OUTPUT_ITEMS_PROP in node:
        del node[_LIGHTGROUP_GENERATED_FILE_OUTPUT_ITEMS_PROP]


def _lightgroup_pass_name(prefix, lightgroup_index, lightgroup_name, used_names):
    combined = prefix == "Combined_"
    preferred_name = prefix + lightgroup_name if combined else f"lg_{lightgroup_name}_{prefix}"
    if (
        len(preferred_name.encode("utf-8")) <= _RENDER_PASS_NAME_MAX_UTF8_BYTES
        and preferred_name not in used_names
    ):
        pass_name = preferred_name
    else:
        fallback_base = (f"{prefix}__lg{lightgroup_index}" if combined else
                         f"lg___lg{lightgroup_index}_{prefix}")
        pass_name = fallback_base
        suffix = 1
        while pass_name in used_names:
            pass_name = f"{fallback_base}_{suffix}"
            suffix += 1

    used_names.add(pass_name)
    return pass_name


def vcm_fallback_reason(scene, view_layer=None):
    import _cycles
    if view_layer is not None:
        graph = view_layer.depsgraph
        if graph and graph.scene_eval:
            scene = graph.scene_eval
    return _cycles.vcm_fallback_reason(scene.as_pointer())


def use_vcm(scene, view_layer=None):
    return scene.cycles.integrator_method == 'VCM' and not vcm_fallback_reason(scene, view_layer)


def vcm_lpe_outputs(layer):
    tree = layer.lpe_node_tree
    if tree is None:
        return
    for link in tree.links:
        node, socket = link.to_node, link.to_socket
        viewer = node.bl_idname == 'LPENodeViewer' and node.active
        if (link.is_muted or node.mute or socket.type not in {'RGBA', 'VALUE'} or
                (not viewer and node.bl_idname != 'LPENodeOutput')):
            continue
        supported = (link.from_node.bl_idname in {'LPENodeLight', 'LPENodeWorld'} and
                     link.from_socket.identifier == 'Total Light')
        yield ("LPE Viewer" if viewer else "LPE_" + socket.name, socket.type, supported)


def list_render_passes(scene, srl):
    import _cycles
    from bpy.app.translations import pgettext_n as n_

    crl = srl.cycles

    # Combined pass.
    yield ("Combined", "RGBA", 'COLOR')
    if use_vcm(scene):
        if srl.use_pass_emit:
            yield (n_("Emission"), "RGB", 'COLOR')
        if srl.use_pass_environment:
            yield (n_("Environment"), "RGB", 'COLOR')
        if scene.cycles.use_denoising and crl.use_denoising:
            yield (n_("Noisy Image"), "RGBA", 'COLOR')
        if crl.pass_debug_sample_count:
            yield (n_("Debug Sample Count"), "X", 'VALUE')
        for name, kind, supported in vcm_lpe_outputs(srl):
            if supported:
                yield (name, "X" if kind == 'VALUE' else "RGB", kind if kind == 'VALUE' else 'COLOR')
        used = set()
        for index, group in enumerate(srl.lightgroups):
            yield (_lightgroup_pass_name("Combined_", index, group.name, used), "RGB", 'COLOR')
        return

    # Keep alignment for readability.
    # autopep8: off

    # Data passes.
    if srl.use_pass_z:                     yield (n_("Depth"),           "Z",    'VALUE')
    if srl.use_pass_mist:                  yield (n_("Mist"),            "Z",    'VALUE')
    if srl.use_pass_position:              yield (n_("Position"),        "XYZ",  'VECTOR')
    if srl.use_pass_normal:                yield (n_("Normal"),          "XYZ",  'VECTOR')
    if srl.use_pass_vector:                yield (n_("Vector"),          "XYZW", 'VECTOR')
    if srl.use_pass_uv:                    yield (n_("UV"),              "UVA",  'VECTOR')
    if srl.use_pass_object_index:          yield (n_("Object Index"),    "X",    'VALUE')
    if srl.use_pass_material_index:        yield (n_("Material Index"),  "X",    'VALUE')

    # Light passes.
    if crl.use_pass_caustics:              yield (n_("Caustics"),              "RGB",  'COLOR')
    if srl.use_pass_diffuse_direct:        yield (n_("Diffuse Direct"),        "RGB",  'COLOR')
    if srl.use_pass_diffuse_indirect:      yield (n_("Diffuse Indirect"),      "RGB",  'COLOR')
    if srl.use_pass_diffuse_color:         yield (n_("Diffuse Color"),         "RGB",  'COLOR')
    if srl.use_pass_subsurface_direct:     yield (n_("SSS Direct"),            "RGB",  'COLOR')
    if srl.use_pass_subsurface_indirect:   yield (n_("SSS Indirect"),          "RGB",  'COLOR')
    if srl.use_pass_subsurface_color:      yield (n_("SSS Color"),             "RGB",  'COLOR')
    if srl.use_pass_glossy_direct:         yield (n_("Glossy Direct"),         "RGB",  'COLOR')
    if srl.use_pass_glossy_indirect:       yield (n_("Glossy Indirect"),       "RGB",  'COLOR')
    if srl.use_pass_glossy_color:          yield (n_("Glossy Color"),          "RGB",  'COLOR')
    if srl.use_pass_transmission_direct:   yield (n_("Transmission Direct"),   "RGB",  'COLOR')
    if srl.use_pass_transmission_indirect: yield (n_("Transmission Indirect"), "RGB",  'COLOR')
    if srl.use_pass_transmission_color:    yield (n_("Transmission Color"),    "RGB",  'COLOR')
    if crl.use_pass_volume_direct:         yield (n_("Volume Direct"),         "RGB",  'COLOR')
    if crl.use_pass_volume_indirect:       yield (n_("Volume Indirect"),       "RGB",  'COLOR')
    if crl.use_pass_volume_scatter:        yield (n_("Volume Scatter"),        "RGB",  'COLOR')
    if crl.use_pass_volume_transmit:       yield (n_("Volume Transmit"),       "RGB",  'COLOR')
    if crl.use_pass_volume_majorant:       yield (n_("Volume Majorant"),       "Z",    'VALUE')
    if srl.use_pass_emit:                  yield (n_("Emission"),              "RGB",  'COLOR')
    if srl.use_pass_environment:           yield (n_("Environment"),           "RGB",  'COLOR')
    if srl.use_pass_ambient_occlusion:     yield (n_("Ambient Occlusion"),     "RGB",  'COLOR')
    if crl.use_pass_shadow_catcher:        yield (n_("Shadow Catcher"),        "RGB",  'COLOR')
    # autopep8: on

    # Debug passes.
    if crl.pass_debug_sample_count:
        yield (n_("Debug Sample Count"), "X", 'VALUE')
    if crl.pass_render_time:
        # Only yield the pass if rendering on CPU
        if scene.cycles.device == 'CPU':
            yield (n_("Render Time"), "X", "VALUE")

    # Cryptomatte passes.
    # NOTE: Name channels are lowercase RGBA so that compression rules check in OpenEXR DWA code
    # uses lossless compression. Reportedly this naming is the only one which works good from the
    # interoperability point of view. Using XYZW naming is not portable.
    crypto_depth = (min(16, srl.pass_cryptomatte_depth) + 1) // 2
    if srl.use_pass_cryptomatte_object:
        for i in range(0, crypto_depth):
            yield ("CryptoObject" + '{:02d}'.format(i), "rgba", 'COLOR')
    if srl.use_pass_cryptomatte_material:
        for i in range(0, crypto_depth):
            yield ("CryptoMaterial" + '{:02d}'.format(i), "rgba", 'COLOR')
    if srl.use_pass_cryptomatte_asset:
        for i in range(0, crypto_depth):
            yield ("CryptoAsset" + '{:02d}'.format(i), "rgba", 'COLOR')

    # Denoising passes.
    if scene.cycles.use_denoising and crl.use_denoising:
        yield (n_("Noisy Image"), "RGBA", 'COLOR')
        if crl.use_pass_shadow_catcher:
            yield (n_("Noisy Shadow Catcher"), "RGB", 'COLOR')
    if crl.denoising_store_passes:
        yield (n_("Denoising Albedo"), "RGB", 'COLOR')
        yield (n_("Denoising Specular Albedo"), "RGB", 'COLOR')
        yield (n_("Denoising Normal"), "XYZ", 'VECTOR')
        yield (n_("Denoising Roughness"), "X", 'VALUE')
        yield (n_("Denoising Depth"), "Z", 'VALUE')
        if crl.denoising_store_motion_passes:
            if not (scene.render.use_motion_blur and srl.use_motion_blur):
                yield (n_("Denoising Backward Motion"), "XYZ", 'VECTOR')
                yield (n_("Denoising Specular Motion"), "XYZ", 'VECTOR')

    # Custom AOV passes.
    for aov in srl.aovs:
        if not aov.is_valid:
            continue
        if aov.type == 'VALUE':
            yield (aov.name, "X", 'VALUE')
        else:
            yield (aov.name, "RGBA", 'COLOR')

    # Native LPE illumination outputs. Shader AOVs keep their separate namespace.
    tree = srl.lpe_node_tree
    if tree is not None:
        if any(node.bl_idname == 'LPENodeViewer' and node.active and not node.mute
               and node.inputs[0].is_linked for node in tree.nodes):
            yield ("LPE Viewer", "RGB", 'COLOR')
        names = set()
        for node in tree.nodes:
            if node.bl_idname != 'LPENodeOutput' or node.mute:
                continue
            for item in node.path_items:
                identifier = "Item_%d" % item.identifier
                socket = next((s for s in node.inputs if s.identifier == identifier), None)
                if socket is None or not socket.is_linked:
                    continue
                name = "LPE_" + item.name
                if name not in names:
                    names.add(name)
                    source_type = socket.type
                    if source_type in {'VALUE', 'INT'}:
                        yield (name, "X", 'VALUE')
                    else:
                        yield (name, "RGB", 'COLOR')

    # Light groups.
    lightgroup_names = [lightgroup.name for lightgroup in srl.lightgroups]
    used_lightgroup_pass_names = set()
    for lightgroup_index, lightgroup_name in enumerate(lightgroup_names):
        yield (
            _lightgroup_pass_name(
                "Combined_", lightgroup_index, lightgroup_name, used_lightgroup_pass_names
            ),
            "RGB",
            'COLOR',
        )

    for index, group in enumerate(srl.lightgroups):
        for prop, suffix in _LIGHTGROUP_SPLIT_PASS_DESCRIPTORS:
            if getattr(group, prop):
                name = _lightgroup_pass_name(suffix, index, group.name, used_lightgroup_pass_names)
                yield (name, "RGB", 'COLOR')

    # Path guiding debug passes.
    if _cycles.with_debug and scene.cycles.use_guiding:
        yield (n_("Guiding Color"), "RGB", 'COLOR')
        yield (n_("Guiding Probability"), "X", 'VALUE')
        yield (n_("Guiding Average Roughness"), "X", 'VALUE')


def register_passes(engine, scene, view_layer):
    for name, channelids, channeltype in list_render_passes(scene, view_layer):
        engine.register_pass(scene, view_layer, name, len(channelids), channelids, channeltype)


def prune_stale_lightgroup_split_file_outputs(scene):
    valid_pass_names = set()
    for view_layer in scene.view_layers:
        valid_pass_names.update(name for name, _, _ in list_render_passes(scene, view_layer))

    visited = set()
    for effect in scene.compositor_effects:
        node_tree = effect.node_group
        if node_tree is None or node_tree.as_pointer() in visited:
            continue
        visited.add(node_tree.as_pointer())
        _prune_stale_lightgroup_split_file_outputs(node_tree, valid_pass_names)


def _prune_stale_lightgroup_split_file_outputs(node_tree, valid_pass_names):
    for node in node_tree.nodes:
        if node.type != 'OUTPUT_FILE':
            continue

        generated_item_names = _lightgroup_generated_file_output_item_names(node)
        if not generated_item_names:
            continue

        stale_item_names = []
        for item in node.file_output_items:
            item_name = item.name
            if item_name not in generated_item_names:
                continue
            if item_name in valid_pass_names:
                continue
            stale_item_names.append(item_name)

        for item_name in stale_item_names:
            stale_item = node.file_output_items.get(item_name)
            if stale_item is not None:
                node.file_output_items.remove(stale_item)

        # Keep a defensive fallback for stale inputs that are not linked to item entries.
        stale_sockets = []
        for socket in node.inputs:
            socket_name = socket.name
            if socket_name not in generated_item_names:
                continue
            if socket_name in valid_pass_names:
                continue
            stale_sockets.append(socket)

        for socket in stale_sockets:
            node.inputs.remove(socket)

        remaining_generated_items = {
            item.name for item in node.file_output_items if item.name in generated_item_names
        }
        _set_lightgroup_generated_file_output_item_names(node, remaining_generated_items)
