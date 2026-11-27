# SPDX-FileCopyrightText: 2011-2022 Blender Foundation
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import bpy
import os
from bpy.app.translations import contexts as i18n_contexts
from bl_ui.utils import PresetPanel

from bpy.types import Panel, Menu

from bl_ui.properties_grease_pencil_common import GreasePencilSimplifyPanel
from bl_ui.properties_render import draw_curves_settings, CompositorPerformanceButtonsPanel, CompositorDenoisePerformanceButtonsPanel
from bl_ui.properties_view_layer import (
    ViewLayerCryptomattePanelHelper,
    ViewLayerAOVPanelHelper,
    ViewLayerLightgroupsPanelHelper,
)

from bl_ui.properties_object import has_geometry_visibility
from bpy.app.translations import (
    pgettext_rpt as rpt_,
)
from .engine import use_vcm, vcm_fallback_reason, vcm_lpe_outputs
from .translations import CONTEXT as photon_caustics_context


class CyclesPresetPanel(PresetPanel, Panel):
    COMPAT_ENGINES = {'CYCLES'}
    preset_operator = "script.execute_preset"

    @staticmethod
    def post_cb(context, _filepath):
        # Modify an arbitrary built-in scene property to force a depsgraph
        # update, because add-on properties don't. (see #62325)
        render = context.scene.render
        render.filter_size = render.filter_size


class CYCLES_PT_sampling_presets(CyclesPresetPanel):
    bl_label = "Sampling Presets"
    preset_subdir = "cycles/sampling"
    preset_add_operator = "render.cycles_sampling_preset_add"


class CYCLES_PT_viewport_sampling_presets(CyclesPresetPanel):
    bl_label = "Viewport Sampling Presets"
    preset_subdir = "cycles/viewport_sampling"
    preset_add_operator = "render.cycles_viewport_sampling_preset_add"


class CYCLES_PT_integrator_presets(CyclesPresetPanel):
    bl_label = "Integrator Presets"
    preset_subdir = "cycles/integrator"
    preset_add_operator = "render.cycles_integrator_preset_add"


class CYCLES_PT_photon_presets(CyclesPresetPanel):
    bl_label = "Photon Presets"
    bl_translation_context = photon_caustics_context
    preset_subdir = "cycles/photon"
    preset_add_operator = "render.cycles_photon_preset_add"


class CYCLES_PT_performance_presets(CyclesPresetPanel):
    bl_label = "Performance Presets"
    preset_subdir = "cycles/performance"
    preset_add_operator = "render.cycles_performance_preset_add"


class CyclesButtonsPanel:
    bl_space_type = "PROPERTIES"
    bl_region_type = "WINDOW"
    bl_context = "render"
    COMPAT_ENGINES = {'CYCLES'}

    @classmethod
    def poll(cls, context):
        return context.engine in cls.COMPAT_ENGINES


class CyclesDebugButtonsPanel(CyclesButtonsPanel):
    @classmethod
    def poll(cls, context):
        prefs = bpy.context.preferences
        return (
            CyclesButtonsPanel.poll(context) and
            prefs.experimental.use_cycles_debug and
            prefs.view.show_developer_ui
        )


# Adapt properties editor panel to display in node editor. We have to
# copy the class rather than inherit due to the way bpy registration works.
def node_panel(cls):
    node_cls = type('NODE_' + cls.__name__, cls.__bases__, dict(cls.__dict__))

    node_cls.bl_space_type = 'NODE_EDITOR'
    node_cls.bl_region_type = 'UI'
    node_cls.bl_category = "Options"
    if hasattr(node_cls, 'bl_parent_id'):
        node_cls.bl_parent_id = 'NODE_' + node_cls.bl_parent_id

    return node_cls


def get_device_type(context):
    return context.preferences.addons[__package__].preferences.compute_device_type


def backend_has_active_gpu(context):
    return context.preferences.addons[__package__].preferences.has_active_device()


def use_cpu(context):
    cscene = context.scene.cycles

    return (get_device_type(context) == 'NONE' or cscene.device == 'CPU' or not backend_has_active_gpu(context))


def use_gpu(context):
    cscene = context.scene.cycles

    return (get_device_type(context) != 'NONE' and cscene.device == 'GPU' and backend_has_active_gpu(context))


def use_metal(context):
    return (get_device_type(context) == 'METAL' and use_gpu(context))


def use_cuda(context):
    return (get_device_type(context) == 'CUDA' and use_gpu(context))


def use_hip(context):
    return (get_device_type(context) == 'HIP' and use_gpu(context))


def use_optix(context):
    return (get_device_type(context) == 'OPTIX' and use_gpu(context))


def use_oneapi(context):
    return (get_device_type(context) == 'ONEAPI' and use_gpu(context))


def use_multi_device(context):
    if use_gpu(context):
        return context.preferences.addons[__package__].preferences.has_multi_device()
    return False


def show_preview_denoise_active(context):
    cscene = context.scene.cycles
    if not cscene.use_preview_denoising:
        return False

    if cscene.preview_denoiser == 'DLSS':
        return has_dlss_gpu_devices(context)

    if cscene.preview_denoiser == 'OPTIX':
        return has_optixdenoiser_gpu_devices(context)

    # OIDN is always available, thanks to CPU support
    return True


def show_denoise_active(context):
    cscene = context.scene.cycles
    if not cscene.use_denoising:
        return False

    if cscene.denoiser == 'OPTIX':
        return has_optixdenoiser_gpu_devices(context)

    # OIDN is always available, thanks to CPU support
    return True


def show_preview_dlss_active(context):
    cscene = context.scene.cycles
    return (cscene.use_preview_denoising and cscene.preview_denoiser == 'DLSS' and
            has_dlss_gpu_devices(context) and not use_vcm(context.scene, context.view_layer))


def get_effective_preview_denoiser(context, has_oidn_gpu):
    scene = context.scene
    cscene = scene.cycles

    if cscene.preview_denoiser != "AUTO":
        return cscene.preview_denoiser

    if has_oidn_gpu:
        return 'OPENIMAGEDENOISE'

    if has_optixdenoiser_gpu_devices(context):
        return 'OPTIX'

    return 'OPENIMAGEDENOISE'


def has_oidn_gpu_devices(context):
    return context.preferences.addons[__package__].preferences.has_oidn_gpu_devices()


def has_dlss_gpu_devices(context):
    return context.preferences.addons[__package__].preferences.has_dlss_gpu_devices()


def has_optixdenoiser_gpu_devices(context):
    return context.preferences.addons[__package__].preferences.has_optixdenoiser_gpu_devices()


class CYCLES_RENDER_PT_sampling(CyclesButtonsPanel, Panel):
    bl_label = "Sampling"

    def draw(self, context):
        pass


class CYCLES_RENDER_PT_sampling_viewport(CyclesButtonsPanel, Panel):
    bl_label = "Viewport"
    bl_parent_id = "CYCLES_RENDER_PT_sampling"

    def draw_header_preset(self, context):
        CYCLES_PT_viewport_sampling_presets.draw_panel_header(self.layout)

    def draw(self, context):
        layout = self.layout

        scene = context.scene
        cscene = scene.cycles

        layout.use_property_split = True
        layout.use_property_decorate = False

        heading = layout.column(align=True, heading="Noise Threshold")
        heading.active = not show_preview_dlss_active(context)
        row = heading.row(align=True)
        row.prop(cscene, "use_preview_adaptive_sampling", text="")
        sub = row.row()
        sub.active = cscene.use_preview_adaptive_sampling
        sub.prop(cscene, "preview_adaptive_threshold", text="")

        col = layout.column(align=True)
        if cscene.use_preview_adaptive_sampling:
            col.prop(cscene, "preview_samples", text="Max Samples")
            col.prop(cscene, "preview_adaptive_min_samples", text="Min Samples")
        else:
            col.prop(cscene, "preview_samples", text="Samples")


class CYCLES_RENDER_PT_sampling_viewport_denoise(CyclesButtonsPanel, Panel):
    bl_label = "Denoise"
    bl_parent_id = 'CYCLES_RENDER_PT_sampling_viewport'
    bl_options = {'DEFAULT_CLOSED'}

    def draw_header(self, context):
        scene = context.scene
        cscene = scene.cycles

        self.layout.prop(context.scene.cycles, "use_preview_denoising", text="")

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False

        scene = context.scene
        cscene = scene.cycles

        col = layout.column()
        col.active = cscene.use_preview_denoising

        sub = col.column()
        sub.active = show_preview_denoise_active(context)
        sub.prop(cscene, "preview_denoiser", text="Denoiser")

        has_oidn_gpu = has_oidn_gpu_devices(context)
        effective_preview_denoiser = get_effective_preview_denoiser(context, has_oidn_gpu)

        if effective_preview_denoiser == 'DLSS':
            if use_vcm(scene, context.view_layer):
                col.label(text="VCM viewport denoising requires OptiX or Open Image Denoise", icon='INFO')
                return
            if has_dlss_gpu_devices(context):
                col.prop(cscene, "preview_denoising_upscale_quality", text="Upscale Mode")
            else:
                col.label_multiline(
                    text=rpt_("Requires NVIDIA GPU with compute capability %s and NVIDIA driver version %s or newer.") %
                    ("7.5", "620"), icon='STATUS_INFO', translate=False)
            return

        if use_vcm(scene, context.view_layer):
            col.label(text="Denoising input: color")
        else:
            col.prop(cscene, "preview_denoising_input_passes", text="Passes")

        if effective_preview_denoiser == 'OPENIMAGEDENOISE':
            col.prop(cscene, "preview_denoising_prefilter", text="Prefilter")
            col.prop(cscene, "preview_denoising_quality", text="Quality")

        col.prop(cscene, "preview_denoising_start_sample", text="Start Sample")

        if effective_preview_denoiser == 'OPENIMAGEDENOISE':
            row = col.row()
            row.active = has_oidn_gpu_devices(context)
            row.prop(cscene, "preview_denoising_use_gpu", text="Use GPU")


class CYCLES_RENDER_PT_sampling_render(CyclesButtonsPanel, Panel):
    bl_label = "Render"
    bl_parent_id = "CYCLES_RENDER_PT_sampling"

    def draw_header_preset(self, context):
        CYCLES_PT_sampling_presets.draw_panel_header(self.layout)

    def draw(self, context):
        layout = self.layout

        scene = context.scene
        cscene = scene.cycles

        layout.use_property_split = True
        layout.use_property_decorate = False

        heading = layout.column(align=True, heading="Noise Threshold")
        row = heading.row(align=True)
        row.prop(cscene, "use_adaptive_sampling", text="")
        sub = row.row()
        sub.active = cscene.use_adaptive_sampling
        sub.prop(cscene, "adaptive_threshold", text="")

        col = layout.column(align=True)
        if cscene.use_adaptive_sampling:
            col.prop(cscene, "samples", text="Max Samples")
            col.prop(cscene, "adaptive_min_samples", text="Min Samples")
        else:
            col.prop(cscene, "samples", text="Samples")
        col.prop(cscene, "time_limit")


class CYCLES_RENDER_PT_sampling_render_denoise(CyclesButtonsPanel, Panel):
    bl_label = "Denoise"
    bl_parent_id = 'CYCLES_RENDER_PT_sampling_render'
    bl_options = {'DEFAULT_CLOSED'}

    def draw_header(self, context):
        scene = context.scene
        cscene = scene.cycles

        self.layout.prop(context.scene.cycles, "use_denoising", text="")

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False

        scene = context.scene
        cscene = scene.cycles

        col = layout.column()
        col.active = cscene.use_denoising

        sub = col.column()
        sub.active = show_denoise_active(context)
        sub.prop(cscene, "denoiser", text="Denoiser")

        if use_vcm(scene, context.view_layer):
            col.label(text="Denoising input: color")
        else:
            col.prop(cscene, "denoising_input_passes", text="Passes")
        if cscene.denoiser == 'OPENIMAGEDENOISE':
            col.prop(cscene, "denoising_prefilter", text="Prefilter")
            col.prop(cscene, "denoising_quality", text="Quality")

        if cscene.denoiser == 'OPENIMAGEDENOISE':
            row = col.row()
            row.active = has_oidn_gpu_devices(context)
            row.prop(cscene, "denoising_use_gpu", text="Use GPU")


class CYCLES_RENDER_PT_sampling_gpu_guiding(CyclesButtonsPanel, Panel):
    bl_label = "GPU Surface Path Guiding"
    bl_translation_context = photon_caustics_context
    bl_parent_id = "CYCLES_RENDER_PT_sampling"
    bl_options = {'DEFAULT_CLOSED'}

    @classmethod
    def poll(cls, context):
        return use_cuda(context) or use_optix(context)

    def draw_header(self, context):
        self.layout.prop(context.scene.cycles, "use_gpu_guiding", text="")

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False
        cscene = context.scene.cycles
        layout.active = cscene.use_gpu_guiding
        layout.prop(cscene, "gpu_guiding_training_samples", text="Training Samples",
                    text_ctxt=photon_caustics_context)
        layout.prop(cscene, "gpu_guiding_probability")


class CYCLES_RENDER_PT_sampling_path_guiding(CyclesButtonsPanel, Panel):
    bl_label = "Path Guiding"
    bl_parent_id = "CYCLES_RENDER_PT_sampling"
    bl_options = {'DEFAULT_CLOSED'}

    @classmethod
    def poll(cls, context):
        from . import engine
        return not use_vcm(context.scene, context.view_layer) and use_cpu(context) and engine.with_path_guiding()

    def draw_header(self, context):
        scene = context.scene
        cscene = scene.cycles

        self.layout.prop(cscene, "use_guiding", text="")

    def draw(self, context):
        scene = context.scene
        cscene = scene.cycles

        layout = self.layout

        layout.use_property_split = True
        layout.use_property_decorate = False
        layout.active = cscene.use_guiding

        layout.prop(cscene, "guiding_training_samples")

        col = layout.column(align=True)
        col.prop(cscene, "use_surface_guiding", text="Surface")
        col.prop(cscene, "use_volume_guiding", text="Volume", text_ctxt=i18n_contexts.id_id)

        if cscene.use_guiding:
            # Calculation matches TileManager::compute_render_tile_size and
            # Session::get_effective_tile_size
            if cscene.tile_size < 128:
                tile_size = cscene.tile_size
            else:
                tile_size = (cscene.tile_size + 128 - 1) & ~(128 - 1)
                tile_size = min(tile_size, 8192)
            tile_area = tile_size ** 2

            render_scale = scene.render.resolution_percentage / 100.0
            render_size_x = int(scene.render.resolution_x * render_scale)
            render_size_y = int(scene.render.resolution_y * render_scale)
            render_area = render_size_x * render_size_y

            if render_area > tile_area and render_size_x <= 8192 and render_size_y <= 8192:
                layout.label(text="May work poorly with render tiling", icon='STATUS_INFO')


class CYCLES_RENDER_PT_sampling_path_guiding_debug(CyclesDebugButtonsPanel, Panel):
    bl_label = "Debug"
    bl_parent_id = "CYCLES_RENDER_PT_sampling_path_guiding"
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        scene = context.scene
        cscene = scene.cycles

        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False
        layout.active = cscene.use_guiding

        layout.prop(cscene, "guiding_distribution_type", text="Distribution Type")
        layout.prop(cscene, "guiding_roughness_threshold")
        layout.prop(cscene, "guiding_directional_sampling_type", text="Directional Sampling Type")

        col = layout.column(align=True)
        col.prop(cscene, "surface_guiding_probability")
        col.prop(cscene, "volume_guiding_probability")

        col = layout.column(align=True)
        col.prop(cscene, "use_deterministic_guiding")
        col.prop(cscene, "use_guiding_direct_light")
        col.prop(cscene, "use_guiding_mis_weights")


class CYCLES_RENDER_PT_sampling_advanced(CyclesButtonsPanel, Panel):
    bl_label = "Advanced"
    bl_parent_id = "CYCLES_RENDER_PT_sampling"
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False

        scene = context.scene
        cscene = scene.cycles

        row = layout.row(align=True)
        row.active = not use_vcm(scene, context.view_layer)
        row.prop(cscene, "sampling_pattern", text="Pattern")

        row = layout.row(align=True)
        row.prop(cscene, "seed")
        row.prop(cscene, "use_animated_seed", text="", icon='TIME')

        if use_vcm(scene, context.view_layer):
            return

        layout.separator()

        prefs = context.preferences
        use_debug = prefs.experimental.use_cycles_debug and prefs.view.show_developer_ui
        if use_debug:
            row = layout.row(align=True)
            row.prop(cscene, "use_pixel_jitter")

            layout.separator()

        if cscene.sampling_pattern == 'TABULATED_SOBOL':
            heading = layout.column(align=True, heading="Scrambling Distance")
            heading.prop(cscene, "auto_scrambling_distance", text="Automatic")
            preview_scrambling_row = heading.row()
            preview_scrambling_row.prop(cscene, "preview_scrambling_distance", text="Viewport")
            # Disable preview scrambling if DLSS denoising is used.
            # Preview scrambling and DLSS are generally incompatible with each other,
            # so preview scrambling is internally disabled when using DLSS.
            preview_scrambling_row.active = not show_preview_dlss_active(context)
            heading.prop(cscene, "scrambling_distance", text="Multiplier")

            layout.separator()

        col = layout.column(align=True)
        col.prop(cscene, "min_light_bounces")
        col.prop(cscene, "min_transparent_bounces")

        for view_layer in scene.view_layers:
            if view_layer.samples > 0:
                layout.separator()
                layout.row().prop(cscene, "use_layer_samples")
                break


class CYCLES_RENDER_PT_sampling_advanced_sample_subset(CyclesButtonsPanel, Panel):
    bl_label = "Sample Subset"
    bl_parent_id = 'CYCLES_RENDER_PT_sampling_advanced'
    bl_options = {'DEFAULT_CLOSED'}

    def draw_header(self, context):
        scene = context.scene
        cscene = scene.cycles

        self.layout.prop(cscene, "use_sample_subset", text="")

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False

        scene = context.scene
        cscene = scene.cycles

        col = layout.column(align=True)
        col.active = cscene.use_sample_subset
        col.prop(cscene, "sample_offset", text="Offset")
        col.prop(cscene, "sample_subset_length", text="Length")


class CYCLES_RENDER_PT_sampling_lights(CyclesButtonsPanel, Panel):
    bl_label = "Lights"
    bl_parent_id = "CYCLES_RENDER_PT_sampling"
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False

        scene = context.scene
        cscene = scene.cycles

        col = layout.column(align=True)
        col.prop(cscene, "use_light_tree")
        sub = col.row()
        sub.prop(cscene, "light_sampling_threshold", text="Light Threshold")
        sub.active = not cscene.use_light_tree


class CYCLES_RENDER_PT_subdivision(CyclesButtonsPanel, Panel):
    bl_label = "Subdivision"
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False

        scene = context.scene
        cscene = scene.cycles

        col = layout.column()
        sub = col.column(align=True)
        sub.prop(cscene, "dicing_rate", text="Dicing Rate Render")
        sub.prop(cscene, "preview_dicing_rate", text="Viewport")

        col.separator()

        col.prop(cscene, "offscreen_dicing_scale", text="Offscreen Scale")
        col.prop(cscene, "max_subdivisions")

        col.prop(cscene, "dicing_camera")


class CYCLES_RENDER_PT_curves(CyclesButtonsPanel, Panel):
    bl_label = "Curves"
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False

        scene = context.scene
        ccscene = scene.cycles_curves

        col = layout.column()
        col.prop(ccscene, "shape", text="Shape")
        if ccscene.shape == 'RIBBONS':
            col.prop(ccscene, "subdivisions", text="Curve Subdivisions")


class CYCLES_RENDER_PT_curves_viewport_display(CyclesButtonsPanel, Panel):
    bl_label = "Viewport Display"
    bl_parent_id = "CYCLES_RENDER_PT_curves"
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        draw_curves_settings(self, context)


class CYCLES_RENDER_PT_volumes(CyclesButtonsPanel, Panel):
    bl_label = "Volumes"
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False

        scene = context.scene
        cscene = scene.cycles

        col = layout.column(align=True)
        col.prop(cscene, "volume_biased", text="Biased")
        if cscene.volume_biased:
            col.prop(cscene, "volume_step_rate", text="Step Rate Render")
            col.prop(cscene, "volume_preview_step_rate", text="Viewport")

            layout.prop(cscene, "volume_max_steps", text="Max Steps")


class CYCLES_RENDER_PT_light_paths(CyclesButtonsPanel, Panel):
    bl_label = "Light Paths"
    bl_options = {'DEFAULT_CLOSED'}

    def draw_header_preset(self, context):
        CYCLES_PT_integrator_presets.draw_panel_header(self.layout)

    def draw(self, context):
        pass


class CYCLES_RENDER_PT_light_paths_integrator(CyclesButtonsPanel, Panel):
    bl_label = "Integrator"
    bl_parent_id = "CYCLES_RENDER_PT_light_paths"

    @classmethod
    def poll(cls, context):
        # VCM is currently enabled through scene.cycles.integrator_method in Python.
        return super().poll(context) and context.scene.cycles.integrator_method == 'VCM'

    def draw(self, context):
        cscene = context.scene.cycles
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False
        reason = vcm_fallback_reason(context.scene, context.view_layer)
        if reason:
            layout.label(text="Rendering with Cycles Path Tracing", icon='INFO')
            layout.label(text=reason)
            return
        if cscene.device == 'GPU' and (get_device_type(context) != 'OPTIX' or
                                       not backend_has_active_gpu(context)):
            layout.label(text="Select an OptiX GPU in Preferences", icon='ERROR')
        layout.prop(cscene, "vcm_paths_per_round")
        if context.view_layer.lpe_node_tree:
            layout.label(text="VCM LPE: connect Light or World Total Light directly to an output", icon='INFO')
            for name, _, supported in vcm_lpe_outputs(context.view_layer):
                if not supported:
                    layout.label(text=rpt_("Unsupported VCM light paths: %s") % name,
                                 translate=False, icon='ERROR')
        layout.label(text="Final passes are listed under View Layer")
        if context.scene.render.use_motion_blur and cscene.rolling_shutter_type != 'NONE':
            layout.label(text="VCM uses a global shutter", icon='INFO')


class CYCLES_RENDER_PT_light_paths_max_bounces(CyclesButtonsPanel, Panel):
    bl_label = "Max Bounces"
    bl_parent_id = "CYCLES_RENDER_PT_light_paths"

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False

        scene = context.scene
        cscene = scene.cycles

        col = layout.column(align=True)
        col.prop(cscene, "max_bounces", text="Total")

        col = layout.column(align=True)
        col.prop(cscene, "diffuse_bounces", text="Diffuse")
        col.prop(cscene, "glossy_bounces", text="Glossy")
        col.prop(cscene, "transmission_bounces", text="Transmission")
        col.prop(cscene, "volume_bounces", text="Volume", text_ctxt=i18n_contexts.id_id)

        col = layout.column(align=True)
        col.prop(cscene, "transparent_max_bounces", text="Transparent")


class CYCLES_RENDER_PT_light_paths_clamping(CyclesButtonsPanel, Panel):
    bl_label = "Clamping"
    bl_parent_id = "CYCLES_RENDER_PT_light_paths"

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False

        scene = context.scene
        cscene = scene.cycles

        col = layout.column(align=True)
        col.prop(cscene, "sample_clamp_direct", text="Direct Light")
        col.prop(cscene, "sample_clamp_indirect", text="Indirect Light")


def photon_caustics_inactive_reason(context):
    # Missing selections are informational. Instances and final light enable flags
    # require evaluated scene data, so the renderer remains authoritative.
    has_light = has_caster = False
    for ob in context.view_layer.objects:
        if ob.hide_render:
            continue
        if ob.instance_type != 'NONE':
            return None
        if ob.type == 'LIGHT':
            light = ob.data
            has_light |= (ob.cycles.visible_photon and light.cycles.is_caustics_light and
                          light.energy != 0 and any(light.color))
        elif ob.type in {'MESH', 'CURVE', 'SURFACE', 'META', 'FONT'}:
            has_caster |= ob.cycles.is_caustics_caster
    world = context.view_layer.world_override or context.scene.world
    has_light |= bool(world and world.cycles.visible_photon and world.cycles.is_caustics_light)
    if not (has_light and has_caster):
        return "No caustic light or caster: continuing ordinary path tracing"
    return None


def photon_caustics_uses_deep_output(scene):
    if scene.render.image_settings.file_format == 'DEEP_EXR':
        return True
    if not scene.render.use_compositing:
        return False
    visited = set()

    def has_deep(tree):
        if tree is None or tree.as_pointer() in visited:
            return False
        visited.add(tree.as_pointer())
        for node in tree.nodes:
            if node.mute:
                continue
            if node.bl_idname == 'CompositorNodeOutputFile':
                if getattr(node.format, "file_format", None) == 'DEEP_EXR':
                    return True
                for slot in getattr(node, "file_slots", ()):
                    if (not slot.use_node_format and slot.format.file_format == 'DEEP_EXR'):
                        return True
            if has_deep(getattr(node, "node_tree", None)):
                return True
        return False

    return any(has_deep(effect.node_group) for effect in scene.compositor_effects
               if effect.enable_for_render)


def photon_caustics_test_hybrid(devices):
    return (os.environ.get("CYCLES_PHOTON_TEST_HYBRID") == "1" and
            sorted(d.type for d in devices) == ['CPU', 'OPTIX'])


def photon_caustics_unavailable_reason(context):
    # Cheap authoring checks. Evaluated shaders, instances and bake state are validated
    # by Cycles during scene upload and produce an error, never an estimator switch.
    if photon_caustics_inactive_reason(context):
        return None
    scene = context.scene
    cscene = scene.cycles
    preferences = context.preferences.addons[__package__].preferences
    multi = False
    if not use_cpu(context):
        backend = get_device_type(context)
        devices = [d for d in preferences.get_devices_for_type(backend) if d.use]
        multi = len(devices) > 1
        if multi:
            if (not photon_caustics_test_hybrid(devices) and
                    (any(d.type != 'OPTIX' for d in devices) or len({d.name for d in devices}) != 1)):
                return "Photon multi-GPU requires identical OptiX GPUs without CPU"
        elif backend not in {'CUDA', 'OPTIX'} or len(devices) != 1:
            return "Use CPU, one CUDA/OptiX GPU, or identical OptiX GPUs"
    if (scene.render.use_motion_blur and context.view_layer.use_motion_blur and
            cscene.rolling_shutter_type != 'NONE'):
        return "Rolling shutter is not supported"
    if cscene.shading_system:
        return "OSL is not supported"
    if photon_caustics_uses_deep_output(scene):
        if multi:
            return "Multi-GPU photon rendering in Deep EXR is not supported"

    world = context.view_layer.world_override or scene.world
    world_volume = (world and world.use_nodes and world.node_tree and
                    any(node.type == 'OUTPUT_WORLD' and node.is_active_output and
                        node.inputs['Volume'].is_linked for node in world.node_tree.nodes))
    geometry_types = {'MESH', 'CURVE', 'SURFACE', 'META', 'FONT', 'VOLUME', 'CURVES', 'POINTCLOUD'}
    visibility = ('visible_diffuse', 'visible_glossy', 'visible_transmission',
                  'visible_volume_scatter', 'visible_shadow')
    for ob in context.view_layer.objects:
        if ob.hide_render:
            continue
        if ob.type not in geometry_types and ob.type != 'LIGHT':
            continue
        if ob.type == 'LIGHT':
            light = ob.data
            if (not ob.cycles.visible_photon or not light.cycles.is_caustics_light or
                    light.energy == 0 or not any(light.color)):
                continue
        # Camera rays use the native prefix in both sampling branches.
        if ob.type != 'LIGHT' and any(not getattr(ob, prop, True) for prop in visibility):
            return "Custom ray visibility is not supported"
        if ob.type != 'LIGHT' and ob.is_shadow_catcher:
            return "Shadow catchers are not supported"
        if ob.cycles.is_caustics_caster:
            if ob.type in {'CURVES', 'POINTCLOUD'}:
                return "Use mesh casters"
        if ob.type == 'LIGHT':
            light = ob.data
            linking = ob.light_linking
            if linking.blocker_collection:
                return "Shadow linking on caustic lights is not supported"
            if light.type not in {'POINT', 'SPOT', 'SUN', 'AREA'}:
                return "Only Point, Spot, Sun and Area lights are supported"
            if light.type == 'SUN' and world_volume:
                return "Sun photon caustics require bounded mesh or VDB volumes"
            if not light.use_shadow:
                return "Enable Cast Shadow on caustic lights"
            if any(light.shadow_color):
                return "Colored caustic shadows are not supported"
    return None


def photon_caustics_performance_hint(context):
    if context.scene.cycles.caustics_method == 'PHOTON' and use_cuda(context):
        return "CUDA photon caustics may be slow; prefer OptiX"
    return None


class CYCLES_MT_caustics_method(Menu):
    bl_label = "Shadow Caustics Method"
    bl_translation_context = photon_caustics_context

    def draw(self, context):
        layout = self.layout
        cscene = context.scene.cycles
        layout.prop_enum(cscene, "caustics_method", 'MNEE')
        unavailable_reason = photon_caustics_unavailable_reason(context)
        row = layout.row()
        row.enabled = unavailable_reason is None
        row.prop_enum(cscene, "caustics_method", 'PHOTON')
        if unavailable_reason:
            layout.separator()
            layout.label(text=unavailable_reason, text_ctxt=photon_caustics_context, icon='INFO')


class CYCLES_RENDER_PT_light_paths_caustics(CyclesButtonsPanel, Panel):
    bl_label = "Caustics"
    bl_parent_id = "CYCLES_RENDER_PT_light_paths"

    @classmethod
    def poll(cls, context):
        return super().poll(context) and not use_vcm(context.scene, context.view_layer)

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False

        scene = context.scene
        cscene = scene.cycles

        col = layout.column()
        col.prop(cscene, "blur_glossy")
        col = layout.column(heading="Caustics", align=True)
        col.prop(cscene, "caustics_reflective", text="Reflective")
        col.prop(cscene, "caustics_refractive", text="Refractive")
        unavailable_reason = photon_caustics_unavailable_reason(context)
        row = layout.row(heading="Shadow Caustics Method", heading_ctxt=photon_caustics_context)
        method = cscene.bl_rna.properties["caustics_method"].enum_items[cscene.caustics_method]
        row.menu("CYCLES_MT_caustics_method", text=method.name, text_ctxt=photon_caustics_context)

        if cscene.caustics_method == 'PHOTON':
            if unavailable_reason:
                layout.label(text=unavailable_reason, text_ctxt=photon_caustics_context, icon='ERROR')
            elif (inactive_reason := photon_caustics_inactive_reason(context)):
                layout.label(text=inactive_reason, text_ctxt=photon_caustics_context, icon='INFO')
            if not unavailable_reason and use_optix(context) and use_multi_device(context):
                preferences = context.preferences.addons[__package__].preferences
                devices = [d for d in preferences.get_devices_for_type('OPTIX') if d.use]
                hint = ("CPU + OptiX test: each device uses the sample budget" if
                        photon_caustics_test_hybrid(devices) else
                        "Each GPU uses the sample budget; results are averaged")
                layout.label(text=hint, text_ctxt=photon_caustics_context, icon='INFO')
            if (performance_hint := photon_caustics_performance_hint(context)):
                layout.label(text=performance_hint, text_ctxt=photon_caustics_context, icon='INFO')


class CYCLES_RENDER_PT_photon_sampling(CyclesButtonsPanel, Panel):
    bl_label = "Photon Sampling"
    bl_translation_context = photon_caustics_context
    bl_parent_id = "CYCLES_RENDER_PT_light_paths_caustics"

    @classmethod
    def poll(cls, context):
        return super().poll(context) and context.scene.cycles.caustics_method == 'PHOTON'

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False
        layout.active = photon_caustics_unavailable_reason(context) is None
        cscene = context.scene.cycles
        col = layout.column(align=True)
        col.prop(cscene, "photon_caustics_count")
        col.prop(cscene, "photon_caustics_update_interval")
        row = layout.row()
        row.active = use_optix(context)
        row.prop(cscene, "use_photon_ser")
        layout.prop(cscene, "use_photon_query_index")
        layout.separator()
        sampling = layout.column(align=True)
        deep_output = photon_caustics_uses_deep_output(context.scene)
        sampling.enabled = not deep_output
        sampling.prop(cscene, "photon_caustics_separate_sampling")
        if cscene.photon_caustics_separate_sampling:
            sampling.prop(cscene, "photon_caustics_sampling_budget")
            if cscene.photon_caustics_sampling_budget == 'INDEPENDENT':
                sampling.prop(cscene, "photon_caustics_samples")
        if deep_output:
            layout.label(text="Deep output uses combined convergence", text_ctxt=photon_caustics_context, icon='INFO')
        draw_photon_size_hint(context, layout)


def _photon_ipr_budget(cscene):
    # The viewport never captures Deep output. Its independent estimator keeps its
    # own sample limit even when the offline render is configured for Deep EXR.
    independent = (cscene.photon_caustics_separate_sampling and
                   cscene.photon_caustics_sampling_budget == 'INDEPENDENT')
    return independent, cscene.photon_caustics_samples if independent else cscene.preview_samples


def _photon_size_probe(context):
    from math import isclose
    from .engine import _photon_probes, _query_viewports

    if not context.window or not context.view_layer:
        return None
    window = context.window.as_pointer()
    area = context.area
    scene = context.scene
    cscene = scene.cycles
    _, sample_limit = _photon_ipr_budget(cscene)
    probes = []
    for key, (_, uid, layer, _) in _query_viewports.items():
        if key[0] != window or uid != scene.session_uid or layer != context.view_layer.name:
            continue
        if area and area.type == 'VIEW_3D' and key[1] != area.as_pointer():
            continue
        probe = _photon_probes.get(key)
        if (probe and probe[13] and isclose(probe[6], cscene.photon_caustics_radius, rel_tol=1e-5)
                and isclose(probe[7], cscene.photon_adaptive_max_scale, rel_tol=1e-5)
                and probe[15] == cscene.photon_caustics_count
                and probe[17] == cscene.photon_caustics_update_interval
                and probe[18] == (sample_limit or (1 << 24))):
            probes.append(probe)
    return max(probes, key=lambda p: p[14], default=None)


def _photon_hint_values(scene, probe, tr):
    from math import ceil, log, log10, sqrt

    cscene = scene.cycles
    values = {"Probe status": tr("Waiting for IPR data")}
    if probe is None:
        return values
    (maps, emitted, stored, sampled, isolated, helped, radius, scale,
     in_frame, subpixel, radius_limited, pixel_scale_sum, volume, device_maps, completed, requested,
     capacity, interval, sample_limit, has_volumes, deposit_cv, max_emission, all_emitted, all_stored) = probe

    def length(value):
        units = scene.unit_settings
        if units.system == 'NONE':
            return f"{value:.4g}"
        return bpy.utils.units.to_string(units.system, 'LENGTH', value * units.scale_length, precision=4)

    budget = (sample_limit + interval - 1) // interval
    independent, configured_limit = _photon_ipr_budget(cscene)
    unlimited = configured_limit == 0
    values.update({
        "Probe status": tr("Collecting observations") if maps < 3 else tr("Sampling"),
        "Probed rounds / device maps": f"{maps:,} / {device_maps:,}",
        "IPR sample budget": tr("Independent photon samples" if independent else "Shared viewport samples"),
        "Camera samples per emission round": f"{interval:,}",
        "Emitted per device / round": f"{emitted / device_maps:,.0f}",
        "Surface deposits / round": f"{(stored - volume) / device_maps:,.0f}",
        "Volume deposits / round": f"{volume / device_maps:,.0f}",
        "Emission rounds / budget": f"{completed:,} / " + ("∞" if unlimited else f"{budget:,}"),
    })
    if all_emitted:
        values["Stored / attempted (all devices)"] = (
            f"{all_stored:,} / {all_emitted:,} ({100 * all_stored / all_emitted:.1f}%)")
    if sampled:
        values["Isolated / tested"] = f"{isolated:,} / {sampled:,} ({100 * isolated / sampled:.0f}%)"
    if in_frame:
        values["Subpixel / in-frame isolated"] = f"{subpixel:,} / {in_frame:,}"
    if isolated:
        values["Helped by widening / isolated"] = f"{helped:,} / {isolated:,}"
    if maps >= 4 and deposit_cv >= 0:
        values["Deposit count variation"] = f"{100 * deposit_cv:.0f}%"

    if maps < 3:
        return values
    if not all_stored:
        values["Probe status"] = tr("Check light and caster")
        return values

    surface_tested = (cscene.use_photon_adaptive_radius and
                      cscene.photon_reconstruction_method == 'ELLIPSE_COVERAGE' and
                      scale > 1 and sampled >= 64)
    sparse = surface_tested and isolated > sampled / 2 + sqrt(sampled)
    radius_first = False
    if surface_tested:
        values["Coverage evidence"] = tr("Sampled surface overlap")
    else:
        values["Coverage evidence"] = tr("Deposit counts only")
    if has_volumes and not surface_tested:
        values["Probe status"] = tr("Check beam coverage") if volume else tr("No volume hits yet")
    elif surface_tested:
        if sparse:
            values["Probe status"] = tr("Sparse surface coverage")
            if in_frame >= 64 and subpixel > in_frame / 2 + sqrt(in_frame):
                if radius_limited > subpixel / 2 + sqrt(subpixel):
                    values["Suggested gather radius"] = length(radius * pixel_scale_sum / (16 * radius_limited))
                    radius_first = True
            elif helped > isolated / 2 + sqrt(isolated):
                values["Suggested gather radius"] = f"{length(radius * sqrt(scale))} – {length(radius * scale)}"
                radius_first = True
        elif isolated < sampled / 2 - sqrt(sampled):
            values["Probe status"] = tr("Surface kernels overlap")
            values["Suggested gather radius"] = length(radius)
        else:
            values["Probe status"] = tr("Mixed surface coverage")

    limit = min(max_emission, 4 * 1024 * 1024)
    if capacity < requested or requested >= limit:
        values["Suggested photons / round"] = tr("At capacity")
    elif radius_first:
        values["Suggested photons / round"] = f"{requested:,}"
    elif sparse:
        # A two-sigma Wilson lower bound tempers small or near-unanimous samples.
        # The sparse check above is the matching score test against p = 0.5.
        fraction = isolated / sampled
        lower = ((fraction + 2 / sampled
                  - 2 * sqrt((fraction * (1 - fraction) + 1 / sampled) / sampled))
                 / (1 + 4 / sampled))
        # Treat weak overlap as a proxy for an empty Poisson neighborhood:
        # at fixed radius, a k-fold increase in density would leave p**k isolated.
        growth = log(2) / max(-log(lower), log(2) / 3)
        trial = min(requested * growth, limit)
        step = 10 ** max(0, int(log10(trial)) - 1)
        suggested = int(trial // step) * step
        if suggested > requested:
            values["Suggested photons / round"] = f"{suggested:,}"

    # Use the observed per-map deposit-count CV as a proxy for relative error
    # of the mean count. It cannot measure pixel noise or volume beam coverage.
    if not unlimited and completed >= budget and maps >= 8 and deposit_cv >= 0:
        estimate = ceil((deposit_cv / 0.05) ** 2)
        if estimate <= budget:
            values["Rounds / samples (5% count RSE)"] = tr("Deposit counts stable at current budget")
        elif sample_limit >= 1 << 24:
            values["Rounds / samples (5% count RSE)"] = tr("At capacity")
        else:
            samples = min(1 << 24, estimate * interval)
            values["Rounds / samples (5% count RSE)"] = f"{ceil(samples / interval):,} / {samples:,}"
    return values


def draw_photon_size_hint(context, layout):
    from bpy.app.translations import pgettext_iface

    header, body = layout.panel("cycles_photon_sampling_advice", default_closed=True)
    header.label(text="Photon Settings Advice", text_ctxt=photon_caustics_context)
    if body is None:
        return
    tr = lambda text: pgettext_iface(text, photon_caustics_context)
    values = _photon_hint_values(context.scene, _photon_size_probe(context), tr)
    # Keep the same rows before the first probe, during IPR and after it stops.
    for label in (
        "Probe status", "Coverage evidence", "IPR sample budget", "Camera samples per emission round",
        "Emission rounds / budget", "Probed rounds / device maps", "Stored / attempted (all devices)",
        "Emitted per device / round", "Surface deposits / round", "Volume deposits / round", "Isolated / tested",
        "Subpixel / in-frame isolated", "Helped by widening / isolated", "Deposit count variation",
        "Suggested gather radius", "Suggested photons / round",
        "Rounds / samples (5% count RSE)",
    ):
        split = body.split(factor=0.54)
        split.label(text=label, text_ctxt=photon_caustics_context)
        split.label(text=values.get(label, "—"), translate=False)

    title, help_body = body.panel("cycles_photon_count_estimate", default_closed=True)
    title.label(text="How the photon count is estimated", text_ctxt=photon_caustics_context)
    if help_body is not None:
        help_body.label_multiline(
            text="An isolated deposit has fewer than 1.75 effective overlapping kernels. "
                 "It can still have neighbors. With the scene and radius held fixed, a local "
                 "Poisson approximation estimates a trial count aiming for half the tested "
                 "deposits to be isolated.", text_ctxt=photon_caustics_context)
        help_body.label_multiline(
            text="The trial uses a conservative lower estimate of the isolated fraction. "
                 "It is limited to three times the current count and the device capacity. "
                 "This estimates surface coverage. Evaluate image noise separately.",
            text_ctxt=photon_caustics_context)


class CYCLES_RENDER_PT_photon_footprint(CyclesButtonsPanel, Panel):
    bl_label = "Photon Behavior"
    bl_translation_context = photon_caustics_context
    bl_parent_id = "CYCLES_RENDER_PT_light_paths_caustics"

    def draw_header_preset(self, context):
        CYCLES_PT_photon_presets.draw_panel_header(self.layout)

    @classmethod
    def poll(cls, context):
        return super().poll(context) and context.scene.cycles.caustics_method == 'PHOTON'

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False
        layout.active = photon_caustics_unavailable_reason(context) is None
        cscene = context.scene.cycles
        differential = cscene.photon_caustics_footprint == 'DIFFERENTIAL'
        col = layout.column(align=True)
        col.prop(cscene, "photon_caustics_footprint")
        if differential:
            col.prop(cscene, "use_photon_spectral_differentials")
        col.prop(cscene, "photon_caustics_radius")
        if differential:
            col.prop(cscene, "photon_caustics_min_radius")
        col.prop(cscene, "photon_caustics_roughness_spread")
        col.prop(cscene, "photon_caustics_gather_max")
        layout.separator()
        col = layout.column(align=True)
        row = col.row()
        row.active = not differential
        row.prop(cscene, "photon_caustics_scatter")
        if differential:
            col.prop(cscene, "photon_caustics_differential_scatter")
        layout.separator()
        col = layout.column(align=True)
        col.prop(cscene, "photon_caustics_noise_strength")
        if differential:
            col.prop(cscene, "photon_caustics_ellipse_noise_strength")


class CYCLES_RENDER_PT_photon_adaptive_reconstruction(CyclesButtonsPanel, Panel):
    bl_label = "Adaptive Photon Reconstruction"
    bl_translation_context = photon_caustics_context
    bl_parent_id = "CYCLES_RENDER_PT_photon_footprint"
    bl_options = {'DEFAULT_CLOSED'}

    @classmethod
    def poll(cls, context):
        return super().poll(context) and context.scene.cycles.caustics_method == 'PHOTON'

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False
        layout.active = photon_caustics_unavailable_reason(context) is None
        cscene = context.scene.cycles
        layout.prop(cscene, "use_photon_adaptive_radius")
        if cscene.use_photon_adaptive_radius:
            col = layout.column(align=True)
            col.prop(cscene, "photon_reconstruction_method")
            col.prop(cscene, "photon_adaptive_neighbors")
            col.prop(cscene, "photon_adaptive_max_scale")
            col = layout.column()
            col.active = cscene.photon_adaptive_max_scale > 1
            col.prop(cscene, "photon_reconstruction_pilot_scale")
            if cscene.photon_reconstruction_method == 'ELLIPSE_COVERAGE':
                cell_row = col.row()
                cell_row.active = cscene.photon_reconstruction_pilot_scale > 1
                cell_row.prop(cscene, "use_photon_cell_statistics")
                neural_row = col.row()
                neural_row.active = cell_row.active and cscene.use_photon_cell_statistics
                neural_row.prop(cscene, "use_photon_neural_bandwidth")
                col.prop(cscene, "use_photon_neighbor_selection")
                if cscene.use_photon_neighbor_selection:
                    col.prop(cscene, "photon_neighbor_samples")
            if cscene.photon_reconstruction_method != 'LEGACY':
                col.prop(cscene, "use_photon_beam_continuity")
            col.prop(cscene, "photon_reconstruction_filter")
            if (cscene.photon_reconstruction_filter in {'NORMAL', 'PATH_NORMAL'} or
                    (cscene.photon_reconstruction_method != 'LEGACY' and cscene.use_photon_beam_continuity)):
                col.prop(cscene, "photon_reconstruction_normal_angle")
            layout.separator()
            layout.prop(cscene, "use_photon_gradient_guard")
            row = layout.row()
            row.active = cscene.use_photon_gradient_guard
            row.prop(cscene, "photon_gradient_strength")
            if (cscene.photon_reconstruction_method == 'ELLIPSE_COVERAGE' and
                    cscene.use_photon_cell_statistics and cscene.photon_adaptive_max_scale > 1 and
                    cscene.photon_reconstruction_pilot_scale > 1):
                layout.label(text="Cell statistics and adaptive queries per emission round",
                             text_ctxt=photon_caustics_context, icon='INFO')
            else:
                layout.label(text="One neighborhood pass per emission round", text_ctxt=photon_caustics_context, icon='INFO')


class CYCLES_RENDER_PT_photon_guiding(CyclesButtonsPanel, Panel):
    bl_label = "Photon Guiding"
    bl_translation_context = photon_caustics_context
    bl_parent_id = "CYCLES_RENDER_PT_light_paths_caustics"
    bl_options = {'DEFAULT_CLOSED'}

    @classmethod
    def poll(cls, context):
        return super().poll(context) and context.scene.cycles.caustics_method == 'PHOTON'

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False
        layout.active = photon_caustics_unavailable_reason(context) is None
        cscene = context.scene.cycles
        layout.prop(cscene, "use_photon_guiding")
        layout.prop(cscene, "use_photon_area_guiding")
        if cscene.use_photon_guiding:
            layout.label(text="Guides Area, Sun and zero-radius Point/Spot", text_ctxt=photon_caustics_context, icon='INFO')
        if cscene.use_photon_area_guiding:
            layout.label(text="Learns textured emission and visible contribution", text_ctxt=photon_caustics_context, icon='INFO')
        if cscene.use_photon_guiding or cscene.use_photon_area_guiding:
            layout.label(text="Zero Spread uses position guiding only", text_ctxt=photon_caustics_context)


class CYCLES_RENDER_PT_light_paths_fast_gi(CyclesButtonsPanel, Panel):
    bl_label = "Fast GI Approximation"
    bl_options = {'DEFAULT_CLOSED'}
    bl_parent_id = "CYCLES_RENDER_PT_light_paths"

    def draw_header(self, context):
        scene = context.scene
        cscene = scene.cycles

        self.layout.prop(cscene, "use_fast_gi", text="")

    def draw(self, context):
        scene = context.scene
        cscene = scene.cycles
        world = scene.world

        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False

        layout.active = cscene.use_fast_gi

        col = layout.column(align=True)
        col.prop(cscene, "fast_gi_method", text="Method")

        if world:
            light = world.light_settings
            col = layout.column(align=True)
            col.prop(light, "ao_factor", text="AO Factor")
            col.prop(light, "distance", text="AO Distance")

        if cscene.fast_gi_method == 'REPLACE':
            col = layout.column(align=True)
            col.prop(cscene, "ao_bounces", text="Viewport Bounces")
            col.prop(cscene, "ao_bounces_render", text="Render Bounces")


class CYCLES_RENDER_PT_motion_blur(CyclesButtonsPanel, Panel):
    bl_label = "Motion Blur"
    bl_options = {'DEFAULT_CLOSED'}

    def draw_header(self, context):
        rd = context.scene.render

        self.layout.prop(rd, "use_motion_blur", text="")

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False

        scene = context.scene
        cscene = scene.cycles
        rd = scene.render
        layout.active = rd.use_motion_blur

        col = layout.column()
        col.prop(rd, "motion_blur_position", text="Position")
        col.prop(rd, "motion_blur_shutter")
        col.prop(cscene, "use_relative_motion_blur")
        col.prop(cscene, "use_photon_motion_blur_correction")
        col.separator()
        col.prop(cscene, "rolling_shutter_type", text="Rolling Shutter")
        sub = col.column()
        sub.active = cscene.rolling_shutter_type != 'NONE'
        sub.prop(cscene, "rolling_shutter_duration")


class CYCLES_RENDER_PT_motion_blur_curve(CyclesButtonsPanel, Panel):
    bl_label = "Shutter Curve"
    bl_parent_id = "CYCLES_RENDER_PT_motion_blur"
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False

        scene = context.scene
        rd = scene.render
        layout.active = rd.use_motion_blur

        col = layout.column()

        col.template_curve_mapping(rd, "motion_blur_shutter_curve")

        col = layout.column(align=True)
        row = col.row(align=True)
        row.operator("render.shutter_curve_preset", icon='SMOOTHCURVE', text="").shape = 'SMOOTH'
        row.operator("render.shutter_curve_preset", icon='SPHERECURVE', text="").shape = 'ROUND'
        row.operator("render.shutter_curve_preset", icon='ROOTCURVE', text="").shape = 'ROOT'
        row.operator("render.shutter_curve_preset", icon='SHARPCURVE', text="").shape = 'SHARP'
        row.operator("render.shutter_curve_preset", icon='LINCURVE', text="").shape = 'LINE'
        row.operator("render.shutter_curve_preset", icon='NOCURVE', text="").shape = 'MAX'


class CYCLES_RENDER_PT_film(CyclesButtonsPanel, Panel):
    bl_label = "Film"
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False
        scene = context.scene
        cscene = scene.cycles

        col = layout.column()
        col.prop(cscene, "film_exposure")


class CYCLES_RENDER_PT_film_transparency(CyclesButtonsPanel, Panel):
    bl_label = "Transparent"
    bl_parent_id = "CYCLES_RENDER_PT_film"

    def draw_header(self, context):
        layout = self.layout

        scene = context.scene
        rd = scene.render

        layout.prop(rd, "film_transparent", text="")

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False
        scene = context.scene
        rd = scene.render
        cscene = scene.cycles

        layout.active = rd.film_transparent

        col = layout.column()
        col.prop(cscene, "film_transparent_glass", text="Transparent Glass")

        sub = col.column()
        sub.active = rd.film_transparent and cscene.film_transparent_glass
        sub.prop(cscene, "film_transparent_roughness", text="Roughness Threshold")


class CYCLES_RENDER_PT_film_pixel_filter(CyclesButtonsPanel, Panel):
    bl_label = "Pixel Filter"
    bl_parent_id = "CYCLES_RENDER_PT_film"

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False
        scene = context.scene
        cscene = scene.cycles

        col = layout.column()
        col.prop(cscene, "pixel_filter_type", text="Type")
        if cscene.pixel_filter_type != 'BOX':
            col.prop(cscene, "filter_width", text="Width")


# REMOVED: CYCLES_RENDER_PT_film_deep_output panel
# Deep output is only available with DEEP_EXR format, which auto-enables it.
# No separate UI panel needed.


class CYCLES_RENDER_PT_performance(CyclesButtonsPanel, Panel):
    bl_label = "Performance"
    bl_options = {'DEFAULT_CLOSED'}

    def draw_header_preset(self, context):
        CYCLES_PT_performance_presets.draw_panel_header(self.layout)

    def draw(self, context):
        pass


class CYCLES_RENDER_PT_performance_compositor(CyclesButtonsPanel, CompositorPerformanceButtonsPanel, Panel):
    bl_parent_id = "CYCLES_RENDER_PT_performance"
    bl_options = {'DEFAULT_CLOSED'}


class CYCLES_RENDER_PT_performance_compositor_denoise_settings(
        CyclesButtonsPanel, CompositorDenoisePerformanceButtonsPanel, Panel):
    bl_parent_id = "CYCLES_RENDER_PT_performance_compositor"
    bl_options = {'DEFAULT_CLOSED'}


class CYCLES_RENDER_PT_performance_threads(CyclesButtonsPanel, Panel):
    bl_label = "Threads"
    bl_parent_id = "CYCLES_RENDER_PT_performance"

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False

        scene = context.scene
        rd = scene.render

        col = layout.column()

        col.prop(rd, "threads_mode")
        sub = col.column(align=True)
        sub.enabled = rd.threads_mode == 'FIXED'
        sub.prop(rd, "threads")


def draw_photon_cache_settings(layout, scene):
    cscene = scene.cycles
    separate = (cscene.caustics_method == 'PHOTON' and cscene.photon_caustics_separate_sampling and
                not photon_caustics_uses_deep_output(scene))
    row = layout.row()
    row.active = separate and cscene.photon_cache_order != 'SIMULTANEOUS'
    row.prop(cscene, "use_photon_cache")
    order = layout.row()
    order.active = separate
    order.prop(cscene, "photon_cache_order")


class CYCLES_RENDER_PT_performance_memory(CyclesButtonsPanel, Panel):
    bl_label = "Memory"
    bl_parent_id = "CYCLES_RENDER_PT_performance"

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False

        scene = context.scene
        cscene = scene.cycles

        if use_vcm(scene, context.view_layer):
            layout.label(text="VCM accumulates light paths across the full frame", icon='INFO')
            return
        layout.prop(cscene, "use_auto_tile")
        tiles = layout.row()
        tiles.active = cscene.use_auto_tile
        tiles.prop(cscene, "tile_size")
        draw_photon_cache_settings(layout, scene)
        layout.prop(cscene, "deep_simple_mode")
        layout.prop(cscene, "deep_tile_budget_mb")
        if (cscene.deep_simple_mode and cscene.caustics_method == 'PHOTON' and
                photon_caustics_uses_deep_output(scene)):
            layout.prop(cscene, "deep_photon_samples")


class CYCLES_RENDER_PT_performance_texture_cache(CyclesButtonsPanel, Panel):
    bl_label = "Texture Cache"
    bl_parent_id = "CYCLES_RENDER_PT_performance"

    def draw_header(self, context):
        rd = context.scene.render
        self.layout.prop(rd, "use_texture_cache", text="")

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False

        rd = context.scene.render

        col = layout.column()
        col.active = rd.use_texture_cache

        col.prop(rd, "use_auto_generate_texture_cache", text="Auto Generate")

        row = col.split(factor=col.property_split_factor)
        row.label()
        sub = row.row(align=True)
        sub.operator("render.generate_texture_cache", text="Generate")
        sub.operator("render.clear_texture_cache", text="Clear")

        prefs = context.preferences
        if prefs.experimental.use_cycles_debug and prefs.view.show_developer_ui:
            cscene = context.scene.cycles
            col = layout.column(heading="Debug")
            col.active = rd.use_texture_cache
            col.prop(cscene, "debug_use_texture_cache_eviction")
            sub = col.column()
            sub.active = cscene.debug_use_texture_cache_eviction
            sub.prop(cscene, "debug_texture_cache_preserve_unused")


class CYCLES_RENDER_PT_performance_acceleration_structure(CyclesButtonsPanel, Panel):
    bl_label = "Acceleration Structure"
    bl_parent_id = "CYCLES_RENDER_PT_performance"

    @classmethod
    def poll(cls, context):
        return not use_optix(context) or use_multi_device(context)

    def draw(self, context):
        import _cycles

        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False

        scene = context.scene
        cscene = scene.cycles

        col = layout.column()

        use_embree = _cycles.with_embree

        if use_cpu(context):
            col.prop(cscene, "debug_use_spatial_splits")
            if use_embree:
                col.prop(cscene, "debug_use_compact_bvh")
            else:
                sub = col.column()
                sub.active = not cscene.debug_use_spatial_splits
                sub.prop(cscene, "debug_bvh_time_steps")

                col.prop(cscene, "debug_use_hair_bvh")

                sub = col.column(align=True)
                sub.label(text="Cycles built without Embree support")
                sub.label(text="CPU raytracing performance will be poor")
        else:
            col.prop(cscene, "debug_use_spatial_splits")
            sub = col.column()
            sub.active = not cscene.debug_use_spatial_splits
            sub.prop(cscene, "debug_bvh_time_steps")

            col.prop(cscene, "debug_use_hair_bvh")

            # CPU is used in addition to a GPU
            if use_multi_device(context) and use_embree:
                col.prop(cscene, "debug_use_compact_bvh")


class CYCLES_RENDER_PT_performance_final_render(CyclesButtonsPanel, Panel):
    bl_label = "Final Render"
    bl_parent_id = "CYCLES_RENDER_PT_performance"

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False

        scene = context.scene
        rd = scene.render

        col = layout.column()

        col.prop(rd, "use_persistent_data", text="Persistent Data")


class CYCLES_RENDER_PT_performance_viewport(CyclesButtonsPanel, Panel):
    bl_label = "Viewport"
    bl_parent_id = "CYCLES_RENDER_PT_performance"

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False

        scene = context.scene
        rd = scene.render
        cscene = scene.cycles

        col = layout.column()
        col.prop(rd, "preview_pixel_size", text="Pixel Size")


class CYCLES_RENDER_PT_filter(CyclesButtonsPanel, Panel):
    bl_label = "Filter"
    bl_options = {'DEFAULT_CLOSED'}
    bl_context = "view_layer"

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False

        scene = context.scene
        rd = scene.render
        view_layer = context.view_layer

        col = layout.column(heading="Include")
        col.prop(view_layer, "use_sky", text="Environment")
        col.prop(view_layer, "use_solid", text="Surfaces")
        col.prop(view_layer, "use_strand", text="Curves")
        col.prop(view_layer, "use_volumes", text="Volumes")
        col.prop(view_layer, "use_grease_pencil", text="Grease Pencil")

        col = layout.column(heading="Use")
        sub = col.row()
        sub.prop(view_layer, "use_motion_blur", text="Motion Blur")
        sub.active = rd.use_motion_blur
        sub = col.row()
        sub.prop(view_layer.cycles, "use_denoising", text="Denoising")
        sub.active = scene.cycles.use_denoising


class CYCLES_RENDER_PT_passes(CyclesButtonsPanel, Panel):
    bl_label = "Passes"
    bl_context = "view_layer"

    def draw(self, context):
        if use_vcm(context.scene, context.view_layer):
            self.layout.label(text="Available VCM passes are shown below", icon='INFO')


class CYCLES_RENDER_PT_passes_data(CyclesButtonsPanel, Panel):
    bl_label = "Data"
    bl_context = "view_layer"
    bl_parent_id = "CYCLES_RENDER_PT_passes"

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False

        scene = context.scene
        rd = scene.render
        view_layer = context.view_layer
        cycles_view_layer = view_layer.cycles

        if use_vcm(scene, view_layer):
            layout.prop(cycles_view_layer, "pass_debug_sample_count", text="Sample Count")
            return

        col = layout.column(heading="Include", align=True)
        col.prop(view_layer, "use_pass_combined")
        col.prop(view_layer, "use_pass_z")
        col.prop(view_layer, "use_pass_mist")
        col.prop(view_layer, "use_pass_position")
        col.prop(view_layer, "use_pass_normal")
        sub = col.column()
        sub.active = not rd.use_motion_blur
        sub.prop(view_layer, "use_pass_vector")
        col.prop(view_layer, "use_pass_uv")
        col.prop(view_layer, "use_pass_grease_pencil", text="Grease Pencil")

        prefs = context.preferences
        use_debug = prefs.experimental.use_cycles_debug and prefs.view.show_developer_ui
        if use_debug:
            col = layout.column(heading="Denoising", align=True)
            col.prop(cycles_view_layer, "denoising_store_passes", text="Data Passes")
            sub = col.column()
            sub.active = cycles_view_layer.denoising_store_passes
            sub.prop(cycles_view_layer, "denoising_store_motion_passes", text="Motion")
            sub.prop(cycles_view_layer, "denoising_pass_follow_reflections", text="Follow Reflections")
            sub.prop(
                cycles_view_layer,
                "denoising_pass_use_albedo_roughness_weighting",
                text="Albedo Roughness Weighting")
        else:
            col.prop(cycles_view_layer, "denoising_store_passes", text="Denoising Data")

        col = layout.column(heading="Indexes", align=True)
        col.prop(view_layer, "use_pass_object_index")
        col.prop(view_layer, "use_pass_material_index")

        col = layout.column(heading="Debug", align=True)
        col.prop(cycles_view_layer, "pass_debug_sample_count", text="Sample Count")

        # Render Time pass - disabled for GPU devices
        scene = context.scene
        cscene = scene.cycles
        row = col.row()
        row.enabled = (cscene.device == 'CPU')
        row.prop(cycles_view_layer, "pass_render_time", text="Render Time")

        layout.prop(view_layer, "pass_alpha_threshold")


class CYCLES_RENDER_PT_passes_light(CyclesButtonsPanel, Panel):
    bl_label = "Light"
    bl_context = "view_layer"
    bl_parent_id = "CYCLES_RENDER_PT_passes"

    def draw(self, context):
        if use_vcm(context.scene, context.view_layer):
            self.layout.prop(context.view_layer, "use_pass_emit")
            self.layout.prop(context.view_layer, "use_pass_environment")
            return
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False

        view_layer = context.view_layer
        cycles_view_layer = view_layer.cycles

        col = layout.column(heading="Diffuse", align=True)
        col.prop(view_layer, "use_pass_diffuse_direct", text="Direct")
        col.prop(view_layer, "use_pass_diffuse_indirect", text="Indirect")
        col.prop(view_layer, "use_pass_diffuse_color", text="Color")

        col = layout.column(heading="SSS", align=True)
        col.prop(view_layer, "use_pass_subsurface_direct", text="Direct")
        col.prop(view_layer, "use_pass_subsurface_indirect", text="Indirect")
        col.prop(view_layer, "use_pass_subsurface_color", text="Color")

        col = layout.column(heading="Glossy", align=True)
        col.prop(view_layer, "use_pass_glossy_direct", text="Direct")
        col.prop(view_layer, "use_pass_glossy_indirect", text="Indirect")
        col.prop(view_layer, "use_pass_glossy_color", text="Color")

        col = layout.column(heading="Transmission", align=True)
        col.prop(view_layer, "use_pass_transmission_direct", text="Direct")
        col.prop(view_layer, "use_pass_transmission_indirect", text="Indirect")
        col.prop(view_layer, "use_pass_transmission_color", text="Color")

        col = layout.column(heading="Volume", heading_ctxt=i18n_contexts.id_id, align=True)
        col.prop(cycles_view_layer, "use_pass_volume_direct", text="Direct")
        col.prop(cycles_view_layer, "use_pass_volume_indirect", text="Indirect")

        prefs = context.preferences
        use_debug = prefs.experimental.use_cycles_debug and prefs.view.show_developer_ui
        if use_debug:
            col.prop(cycles_view_layer, "use_pass_volume_scatter", text="Scatter")
            col.prop(cycles_view_layer, "use_pass_volume_transmit", text="Transmit")
            col.prop(cycles_view_layer, "use_pass_volume_majorant", text="Majorant")

        col = layout.column(heading="Other", align=True)
        col.prop(cycles_view_layer, "use_pass_caustics")
        col.prop(view_layer, "use_pass_emit", text="Emission")
        col.prop(view_layer, "use_pass_environment")
        col.prop(view_layer, "use_pass_ambient_occlusion", text="Ambient Occlusion")
        col.prop(cycles_view_layer, "use_pass_shadow_catcher")


class CYCLES_RENDER_PT_passes_crypto(CyclesButtonsPanel, ViewLayerCryptomattePanelHelper, Panel):
    bl_label = "Cryptomatte"
    bl_context = "view_layer"
    bl_parent_id = "CYCLES_RENDER_PT_passes"

    @classmethod
    def poll(cls, context):
        return super().poll(context) and not use_vcm(context.scene, context.view_layer)



class CYCLES_RENDER_PT_passes_aov(CyclesButtonsPanel, ViewLayerAOVPanelHelper, Panel):
    bl_label = "Shader AOV"
    bl_context = "view_layer"
    bl_parent_id = "CYCLES_RENDER_PT_passes"

    @classmethod
    def poll(cls, context):
        return super().poll(context) and not use_vcm(context.scene, context.view_layer)



class CYCLES_RENDER_PT_passes_lightgroups(CyclesButtonsPanel, ViewLayerLightgroupsPanelHelper, Panel):
    bl_label = "Light Groups"
    bl_context = "view_layer"
    bl_parent_id = "CYCLES_RENDER_PT_passes"


class CYCLES_RENDER_PT_passes_lightgroups_light_pass_aovs(CyclesButtonsPanel, Panel):
    bl_label = "Light Pass AOVs"
    bl_context = "view_layer"
    bl_parent_id = "CYCLES_RENDER_PT_passes_lightgroups"

    bl_options = {'DEFAULT_CLOSED'}

    @classmethod
    def poll(cls, context):
        return (super().poll(context) and context.view_layer.active_lightgroup is not None and
                not use_vcm(context.scene, context.view_layer))

    def draw(self, context):
        group = context.view_layer.active_lightgroup
        layout = self.layout
        layout.use_property_decorate = False
        layout.label(text=group.name, translate=False, icon='LIGHT_DATA')
        lobes = ("Diffuse", "Glossy", "Transmission", "Subsurface", "Volume")
        table = layout.split(factor=0.30)
        labels = table.column(align=True)
        labels.label(text="")
        for lobe in lobes:
            labels.label(text=lobe)
        columns = table.row(align=True)
        for component in ("All", "Combined", "Direct", "Indirect"):
            col = columns.column(align=True)
            col.alignment = 'CENTER'
            col.label(text=component)
            for lobe in lobes:
                col.prop(group, f"use_pass_{lobe.lower()}_{component.lower()}", text="")
        layout.prop(group, "use_pass_photon")


class CYCLES_PT_post_processing(CyclesButtonsPanel, Panel):
    bl_label = "Post Processing"
    bl_options = {'DEFAULT_CLOSED'}
    bl_context = "output"

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False

        rd = context.scene.render

        col = layout.column(align=True, heading="Pipeline")
        col.prop(rd, "use_compositing")
        col.prop(rd, "use_sequencer")

        layout.prop(rd, "dither_intensity", text="Dither", slider=True)


class CYCLES_CAMERA_PT_dof(CyclesButtonsPanel, Panel):
    bl_label = "Depth of Field"
    bl_context = "data"

    @classmethod
    def poll(cls, context):
        return context.camera and CyclesButtonsPanel.poll(context)

    def draw_header(self, context):
        cam = context.camera
        dof = cam.dof
        self.layout.prop(dof, "use_dof", text="")

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True

        cam = context.camera
        dof = cam.dof
        layout.active = dof.use_dof

        split = layout.split()

        col = split.column()
        col.prop(dof, "focus_object", text="Focus on Object")
        if dof.focus_object and dof.focus_object.type == 'ARMATURE':
            col.prop_search(dof, "focus_subtarget", dof.focus_object.data, "bones", text="Focus Bone")

        sub = col.row()
        sub.active = dof.focus_object is None
        sub.prop(dof, "focus_distance", text="Focus Distance")
        sub.operator(
            "ui.eyedropper_depth",
            icon='EYEDROPPER',
            text="").prop_data_path = "scene.camera.data.dof.focus_distance"


class CYCLES_CAMERA_PT_dof_aperture(CyclesButtonsPanel, Panel):
    bl_label = "Aperture"
    bl_parent_id = "CYCLES_CAMERA_PT_dof"

    @classmethod
    def poll(cls, context):
        return context.camera and CyclesButtonsPanel.poll(context)

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True

        cam = context.camera
        dof = cam.dof
        layout.active = dof.use_dof
        flow = layout.grid_flow(row_major=True, columns=0, even_columns=True, even_rows=False, align=False)

        col = flow.column()
        col.prop(dof, "aperture_fstop")
        col.prop(dof, "aperture_blades")
        col.prop(dof, "aperture_rotation")
        col.prop(dof, "aperture_ratio")


class CYCLES_CAMERA_PT_lens_custom_parameters(CyclesButtonsPanel, Panel):
    bl_label = "Parameters"
    bl_parent_id = "DATA_PT_lens"

    @classmethod
    def poll(cls, context):
        cam = context.camera
        return (super().poll(context) and
                cam and
                cam.type == 'CUSTOM' and
                len(cam.cycles_custom.keys()) > 0)

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True

        cam = context.camera
        ccam = cam.cycles_custom

        col = layout.column()
        for key in ccam.keys():
            col.prop(ccam, f'["{key}"]', text=bpy.path.display_name(key))


class CYCLES_PT_context_material(CyclesButtonsPanel, Panel):
    bl_label = ""
    bl_context = "material"
    bl_options = {'HIDE_HEADER'}

    @classmethod
    def poll(cls, context):
        if context.active_object and context.active_object.type == 'GREASEPENCIL':
            return False
        else:
            return (context.material or context.object) and CyclesButtonsPanel.poll(context)

    def draw(self, context):
        layout = self.layout

        mat = context.material
        ob = context.object
        slot = context.material_slot
        space = context.space_data

        if ob:
            is_sortable = len(ob.material_slots) > 1
            rows = 3
            if (is_sortable):
                rows = 4

            row = layout.row()

            row.template_list("MATERIAL_UL_matslots", "", ob, "material_slots", ob, "active_material_index", rows=rows)

            col = row.column(align=True)
            col.operator("object.material_slot_add", icon='ADD', text="")
            col.operator("object.material_slot_remove", icon='REMOVE', text="")
            col.separator()
            col.menu("MATERIAL_MT_context_menu", icon='DOWNARROW_HLT', text="")

            if is_sortable:
                col.separator()

                col.operator("object.material_slot_move", icon='TRIA_UP', text="").direction = 'UP'
                col.operator("object.material_slot_move", icon='TRIA_DOWN', text="").direction = 'DOWN'

            if ob.mode == 'EDIT':
                row = layout.row(align=True)
                row.operator("object.material_slot_assign", text="Assign")
                row.operator("object.material_slot_select", text="Select")
                row.operator("object.material_slot_deselect", text="Deselect")

        row = layout.row()

        if ob:
            row.template_ID(ob, "active_material", new="material.new")

            if slot:
                icon_link = 'MESH_DATA' if slot.link == 'DATA' else 'OBJECT_DATA'
                row.prop(slot, "link", text="", icon=icon_link, icon_only=True)

        elif mat:
            layout.template_ID(space, "pin_id")
            layout.separator()


class CYCLES_OBJECT_PT_motion_blur(CyclesButtonsPanel, Panel):
    bl_label = "Motion Blur"
    bl_context = "object"
    bl_options = {'DEFAULT_CLOSED'}

    @classmethod
    def poll(cls, context):
        ob = context.object
        if CyclesButtonsPanel.poll(context) and ob:
            if ob.type in {'MESH', 'CURVE', 'CURVE', 'SURFACE', 'FONT',
                           'META', 'CAMERA', 'CURVES', 'POINTCLOUD', 'VOLUME'}:
                return True
            if ob.instance_type == 'COLLECTION' and ob.instance_collection:
                return True
            # TODO(sergey): More duplicator types here?
        return False

    def draw_header(self, context):
        layout = self.layout

        rd = context.scene.render
        # scene = context.scene

        layout.active = rd.use_motion_blur

        ob = context.object
        cob = ob.cycles

        layout.prop(cob, "use_motion_blur", text="")

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True

        rd = context.scene.render
        # scene = context.scene

        ob = context.object
        cob = ob.cycles

        layout.active = (rd.use_motion_blur and cob.use_motion_blur)

        col = layout.column()
        col.prop(cob, "motion_steps", text="Steps")
        if ob.type != 'CAMERA':
            col.prop(cob, "use_deform_motion", text="Deformation")


class CYCLES_OBJECT_PT_shading_gi_approximation(CyclesButtonsPanel, Panel):
    bl_label = "Fast GI Approximation"
    bl_parent_id = "OBJECT_PT_shading"
    bl_context = "object"

    @classmethod
    def poll(cls, context):
        return CyclesButtonsPanel.poll(context) and context.object.type != 'LIGHT'

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True

        scene = context.scene
        ob = context.object

        cob = ob.cycles
        cscene = scene.cycles

        col = layout.column()
        col.active = cscene.use_fast_gi
        col.prop(cob, "ao_distance")


class CYCLES_OBJECT_PT_shading_caustics(CyclesButtonsPanel, Panel):
    bl_label = "Caustics"
    bl_parent_id = "OBJECT_PT_shading"
    bl_context = "object"

    @classmethod
    def poll(cls, context):
        return CyclesButtonsPanel.poll(context) and context.object.type != 'LIGHT'

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False

        col = layout.column()

        ob = context.object
        cob = ob.cycles
        col.prop(cob, "is_caustics_caster")
        if context.scene.cycles.caustics_method == 'PHOTON':
            sub = col.column()
            sub.active = cob.is_caustics_caster
            sub.prop(cob, "photon_caustics_geometry_normal")
            sub.prop(cob, "photon_caustics_zero_roughness")
            col.label(text="Photon reception is automatic", text_ctxt=photon_caustics_context, icon='INFO')
        else:
            col.prop(cob, "is_caustics_receiver")


class CYCLES_OBJECT_PT_nested_dielectric(CyclesButtonsPanel, Panel):
    bl_label = "Nested Dielectric"
    bl_parent_id = "OBJECT_PT_shading"
    bl_context = "object"
    bl_options = {'DEFAULT_CLOSED'}

    @classmethod
    def poll(cls, context):
        return CyclesButtonsPanel.poll(context) and context.object.type in {
            'MESH', 'CURVE', 'SURFACE', 'META', 'FONT', 'CURVES', 'POINTCLOUD', 'VOLUME', 'EMPTY'}

    def draw_header(self, context):
        self.layout.prop(context.object.cycles, "use_nested_dielectric", text="")

    def draw(self, context):
        self.layout.use_property_split = True
        self.layout.use_property_decorate = False
        self.layout.active = context.object.cycles.use_nested_dielectric
        self.layout.label(text="Set Medium Priority in material settings", icon='INFO')
        self.layout.label(text="Every instancer must opt in; adds medium tracking cost")


class CYCLES_OBJECT_PT_lightgroup(CyclesButtonsPanel, Panel):
    bl_label = "Light Group"
    bl_parent_id = "OBJECT_PT_shading"
    bl_context = "object"
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True

        ob = context.object

        view_layer = context.view_layer

        row = layout.row(align=True)
        row.use_property_decorate = False

        sub = row.column(align=True)
        sub.prop_search(ob, "lightgroup", view_layer, "lightgroups", text="Light Group", results_are_suggestions=True)

        sub = row.column(align=True)
        sub.enabled = bool(ob.lightgroup) and not any(lg.name == ob.lightgroup for lg in view_layer.lightgroups)
        sub.operator("scene.view_layer_add_lightgroup", icon='ADD', text="").name = ob.lightgroup


class CYCLES_OBJECT_PT_visibility(CyclesButtonsPanel, Panel):
    bl_label = "Visibility"
    bl_context = "object"
    bl_options = {'DEFAULT_CLOSED'}

    @classmethod
    def poll(cls, context):
        return CyclesButtonsPanel.poll(context) and (context.object)

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True

        ob = context.object

        layout.prop(ob, "hide_select", text="Selectable", invert_checkbox=True, toggle=False)
        layout.prop(ob, "hide_surface_pick", text="Surface Picking", toggle=False, invert_checkbox=True)

        col = layout.column(heading="Show In")
        col.prop(ob, "hide_viewport", text="Viewports", invert_checkbox=True, toggle=False)
        col.prop(ob, "hide_render", text="Renders", invert_checkbox=True, toggle=False)

        if has_geometry_visibility(ob):
            col = layout.column(heading="Mask")
            col.prop(ob, "is_shadow_catcher")
            col.prop(ob, "is_holdout", toggle=False)


class CYCLES_OBJECT_PT_visibility_ray_visibility(CyclesButtonsPanel, Panel):
    bl_label = "Ray Visibility"
    bl_parent_id = "CYCLES_OBJECT_PT_visibility"
    bl_context = "object"

    @classmethod
    def poll(cls, context):
        ob = context.object
        return CyclesButtonsPanel.poll(context) and has_geometry_visibility(ob)

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False

        scene = context.scene
        ob = context.object

        col = layout.column()
        col.prop(ob, "visible_camera", text="Camera", toggle=False)
        col.prop(ob, "visible_diffuse", text="Diffuse")
        col.prop(ob, "visible_glossy", text="Glossy")
        col.prop(ob, "visible_transmission", text="Transmission")
        col.prop(ob, "visible_volume_scatter", text="Volume Scatter")

        if ob.type == 'LIGHT' and scene.cycles.caustics_method == 'PHOTON':
            col.prop(ob.cycles, "visible_photon")

        if ob.type != 'LIGHT':
            sub = col.column()
            sub.prop(ob, "visible_shadow", text="Shadow")
            sub.prop(ob, "visible_raycast", text="Raycast")


class CYCLES_OBJECT_PT_visibility_culling(CyclesButtonsPanel, Panel):
    bl_label = "Culling"
    bl_parent_id = "CYCLES_OBJECT_PT_visibility"
    bl_context = "object"

    @classmethod
    def poll(cls, context):
        ob = context.object
        return CyclesButtonsPanel.poll(context) and has_geometry_visibility(ob)

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False

        scene = context.scene
        cscene = scene.cycles
        ob = context.object
        cob = ob.cycles

        row = layout.row()
        row.active = scene.render.use_simplify and cscene.use_camera_cull
        row.prop(cob, "use_camera_cull")

        row = layout.row()
        row.active = scene.render.use_simplify and cscene.use_distance_cull
        row.prop(cob, "use_distance_cull")


def panel_node_draw(layout, id_data, input_name):
    from bpy_extras.node_utils import find_node_input

    ntree = id_data.node_tree

    if ntree is None:
        return False

    node = ntree.get_output_node('CYCLES')
    if node:
        input = find_node_input(node, input_name)
        if input:
            layout.template_node_view(ntree, node, input)
        else:
            layout.label(text="Incompatible output node")
    else:
        layout.label(text="No output node")

    return True


class CYCLES_LIGHT_PT_preview(CyclesButtonsPanel, Panel):
    bl_label = "Preview"
    bl_context = "data"
    bl_options = {'DEFAULT_CLOSED'}

    @classmethod
    def poll(cls, context):
        return (
            context.light and
            not (
                context.light.type == 'AREA' and
                context.light.cycles.is_portal
            ) and
            CyclesButtonsPanel.poll(context)
        )

    def draw(self, context):
        self.layout.template_preview(context.light)


def draw_negative_caustic_light_notice(layout, light, scene):
    if (light.energy < 0 and light.cycles.is_caustics_light and
            scene.cycles.caustics_method == 'PHOTON' and
            not (light.type == 'AREA' and light.cycles.is_portal)):
        layout.label_multiline(
            text="Negative power subtracts photon caustics on paths without dispersion. "
                 "Paths with dispersion contribute zero from this light.",
            text_ctxt=photon_caustics_context, icon='INFO')


class CYCLES_LIGHT_PT_light(CyclesButtonsPanel, Panel):
    bl_label = "Light"
    bl_context = "data"

    @classmethod
    def poll(cls, context):
        return context.light and CyclesButtonsPanel.poll(context)

    def draw(self, context):
        layout = self.layout

        light = context.light

        if self.bl_space_type == 'PROPERTIES':
            layout.row().prop(light, "type", expand=True)
            layout.use_property_split = True
        else:
            layout.use_property_split = True
            layout.row().prop(light, "type")

        col = layout.column()
        heading = col.column(align=True, heading="Temperature")
        row = heading.column(align=True).row(align=True)
        row.use_property_decorate = False
        row.prop(light, "use_temperature", text="")
        # Don't show color preview for now, it is grayed out so the color
        # is not accurate. Would not a change in the UI code to allow
        # non-editable colors to be displayed as is.
        if False:  # light.use_temperature:
            sub = row.split(factor=0.7, align=True)
            sub.active = light.use_temperature
            sub.prop(light, "temperature", text="")
            sub.prop(light, "temperature_color", text="")
        else:
            sub = row.row()
            sub.active = light.use_temperature
            sub.prop(light, "temperature", text="")
            row.prop_decorator(light, "temperature")

        if light.use_temperature:
            col.prop(light, "color", text="Tint")
        else:
            col.prop(light, "color", text="Color")

        layout.separator()

        col = layout.column()
        col.prop(light, "energy")
        draw_negative_caustic_light_notice(col, light, context.scene)
        col.prop(light, "exposure")
        col.prop(light, "normalize")

        layout.separator()

        col = layout.column()
        if light.type in {'POINT', 'SPOT'}:
            col.prop(light, "shadow_soft_size", text="Radius")
            col.prop(light, "use_soft_falloff")
        elif light.type == 'SUN':
            col.prop(light, "angle")
        elif light.type == 'AREA':
            col.prop(light, "shape", text="Shape")
            sub = col.column(align=True)

            if light.shape in {'SQUARE', 'DISK'}:
                sub.prop(light, "size")
            elif light.shape in {'RECTANGLE', 'ELLIPSE'}:
                sub.prop(light, "size", text="Size X")
                sub.prop(light, "size_y", text="Y")


class CYCLES_LIGHT_PT_settings(CyclesButtonsPanel, Panel):
    bl_label = "Settings"
    bl_context = "data"
    bl_options = {'DEFAULT_CLOSED'}

    @classmethod
    def poll(cls, context):
        return context.light and CyclesButtonsPanel.poll(context)

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True

        light = context.light
        clamp = light.cycles

        col = layout.column()

        if not (light.type == 'AREA' and clamp.is_portal):
            col.separator()
            sub = col.column()
            sub.prop(clamp, "max_bounces")

        sub = col.column(align=True)
        sub.active = not (light.type == 'AREA' and clamp.is_portal)
        sub.prop(light, "use_shadow", text="Cast Shadow")
        sub.prop(light, "shadow_color", text="Shadow Color")
        sub.prop(clamp, "use_multiple_importance_sampling", text="Multiple Importance")
        sub.prop(clamp, "is_caustics_light", text="Shadow Caustics")
        draw_negative_caustic_light_notice(col, light, context.scene)
        if context.scene.cycles.caustics_method == 'PHOTON':
            photon = col.column(align=True)
            photon.active = clamp.is_caustics_light and not (light.type == 'AREA' and clamp.is_portal)
            photon.prop(clamp, "photon_strength")
            photon.prop(clamp, "photon_radius_scale")

        if light.type == 'AREA':
            col.prop(clamp, "is_portal", text="Portal")


class CYCLES_LIGHT_PT_nodes(CyclesButtonsPanel, Panel):
    bl_label = "Nodes"
    bl_context = "data"

    @classmethod
    def poll(cls, context):
        return context.light and not (context.light.type == 'AREA' and
                                      context.light.cycles.is_portal) and \
            CyclesButtonsPanel.poll(context)

    def draw(self, context):
        layout = self.layout

        layout.use_property_split = True

        light = context.light
        panel_node_draw(layout, light, 'Surface')


class CYCLES_LIGHT_PT_beam_shape(CyclesButtonsPanel, Panel):
    bl_label = "Beam Shape"
    bl_parent_id = "CYCLES_LIGHT_PT_light"
    bl_context = "data"

    @classmethod
    def poll(cls, context):
        if context.light.type in {'SPOT', 'AREA'}:
            return context.light and CyclesButtonsPanel.poll(context)

    def draw(self, context):
        layout = self.layout
        light = context.light
        layout.use_property_split = True

        col = layout.column()
        if light.type == 'SPOT':
            col.prop(light, "spot_size", text="Angle")
            col.prop(light, "spot_blend", text="Blend", slider=True)
            col.prop(light, "show_cone")
        elif light.type == 'AREA':
            col.prop(light, "spread", text="Spread")


class CYCLES_WORLD_PT_preview(CyclesButtonsPanel, Panel):
    bl_label = "Preview"
    bl_context = "world"
    bl_options = {'DEFAULT_CLOSED'}

    @classmethod
    def poll(cls, context):
        return context.world and CyclesButtonsPanel.poll(context)

    def draw(self, context):
        self.layout.template_preview(context.world)


class CYCLES_WORLD_PT_surface(CyclesButtonsPanel, Panel):
    bl_label = "Surface"
    bl_context = "world"

    @classmethod
    def poll(cls, context):
        return context.world and CyclesButtonsPanel.poll(context)

    def draw(self, context):
        layout = self.layout

        layout.use_property_split = True

        world = context.world

        if not panel_node_draw(layout, world, 'Surface'):
            layout.prop(world, "color")


class CYCLES_WORLD_PT_volume(CyclesButtonsPanel, Panel):
    bl_label = "Volume"
    bl_translation_context = i18n_contexts.id_id
    bl_context = "world"
    bl_options = {'DEFAULT_CLOSED'}

    @classmethod
    def poll(cls, context):
        world = context.world
        return world and world.node_tree and CyclesButtonsPanel.poll(context)

    def draw(self, context):
        layout = self.layout

        layout.use_property_split = True

        world = context.world
        panel_node_draw(layout, world, 'Volume')


class CYCLES_WORLD_PT_mist(CyclesButtonsPanel, Panel):
    bl_label = "Mist Pass"
    bl_context = "world"
    bl_options = {'DEFAULT_CLOSED'}

    @classmethod
    def poll(cls, context):
        if CyclesButtonsPanel.poll(context):
            if context.world:
                for view_layer in context.scene.view_layers:
                    if view_layer.use_pass_mist:
                        return True

        return False

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True

        world = context.world

        col = layout.column(align=True)
        col.prop(world.mist_settings, "start")
        col.prop(world.mist_settings, "depth")

        col = layout.column()
        col.prop(world.mist_settings, "falloff")


class CYCLES_WORLD_PT_ray_visibility(CyclesButtonsPanel, Panel):
    bl_label = "Ray Visibility"
    bl_context = "world"
    bl_options = {'DEFAULT_CLOSED'}

    @classmethod
    def poll(cls, context):
        return CyclesButtonsPanel.poll(context) and context.world

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False

        world = context.world
        visibility = world.cycles_visibility

        col = layout.column()
        col.prop(visibility, "camera")
        col.prop(visibility, "diffuse")
        col.prop(visibility, "glossy")
        col.prop(visibility, "transmission")
        col.prop(visibility, "scatter")
        if context.scene.cycles.caustics_method == 'PHOTON':
            col.prop(world.cycles, "visible_photon")


class CYCLES_WORLD_PT_settings(CyclesButtonsPanel, Panel):
    bl_label = "Settings"
    bl_context = "world"
    bl_options = {'DEFAULT_CLOSED'}

    @classmethod
    def poll(cls, context):
        return context.world and CyclesButtonsPanel.poll(context)

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False

        layout.column()


class CYCLES_WORLD_PT_settings_surface(CyclesButtonsPanel, Panel):
    bl_label = "Surface"
    bl_parent_id = "CYCLES_WORLD_PT_settings"
    bl_context = "world"

    @classmethod
    def poll(cls, context):
        return context.world and CyclesButtonsPanel.poll(context)

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False

        world = context.world
        cworld = world.cycles

        col = layout.column()
        col.prop(cworld, "sampling_method", text="Sampling")

        sub = col.column()
        sub.active = cworld.sampling_method != 'NONE'
        subsub = sub.row(align=True)
        subsub.active = cworld.sampling_method == 'MANUAL'
        subsub.prop(cworld, "sample_map_resolution")
        sub.prop(cworld, "max_bounces")
        sub.prop(cworld, "is_caustics_light", text="Shadow Caustics")
        sub.prop(cworld, "use_shadows", text="Cast Shadow")
        shadow_col = sub.column()
        shadow_col.active = cworld.use_shadows
        shadow_col.prop(cworld, "shadow_color", text="Shadow Color")


class CYCLES_WORLD_PT_photon_source(CyclesButtonsPanel, Panel):
    bl_label = "HDR Photon Source"
    bl_parent_id = "CYCLES_WORLD_PT_settings_surface"
    bl_context = "world"

    @classmethod
    def poll(cls, context):
        return (context.world and CyclesButtonsPanel.poll(context) and
                context.scene.cycles.caustics_method == 'PHOTON')

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False
        cworld = context.world.cycles
        layout.active = (cworld.is_caustics_light and cworld.visible_photon and
                         cworld.sampling_method != 'NONE')
        layout.prop(cworld, "photon_sun_threshold")
        layout.prop(cworld, "photon_sun_radius_scale")
        layout.prop(cworld, "show_photon_sun")


class CYCLES_WORLD_PT_settings_volume(CyclesButtonsPanel, Panel):
    bl_label = "Volume"
    bl_translation_context = i18n_contexts.id_id
    bl_parent_id = "CYCLES_WORLD_PT_settings"
    bl_context = "world"

    @classmethod
    def poll(cls, context):
        return context.world and CyclesButtonsPanel.poll(context)

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False

        world = context.world
        cworld = world.cycles

        col = layout.column()

        sub = col.column()
        col.prop(cworld, "volume_sampling", text="Sampling")
        col.prop(cworld, "volume_interpolation", text="Interpolation")
        if context.scene.cycles.volume_biased:
            col.prop(cworld, "volume_step_size")


class CYCLES_WORLD_PT_settings_light_group(CyclesButtonsPanel, Panel):
    bl_label = "Light Group"
    bl_parent_id = "CYCLES_WORLD_PT_settings"
    bl_context = "world"

    @classmethod
    def poll(cls, context):
        return context.world and CyclesButtonsPanel.poll(context)

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False

        world = context.world
        view_layer = context.view_layer

        row = layout.row(align=True)

        sub = row.column(align=True)
        sub.prop_search(
            world,
            "lightgroup",
            view_layer,
            "lightgroups",
            text="Light Group",
            results_are_suggestions=True,
        )

        sub = row.column(align=True)
        sub.enabled = bool(world.lightgroup) and not any(lg.name == world.lightgroup for lg in view_layer.lightgroups)
        sub.operator("scene.view_layer_add_lightgroup", icon='ADD', text="").name = world.lightgroup


class CYCLES_MATERIAL_PT_preview(CyclesButtonsPanel, Panel):
    bl_label = "Preview"
    bl_context = "material"
    bl_options = {'DEFAULT_CLOSED'}

    @classmethod
    def poll(cls, context):
        mat = context.material
        return mat and (not mat.grease_pencil) and CyclesButtonsPanel.poll(context)

    def draw(self, context):
        self.layout.template_preview(context.material)


class CYCLES_MATERIAL_PT_surface(CyclesButtonsPanel, Panel):
    bl_label = "Surface"
    bl_context = "material"

    @classmethod
    def poll(cls, context):
        mat = context.material
        return mat and (not mat.grease_pencil) and CyclesButtonsPanel.poll(context)

    def draw(self, context):
        layout = self.layout

        layout.use_property_split = True

        mat = context.material
        if not panel_node_draw(layout, mat, 'Surface'):
            layout.prop(mat, "diffuse_color")


class CYCLES_MATERIAL_PT_volume(CyclesButtonsPanel, Panel):
    bl_label = "Volume"
    bl_translation_context = i18n_contexts.id_id
    bl_context = "material"
    bl_options = {'DEFAULT_CLOSED'}

    @classmethod
    def poll(cls, context):
        mat = context.material
        return mat and (not mat.grease_pencil) and mat.node_tree and CyclesButtonsPanel.poll(context)

    def draw(self, context):
        layout = self.layout

        layout.use_property_split = True

        mat = context.material
        # cmat = mat.cycles

        panel_node_draw(layout, mat, 'Volume')


class CYCLES_MATERIAL_PT_displacement(CyclesButtonsPanel, Panel):
    bl_label = "Displacement"
    bl_context = "material"

    @classmethod
    def poll(cls, context):
        mat = context.material
        return mat and (not mat.grease_pencil) and mat.node_tree and CyclesButtonsPanel.poll(context)

    def draw(self, context):
        layout = self.layout

        layout.use_property_split = True

        mat = context.material
        panel_node_draw(layout, mat, 'Displacement')


class CYCLES_MATERIAL_PT_settings(CyclesButtonsPanel, Panel):
    bl_label = "Settings"
    bl_context = "material"
    bl_options = {'DEFAULT_CLOSED'}

    @classmethod
    def poll(cls, context):
        mat = context.material
        return mat and (not mat.grease_pencil) and CyclesButtonsPanel.poll(context)

    @staticmethod
    def draw_shared(self, mat):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False

        layout.prop(mat, "pass_index")

    def draw(self, context):
        self.draw_shared(self, context.material)


class CYCLES_MATERIAL_PT_settings_surface(CyclesButtonsPanel, Panel):
    bl_label = "Surface"
    bl_parent_id = "CYCLES_MATERIAL_PT_settings"
    bl_context = "material"

    @staticmethod
    def draw_shared(self, mat):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False

        cmat = mat.cycles

        col = layout.column()
        col.prop(mat, "displacement_method", text="Displacement")
        col.prop(cmat, "emission_sampling")
        col.prop(mat, "use_transparent_shadow")
        col.prop(cmat, "use_bump_map_correction")
        col.prop(cmat, "use_disney_glossy_terminator")

    def draw(self, context):
        self.draw_shared(self, context.material)


class CYCLES_MATERIAL_PT_settings_caustics(CyclesButtonsPanel, Panel):
    bl_label = "Caustics"
    bl_parent_id = "CYCLES_MATERIAL_PT_settings"
    bl_context = "material"

    @staticmethod
    def draw_shared(self, mat):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False
        col = layout.column(align=True)
        col.prop(mat.cycles, "caustics_reflective", text="Reflective")
        col.prop(mat.cycles, "caustics_refractive", text="Refractive")
        col.prop(mat.cycles, "photon_caustics_geometry_normal")
        col.prop(mat.cycles, "photon_caustics_zero_roughness")

    def draw(self, context):
        self.draw_shared(self, context.material)


class CYCLES_MATERIAL_PT_settings_nested_dielectric(CyclesButtonsPanel, Panel):
    bl_label = "Nested Dielectric"
    bl_parent_id = "CYCLES_MATERIAL_PT_settings"
    bl_context = "material"
    bl_options = {'DEFAULT_CLOSED'}

    @staticmethod
    def draw_shared(self, mat):
        self.layout.use_property_decorate = False
        enabled = mat.cycles.nested_dielectric_priority > 0
        row = self.layout.split(factor=0.4)
        label = row.row()
        label.alignment = 'RIGHT'
        label.active = enabled
        label.label(text="Medium Priority")
        # Zero is the enable control itself; keep the number editable.
        row.prop(mat.cycles, "nested_dielectric_priority", text="")
        row = self.layout.row()
        row.active = enabled
        row.prop(mat.cycles, "use_nested_dielectric_union")
        if not enabled:
            status = self.layout.row()
            status.active = False
            status.label(text="Disabled (Priority 0)")

    def draw(self, context):
        self.draw_shared(self, context.material)


class CYCLES_MATERIAL_PT_settings_volume(CyclesButtonsPanel, Panel):
    bl_label = "Volume"
    bl_translation_context = i18n_contexts.id_id
    bl_parent_id = "CYCLES_MATERIAL_PT_settings"
    bl_context = "material"

    @staticmethod
    def draw_shared(self, context, mat):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False

        cmat = mat.cycles

        col = layout.column()
        sub = col.column()
        col.prop(cmat, "volume_sampling", text="Sampling")
        col.prop(cmat, "volume_interpolation", text="Interpolation")
        if context.scene.cycles.volume_biased:
            col.prop(cmat, "volume_step_rate")

    def draw(self, context):
        self.draw_shared(self, context, context.material)


class CYCLES_RENDER_PT_bake(CyclesButtonsPanel, Panel):
    bl_label = "Bake"
    bl_context = "render"
    bl_options = {'DEFAULT_CLOSED'}
    COMPAT_ENGINES = {'CYCLES'}

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False  # No animation.

        scene = context.scene
        cscene = scene.cycles
        cbk = scene.render.bake

        if cbk.use_multires:
            layout.operator("object.bake_image", icon='RENDER_STILL')
            layout.prop(cbk, "use_multires")
            layout.prop(cbk, "type")

        else:
            layout.operator("object.bake", icon='RENDER_STILL').type = cscene.bake_type
            layout.prop(cbk, "use_multires")
            layout.prop(cscene, "bake_type")

        if not scene.render.bake.use_multires and cscene.bake_type not in {
                "AO", "POSITION", "NORMAL", "UV", "ROUGHNESS", "ENVIRONMENT"}:
            row = layout.row()
            row.prop(cbk, "view_from")
            row.active = scene.camera is not None


class CYCLES_RENDER_PT_bake_influence(CyclesButtonsPanel, Panel):
    bl_label = "Influence"
    bl_context = "render"
    bl_parent_id = "CYCLES_RENDER_PT_bake"
    COMPAT_ENGINES = {'CYCLES'}

    @classmethod
    def poll(cls, context):
        scene = context.scene
        cscene = scene.cycles
        rd = scene.render
        if scene.render.bake.use_multires == False and cscene.bake_type in {
                'NORMAL', 'COMBINED', 'DIFFUSE', 'GLOSSY', 'TRANSMISSION'}:
            return True

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False  # No animation.

        scene = context.scene
        cscene = scene.cycles
        cbk = scene.render.bake
        rd = scene.render

        col = layout.column()

        if cscene.bake_type == 'NORMAL':
            col.prop(cbk, "normal_space", text="Space")

            sub = col.column(align=True)
            sub.prop(cbk, "normal_r", text="Swizzle R")
            sub.prop(cbk, "normal_g", text="G", text_ctxt=i18n_contexts.color)
            sub.prop(cbk, "normal_b", text="B", text_ctxt=i18n_contexts.color)

        elif cscene.bake_type == 'COMBINED':

            col = layout.column(heading="Lighting", align=True)
            col.prop(cbk, "use_pass_direct")
            col.prop(cbk, "use_pass_indirect")

            col = layout.column(heading="Contributions", align=True)
            col.active = cbk.use_pass_direct or cbk.use_pass_indirect
            col.prop(cbk, "use_pass_diffuse")
            col.prop(cbk, "use_pass_glossy")
            col.prop(cbk, "use_pass_transmission")
            col.prop(cbk, "use_pass_emit")

        elif cscene.bake_type in {'DIFFUSE', 'GLOSSY', 'TRANSMISSION'}:
            col = layout.column(heading="Contributions", align=True)
            col.prop(cbk, "use_pass_direct")
            col.prop(cbk, "use_pass_indirect")
            col.prop(cbk, "use_pass_color")


class CYCLES_RENDER_PT_bake_selected_to_active(CyclesButtonsPanel, Panel):
    bl_label = "Selected to Active"
    bl_context = "render"
    bl_parent_id = "CYCLES_RENDER_PT_bake"
    bl_options = {'DEFAULT_CLOSED'}
    COMPAT_ENGINES = {'CYCLES'}

    @classmethod
    def poll(cls, context):
        scene = context.scene
        rd = scene.render
        return rd.bake.use_multires == False

    def draw_header(self, context):
        scene = context.scene
        cbk = scene.render.bake
        self.layout.prop(cbk, "use_selected_to_active", text="")

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False  # No animation.

        scene = context.scene
        cscene = scene.cycles
        cbk = scene.render.bake
        rd = scene.render

        layout.active = cbk.use_selected_to_active
        col = layout.column()

        col.prop(cbk, "use_cage", text="Cage")
        if cbk.use_cage:
            col.prop(cbk, "cage_object")
            col = layout.column()
            col.prop(cbk, "cage_extrusion")
            col.active = cbk.cage_object is None
        else:
            col.prop(cbk, "cage_extrusion", text="Extrusion")

        col = layout.column()
        col.prop(cbk, "max_ray_distance")


class CYCLES_RENDER_PT_bake_output(CyclesButtonsPanel, Panel):
    bl_label = "Output"
    bl_context = "render"
    bl_parent_id = "CYCLES_RENDER_PT_bake"
    COMPAT_ENGINES = {'CYCLES'}

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False  # No animation.

        scene = context.scene
        cbk = scene.render.bake

        if cbk.use_multires:
            layout.prop(cbk, "use_clear", text="Clear Image")
            if cbk.type in {'DISPLACEMENT', 'VECTOR_DISPLACEMENT'}:
                layout.prop(cbk, "use_lores_mesh")
            if cbk.type == 'VECTOR_DISPLACEMENT':
                layout.prop(cbk, "displacement_space", text="Space")
        else:
            layout.prop(cbk, "target")
            if cbk.target == 'IMAGE_TEXTURES':
                layout.prop(cbk, "use_clear", text="Clear Image")


class CYCLES_RENDER_PT_bake_output_margin(CyclesButtonsPanel, Panel):
    bl_label = "Margin"
    bl_context = "render"
    bl_parent_id = "CYCLES_RENDER_PT_bake_output"
    COMPAT_ENGINES = {'CYCLES'}

    @classmethod
    def poll(cls, context):
        scene = context.scene
        cbk = scene.render.bake
        return cbk.target == 'IMAGE_TEXTURES'

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False  # No animation.

        scene = context.scene
        cscene = scene.cycles
        cbk = scene.render.bake

        if (cscene.bake_type == 'NORMAL' and cbk.normal_space == 'TANGENT') or cscene.bake_type == 'UV':
            if cbk.use_multires or cbk.target == 'IMAGE_TEXTURES':
                layout.prop(cbk, "margin", text="Size")
        else:
            if cbk.use_multires or cbk.target == 'IMAGE_TEXTURES':
                layout.prop(cbk, "margin_type", text="Type")
                layout.prop(cbk, "margin", text="Size")


class CYCLES_RENDER_PT_debug(CyclesDebugButtonsPanel, Panel):
    bl_label = "Debug"
    bl_context = "render"
    bl_options = {'DEFAULT_CLOSED'}
    COMPAT_ENGINES = {'CYCLES'}

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False  # No animation.

        scene = context.scene
        cscene = scene.cycles

        col = layout.column(heading="CPU")

        row = col.row(align=True)
        row.prop(cscene, "debug_use_cpu_sse42", toggle=True)
        row.prop(cscene, "debug_use_cpu_avx2", toggle=True)
        col.prop(cscene, "debug_bvh_layout", text="BVH")

        import platform
        is_macos = platform.system() == 'Darwin'
        col.separator()

        if is_macos:
            col = layout.column(heading="Metal")
            col.prop(cscene, "debug_use_metal_adaptive_compile")
        else:
            col = layout.column(heading="CUDA")
            col.prop(cscene, "debug_use_cuda_adaptive_compile")
            col = layout.column(heading="OptiX")
            col.prop(cscene, "debug_use_optix_debug", text="Module Debug")

            col.separator()

            col = layout.column(heading="HIP")
            col.prop(cscene, "debug_use_hip_adaptive_compile")

        col.separator()

        col.prop(cscene, "debug_bvh_type", text="Viewport BVH")

        col.separator()

        import _cycles
        if _cycles.with_debug:
            col.prop(cscene, "direct_light_sampling_type")


class CYCLES_RENDER_PT_simplify(CyclesButtonsPanel, Panel):
    bl_label = "Simplify"
    bl_context = "render"
    bl_options = {'DEFAULT_CLOSED'}
    COMPAT_ENGINES = {'CYCLES'}

    def draw_header(self, context):
        rd = context.scene.render
        self.layout.prop(rd, "use_simplify", text="")

    def draw(self, context):
        pass


class CYCLES_RENDER_PT_simplify_viewport(CyclesButtonsPanel, Panel):
    bl_label = "Viewport"
    bl_context = "render"
    bl_parent_id = "CYCLES_RENDER_PT_simplify"
    COMPAT_ENGINES = {'CYCLES'}

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False

        scene = context.scene
        rd = scene.render
        cscene = scene.cycles

        layout.active = rd.use_simplify

        col = layout.column()
        col.prop(rd, "simplify_subdivision", text="Max Subdivision")
        col.prop(rd, "simplify_child_particles", text="Child Particles")
        col.prop(rd, "use_simplify_normals", text="Normals")
        col.prop(rd, "simplify_volumes", text="Volume Resolution")

        col.separator()

        col.prop(cscene, "texture_resolution", text="Texture Resolution")
        col.prop(cscene, "texture_limit", text="Texture Size Limit")


class CYCLES_RENDER_PT_simplify_render(CyclesButtonsPanel, Panel):
    bl_label = "Render"
    bl_context = "render"
    bl_parent_id = "CYCLES_RENDER_PT_simplify"
    COMPAT_ENGINES = {'CYCLES'}

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False

        scene = context.scene
        rd = scene.render
        cscene = scene.cycles

        layout.active = rd.use_simplify

        col = layout.column()

        col.prop(rd, "simplify_subdivision_render", text="Max Subdivision")
        col.prop(rd, "simplify_child_particles_render", text="Child Particles")

        col.separator()

        col.prop(cscene, "texture_resolution_render", text="Texture Resolution")
        col.prop(cscene, "texture_limit_render", text="Texture Size Limit")


class CYCLES_RENDER_PT_simplify_culling(CyclesButtonsPanel, Panel):
    bl_label = "Culling"
    bl_context = "render"
    bl_parent_id = "CYCLES_RENDER_PT_simplify"
    bl_options = {'DEFAULT_CLOSED'}
    COMPAT_ENGINES = {'CYCLES'}

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False

        scene = context.scene
        rd = scene.render
        cscene = scene.cycles

        layout.active = rd.use_simplify

        row = layout.row(heading="Camera Culling")
        row.prop(cscene, "use_camera_cull", text="")
        sub = row.column()
        sub.active = cscene.use_camera_cull
        sub.prop(cscene, "camera_cull_margin", text="")

        row = layout.row(heading="Distance Culling")
        row.prop(cscene, "use_distance_cull", text="")
        sub = row.column()
        sub.active = cscene.use_distance_cull
        sub.prop(cscene, "distance_cull_margin", text="")


class CyclesShadingButtonsPanel(CyclesButtonsPanel):
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'HEADER'
    bl_parent_id = 'VIEW3D_PT_shading'

    @classmethod
    def poll(cls, context):
        return (
            CyclesButtonsPanel.poll(context) and
            context.space_data.shading.type == 'RENDERED'
        )


class CYCLES_VIEW3D_PT_shading_render_pass(CyclesShadingButtonsPanel, Panel):
    bl_label = "Render Pass"

    def draw(self, context):
        shading = context.space_data.shading

        layout = self.layout
        if use_vcm(context.scene, context.view_layer):
            layout.label(text="Combined")
        else:
            layout.prop(shading.cycles, "display_pass", text="")


class CYCLES_VIEW3D_PT_lpe_preview_overlay(CyclesShadingButtonsPanel, Panel):
    bl_label = "LPE Preview Overlay"

    @classmethod
    def poll(cls, context):
        return (super().poll(context) and context.view_layer.lpe_node_tree is not None and
                not use_vcm(context.scene, context.view_layer))

    def draw(self, context):
        self.layout.prop(context.space_data.shading.cycles, "lpe_overlay", text="")


class CYCLES_VIEW3D_PT_shading_debug(CyclesDebugButtonsPanel,
                                     CyclesShadingButtonsPanel,
                                     Panel):
    bl_label = "Debug"

    @classmethod
    def poll(cls, context):
        return (
            CyclesDebugButtonsPanel.poll(context) and
            CyclesShadingButtonsPanel.poll(context)
        )

    def draw(self, context):
        shading = context.space_data.shading

        layout = self.layout
        layout.active = context.scene.cycles.use_preview_adaptive_sampling
        layout.prop(shading.cycles, "show_active_pixels")


class CYCLES_VIEW3D_PT_shading_lighting(Panel):
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'HEADER'
    bl_label = "Lighting"
    bl_parent_id = 'VIEW3D_PT_shading'
    COMPAT_ENGINES = {'CYCLES'}

    @classmethod
    def poll(cls, context):
        return (
            context.engine in cls.COMPAT_ENGINES and
            context.space_data.shading.type == 'RENDERED'
        )

    def draw(self, context):
        layout = self.layout
        col = layout.column()
        split = col.split(factor=0.9)

        shading = context.space_data.shading
        col.prop(shading, "use_scene_lights_render")
        col.prop(shading, "use_scene_world_render")

        if not shading.use_scene_world_render:
            col = layout.column()
            split = col.split(factor=0.9)

            col = split.column()
            sub = col.row()
            sub.scale_y = 0.6
            sub.template_icon_view(shading, "studio_light", scale_popup=3)

            col = split.column()
            col.operator("screen.userpref_show", emboss=False, text="", icon='PREFERENCES').section = 'LIGHTS'

            split = layout.split(factor=0.9)
            col = split.column()
            col.prop(shading, "studiolight_rotate_z", text="Rotation")
            col.prop(shading, "studiolight_intensity")
            col.prop(shading, "studiolight_background_alpha")


class CYCLES_VIEW3D_PT_simplify_greasepencil(CyclesButtonsPanel, Panel, GreasePencilSimplifyPanel):
    bl_label = "Grease Pencil"
    bl_parent_id = "CYCLES_RENDER_PT_simplify"
    COMPAT_ENGINES = {'CYCLES'}
    bl_options = {'DEFAULT_CLOSED'}


def draw_device(self, context):
    scene = context.scene
    layout = self.layout
    layout.use_property_split = True
    layout.use_property_decorate = False

    if context.engine == 'CYCLES':
        from . import engine
        cscene = scene.cycles

        col = layout.column()
        col.prop(cscene, "device")

        if not use_vcm(scene, context.view_layer) and engine.with_osl() and (
            use_cpu(context) or (
                use_optix(context) and (
                engine.osl_version()[1] >= 13 or engine.osl_version()[0] > 1))):
            osl_col = layout.column()
            osl_col.prop(cscene, "shading_system")


def get_panels():
    exclude_panels = {
        'DATA_PT_camera_dof',
        'DATA_PT_falloff_curve',
        'DATA_PT_light',
        'DATA_PT_preview',
        'DATA_PT_spot',
        'MATERIAL_PT_context_material',
        'MATERIAL_PT_preview',
        'NODE_DATA_PT_light',
        'NODE_DATA_PT_spot',
        'OBJECT_PT_visibility',
        'VIEWLAYER_PT_filter',
        'VIEWLAYER_PT_layer_passes',
        'RENDER_PT_post_processing',
        'RENDER_PT_simplify',
    }

    panels = []
    for panel in bpy.types.Panel.__subclasses__():
        if hasattr(panel, 'COMPAT_ENGINES') and 'BLENDER_RENDER' in panel.COMPAT_ENGINES:
            if panel.__name__ not in exclude_panels:
                panels.append(panel)

    return panels


classes = (
    CYCLES_PT_sampling_presets,
    CYCLES_PT_viewport_sampling_presets,
    CYCLES_PT_integrator_presets,
    CYCLES_PT_photon_presets,
    CYCLES_PT_performance_presets,
    CYCLES_RENDER_PT_sampling,
    CYCLES_RENDER_PT_sampling_viewport,
    CYCLES_RENDER_PT_sampling_viewport_denoise,
    CYCLES_RENDER_PT_sampling_render,
    CYCLES_RENDER_PT_sampling_render_denoise,
    CYCLES_RENDER_PT_sampling_path_guiding,
    CYCLES_RENDER_PT_sampling_gpu_guiding,
    CYCLES_RENDER_PT_sampling_path_guiding_debug,
    CYCLES_RENDER_PT_sampling_lights,
    CYCLES_RENDER_PT_sampling_advanced,
    CYCLES_RENDER_PT_sampling_advanced_sample_subset,
    CYCLES_RENDER_PT_light_paths,
    CYCLES_RENDER_PT_light_paths_integrator,
    CYCLES_RENDER_PT_light_paths_max_bounces,
    CYCLES_RENDER_PT_light_paths_clamping,
    CYCLES_MT_caustics_method,
    CYCLES_RENDER_PT_light_paths_caustics,
    CYCLES_RENDER_PT_photon_sampling,
    CYCLES_RENDER_PT_photon_footprint,
    CYCLES_RENDER_PT_photon_adaptive_reconstruction,
    CYCLES_RENDER_PT_photon_guiding,
    CYCLES_RENDER_PT_light_paths_fast_gi,
    CYCLES_RENDER_PT_volumes,
    CYCLES_RENDER_PT_subdivision,
    CYCLES_RENDER_PT_curves,
    CYCLES_RENDER_PT_curves_viewport_display,
    CYCLES_RENDER_PT_simplify,
    CYCLES_RENDER_PT_simplify_viewport,
    CYCLES_RENDER_PT_simplify_render,
    CYCLES_RENDER_PT_simplify_culling,
    CYCLES_VIEW3D_PT_simplify_greasepencil,
    CYCLES_VIEW3D_PT_shading_lighting,
    CYCLES_VIEW3D_PT_shading_render_pass,
    CYCLES_VIEW3D_PT_lpe_preview_overlay,
    CYCLES_VIEW3D_PT_shading_debug,
    CYCLES_RENDER_PT_motion_blur,
    CYCLES_RENDER_PT_motion_blur_curve,
    CYCLES_RENDER_PT_film,
    CYCLES_RENDER_PT_film_pixel_filter,
    CYCLES_RENDER_PT_film_transparency,
    CYCLES_RENDER_PT_performance,
    CYCLES_RENDER_PT_performance_compositor,
    CYCLES_RENDER_PT_performance_compositor_denoise_settings,
    CYCLES_RENDER_PT_performance_threads,
    CYCLES_RENDER_PT_performance_memory,
    CYCLES_RENDER_PT_performance_texture_cache,
    CYCLES_RENDER_PT_performance_acceleration_structure,
    CYCLES_RENDER_PT_performance_final_render,
    CYCLES_RENDER_PT_performance_viewport,
    CYCLES_RENDER_PT_passes,
    CYCLES_RENDER_PT_passes_data,
    CYCLES_RENDER_PT_passes_light,
    CYCLES_RENDER_PT_passes_crypto,
    CYCLES_RENDER_PT_passes_aov,
    CYCLES_RENDER_PT_passes_lightgroups,
    CYCLES_RENDER_PT_passes_lightgroups_light_pass_aovs,
    CYCLES_RENDER_PT_filter,
    CYCLES_PT_post_processing,
    CYCLES_CAMERA_PT_dof,
    CYCLES_CAMERA_PT_dof_aperture,
    CYCLES_CAMERA_PT_lens_custom_parameters,
    CYCLES_PT_context_material,
    CYCLES_OBJECT_PT_motion_blur,
    CYCLES_OBJECT_PT_shading_gi_approximation,
    CYCLES_OBJECT_PT_shading_caustics,
    CYCLES_OBJECT_PT_nested_dielectric,
    CYCLES_OBJECT_PT_lightgroup,
    CYCLES_OBJECT_PT_visibility,
    CYCLES_OBJECT_PT_visibility_ray_visibility,
    CYCLES_OBJECT_PT_visibility_culling,
    CYCLES_LIGHT_PT_preview,
    CYCLES_LIGHT_PT_light,
    CYCLES_LIGHT_PT_settings,
    CYCLES_LIGHT_PT_nodes,
    CYCLES_LIGHT_PT_beam_shape,
    CYCLES_WORLD_PT_preview,
    CYCLES_WORLD_PT_surface,
    CYCLES_WORLD_PT_volume,
    CYCLES_WORLD_PT_mist,
    CYCLES_WORLD_PT_ray_visibility,
    CYCLES_WORLD_PT_settings,
    CYCLES_WORLD_PT_settings_surface,
    CYCLES_WORLD_PT_photon_source,
    CYCLES_WORLD_PT_settings_volume,
    CYCLES_WORLD_PT_settings_light_group,
    CYCLES_MATERIAL_PT_preview,
    CYCLES_MATERIAL_PT_surface,
    CYCLES_MATERIAL_PT_volume,
    CYCLES_MATERIAL_PT_displacement,
    CYCLES_MATERIAL_PT_settings,
    CYCLES_MATERIAL_PT_settings_surface,
    CYCLES_MATERIAL_PT_settings_caustics,
    CYCLES_MATERIAL_PT_settings_nested_dielectric,
    CYCLES_MATERIAL_PT_settings_volume,
    CYCLES_RENDER_PT_bake,
    CYCLES_RENDER_PT_bake_influence,
    CYCLES_RENDER_PT_bake_selected_to_active,
    CYCLES_RENDER_PT_bake_output,
    CYCLES_RENDER_PT_bake_output_margin,
    CYCLES_RENDER_PT_debug,
    node_panel(CYCLES_MATERIAL_PT_settings),
    node_panel(CYCLES_MATERIAL_PT_settings_surface),
    node_panel(CYCLES_MATERIAL_PT_settings_caustics),
    node_panel(CYCLES_MATERIAL_PT_settings_nested_dielectric),
    node_panel(CYCLES_MATERIAL_PT_settings_volume),
    node_panel(CYCLES_WORLD_PT_ray_visibility),
    node_panel(CYCLES_WORLD_PT_settings),
    node_panel(CYCLES_WORLD_PT_settings_surface),
    node_panel(CYCLES_WORLD_PT_settings_volume),
    node_panel(CYCLES_LIGHT_PT_light),
    node_panel(CYCLES_LIGHT_PT_settings),
    node_panel(CYCLES_LIGHT_PT_beam_shape)
)


def register():
    from bpy.utils import register_class

    bpy.types.RENDER_PT_context.append(draw_device)

    for panel in get_panels():
        panel.COMPAT_ENGINES.add('CYCLES')

    for cls in classes:
        register_class(cls)


def unregister():
    from bpy.utils import unregister_class

    bpy.types.RENDER_PT_context.remove(draw_device)

    for panel in get_panels():
        if 'CYCLES' in panel.COMPAT_ENGINES:
            panel.COMPAT_ENGINES.remove('CYCLES')

    for cls in classes:
        unregister_class(cls)
