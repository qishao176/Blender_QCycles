# SPDX-FileCopyrightText: 2026 Blender Authors
#
# SPDX-License-Identifier: GPL-2.0-or-later

bl_info = {
    "name": "QCycles Production Guide",
    "author": "QCycles",
    "version": (1, 3, 0),
    "blender": (5, 3, 0),
    "location": "3D Viewport > Sidebar > QCycles",
    "description": "Settings and guidance for QCycles caustics, light paths, materials, Deep EXR and motion blur",
    "category": "Render",
}

import bpy
from bpy.props import BoolProperty
from bpy.app.translations import pgettext_iface as iface_

_GEOMETRY = {'MESH', 'CURVE', 'SURFACE', 'META', 'FONT', 'CURVES', 'POINTCLOUD', 'VOLUME'}


def _photon_get(scene):
    return scene.cycles.caustics_method == 'PHOTON'


def _photon_set(scene, enabled):
    scene.cycles.caustics_method = 'PHOTON' if enabled else 'MNEE'


def _active_object(context):
    ob = context.view_layer.objects.active
    return ob if ob and ob.select_get(view_layer=context.view_layer) else None


def _section(layout, key, title, *, closed=True, text_ctxt="*"):
    header, body = layout.panel(key, default_closed=closed)
    header.label(text=title, text_ctxt=text_ctxt)
    return body


def _help(layout, key, title, *lines, text_ctxt="*"):
    body = _section(layout, key, title, text_ctxt=text_ctxt)
    if body:
        for line in lines:
            body.label_multiline(text=line)


def _property_help(layout, settings, topics):
    for identifier, *lines in topics:
        prop = settings.bl_rna.properties[identifier]
        _help(layout, identifier, prop.name, *lines, text_ctxt=prop.translation_context)


class GuidePanel:
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "QCycles"
    bl_options = {'DEFAULT_CLOSED'}

    @classmethod
    def poll(cls, context):
        return context.scene and hasattr(context.scene, "cycles") and context.engine == 'CYCLES'


class GZGUIDE_PT_photons(GuidePanel, bpy.types.Panel):
    bl_label = "Photon Caustics"
    bl_options = set()
    bl_order = 0

    @classmethod
    def poll(cls, context):
        return bool(context.scene and hasattr(context.scene, "cycles"))

    def draw(self, context):
        from cycles import ui
        layout = self.layout
        scene = context.scene
        if context.engine != 'CYCLES':
            layout.label(text="Switch the render engine to Cycles to use this guide.")
            layout.prop(scene.render, "engine")
            return
        if ui.use_vcm(scene, context.view_layer):
            layout.label(text="Photon caustics use the Path Tracing integrator.", icon='INFO')
            return
        enabled = _photon_get(scene)
        unavailable = ui.photon_caustics_unavailable_reason(context)
        row = layout.row()
        row.enabled = enabled or unavailable is None
        row.prop(scene, "gz_guide_use_photons", text="Enable Photon Caustics")
        if unavailable:
            layout.label(text=unavailable, text_ctxt=ui.photon_caustics_context, icon='INFO')
        if not enabled:
            return
        row = layout.row(align=True)
        row.prop(scene.cycles, "caustics_reflective", text="Reflection")
        row.prop(scene.cycles, "caustics_refractive", text="Refraction")
        layout.label(text="Enable shadow caustics on the light and the casting object.")
        layout.label(text="On the material, enable reflection, refraction, or both.")
        _help(layout, "participation", "Which Paths Can Cast Caustics?",
              "Enable reflection for a metallic bounce and refraction for a path through glass. The scene, the caster and its material must permit the event. Enable casting on each parent instancer as well. Diffuse surfaces can receive photon caustics without a receiver flag.")


class PhotonPanel(GuidePanel):
    bl_parent_id = "GZGUIDE_PT_photons"

    @classmethod
    def poll(cls, context):
        from cycles.engine import use_vcm
        return (super().poll(context) and _photon_get(context.scene) and
                not use_vcm(context.scene, context.view_layer))


def _draw_light(layout, ob, scene):
    light = ob.data
    layout.label(text=ob.name, translate=False, icon='LIGHT_DATA')
    layout.prop(light.cycles, "is_caustics_light", text="Cast Shadow Caustics")
    photon_mode = _photon_get(scene)
    col = layout.column(align=True)
    col.active = (photon_mode and light.cycles.is_caustics_light and ob.cycles.visible_photon and
                  not (light.type == 'AREA' and light.cycles.is_portal))
    col.prop(light.cycles, "photon_radius_scale", text="Photon Size Multiplier")
    col.prop(light.cycles, "photon_strength", text="Photon Strength Multiplier")
    if photon_mode:
        layout.prop(ob.cycles, "visible_photon", text="Photon Ray Visibility")
    else:
        layout.label_multiline(text="Enable Photon Caustics to adjust these multipliers.")
    _help(layout, "light_factors", "Adjusting This Light",
          "Both multipliers default to 1. Size scales this light's surface and volume footprints after reconstruction. A wider footprint distributes the same integrated energy over a larger area or volume.",
          "Strength scales this light's photon contribution. Increase it to brighten the caustic while keeping its width.",
          "The light data controls casting and the two multipliers for all objects sharing it. Photon Ray Visibility belongs to the selected light object. For a photon-only source, leave photon visibility enabled while adjusting ordinary ray visibility independently.")


_WORLD_TOPICS = (
    ("sampling_method",
     "World sampling must be enabled to detect the bright HDR region. Choose Auto or Manual and inspect the source in the rendered viewport before raising photon counts."),
    ("is_caustics_light",
     "Allows the detected World source to cast caustics. Enable a caster object and the corresponding reflective or refractive material as well."),
    ("visible_photon",
     "Controls whether photons may leave the detected World source, independently of camera, diffuse and glossy ray visibility."),
    ("photon_sun_threshold",
     "Minimum ratio of peak brightness to the World average. Lower it if a sun in a hazy HDRI is missed. Raising it rejects weaker bright regions. Check the detected cone after changing the threshold."),
    ("photon_sun_radius_scale",
     "Scales the angular radius of the detected source. Start at 1 and compare its outline with the bright disk in the HDRI. Widening the cone includes surrounding directions evaluated from the original World shader."),
    ("show_photon_sun",
     "Draws the detected cone as a red outline in the rendered viewport with overlays enabled. Use it to check direction and angular size. It does not appear in the final render."),
)


class GZGUIDE_PT_photon_sources(PhotonPanel, bpy.types.Panel):
    bl_label = "Lights and HDR Worlds"

    def draw(self, context):
        layout = self.layout
        ob = _active_object(context)
        if ob and ob.type == 'LIGHT':
            _draw_light(layout, ob, context.scene)
        else:
            layout.label(text="Select a light to adjust its photon strength and size.")
        world = context.view_layer.world_override or context.scene.world
        if world:
            layout.label(text=world.name, translate=False, icon='WORLD')
            cw = world.cycles
            layout.prop(cw, "sampling_method")
            row = layout.row()
            row.active = cw.sampling_method != 'NONE'
            row.prop(cw, "is_caustics_light")
            layout.prop(cw, "visible_photon")
            col = layout.column()
            col.active = cw.sampling_method != 'NONE' and cw.visible_photon and cw.is_caustics_light
            col.prop(cw, "photon_sun_threshold")
            col.prop(cw, "photon_sun_radius_scale")
            col.prop(cw, "show_photon_sun")
            body = _section(layout, "world_help", "Photon Source Settings")
            if body:
                _property_help(body, cw, _WORLD_TOPICS)


class PhotonHelpPanel(PhotonPanel):
    """Keep the guide readable even when a native setting is disabled or hidden."""
    bl_label = "Parameter Guide"

    def draw(self, context):
        _property_help(self.layout, context.scene.cycles, self.topics)


class GZGUIDE_PT_sampling_help(PhotonHelpPanel, bpy.types.Panel):
    bl_parent_id = "GZGUIDE_PT_photon_sampling"
    topics = (
        ("photon_caustics_count",
         "Sets the number of light paths attempted per emission round. Deposits depend on hitting a caster and then a receiver. Raise this first when individual rounds leave large gaps. More paths require more tracing, map memory and gathering."),
        ("photon_caustics_update_interval",
         "Camera samples between new maps. At 1 each sample gets a fresh emission round. At 4 up to four samples share a map. A longer interval amortizes emission, but a static pattern takes more samples to average away."),
        ("use_photon_ser",
         "On compatible OptiX devices, Shader Execution Reordering groups photon shading work on the GPU. Try it with many different caster materials and compare equal-budget render times. It does not change photon count or radius."),
        ("use_photon_query_index",
         "Groups stored photons by light and by surface or volume so a lookup can skip unrelated records. Keep it enabled for scenes with several lights or fog. The original photon identities are retained for light groups and LPEs."),
        ("photon_caustics_separate_sampling",
         "Gives ordinary lighting and photon caustics their own sample accounting before combining their estimates. Use this to keep a clean direct-light region from dictating the photon budget. Deep output uses combined sampling."),
        ("photon_caustics_sampling_budget",
         "Shared Maximum Samples uses the scene limit for both estimators. Independent Maximum Samples keeps the photon limit separate. Use the latter to compare reconstruction settings while holding ordinary lighting fixed."),
        ("photon_caustics_samples",
         "Caps camera samples spent evaluating photon caustics. With 256 samples and an emission interval of 4, a full run normally uses 64 maps. To add more paths to each map, change Photons per Round."),
    )


class GZGUIDE_PT_schedule_help(PhotonHelpPanel, bpy.types.Panel):
    bl_parent_id = "GZGUIDE_PT_photon_schedule"
    topics = (
        ("use_photon_cache",
         "In a separate photon pass, accumulate the full image in a compact buffer shared by render tiles. This avoids recomputing the same photon image for each tile. The buffer grows with the resolution and enabled passes."),
        ("photon_cache_order",
         "Before Sampling shows caustics first. After Sampling shows ordinary lighting first. Simultaneous interleaves both estimators and is available with tiling off. The order affects when each part appears in the render window."),
        ("deep_photon_samples",
         "In Simple Deep Color output, sets the depth resolution used for volume photon lighting. More layers help preserve separation along the camera ray and use more memory. Compare after a Deep merge at the depth of the beam."),
    )


class GZGUIDE_PT_reconstruction_help(PhotonHelpPanel, bpy.types.Panel):
    bl_parent_id = "GZGUIDE_PT_photon_adaptive_reconstruction"
    topics = (
        ("use_photon_adaptive_radius",
         "After each emission round, reconstruction adjusts each footprint using nearby coverage. Widening footprints can fill gaps between sparse deposits and lowers their peaks. Regions beyond the reach of any deposit need more successful light paths.",
         "Compare grain in a dim patch with the narrowest bright line in the same crop. Surfaces offer four methods. Volume photons use local scalar reconstruction."),
        ("photon_reconstruction_method",
         "Local Scalar scales the original footprint uniformly using nearby photon density. Start here to identify areas with sparse coverage.",
         "Regional Coverage estimates the effective count using the relative contributions of nearby photons. A region with many deposits can still need wider footprints if a few deposits carry most of the energy.",
         "Regional Anisotropic uses gradient protection to expand by different amounts in two directions. Try it on thin bands that need smoothing along their length.",
         "Adaptive Ellipse Multiplier estimates overlap between the original footprints and chooses a scale for each photon. It preserves ellipse orientation and aspect ratio. Try it when the differential already outlines the sharp caustic."),
        ("photon_adaptive_neighbors",
         "Sets the target local coverage. Raising it expands more sparse footprints and can fill dim gaps. Lower it if thin lines start to thicken.",
         "This target controls kernel size. Photons per Round controls how many paths leave the lights."),
        ("photon_adaptive_max_scale",
         "A limit of 1 skips adaptive reconstruction. Local Scalar and Adaptive Ellipse Multiplier scale each original footprint. The regional methods bound extra spread relative to Gather Radius.",
         "Raise the limit for broken coverage in sparse areas. Lower it if adjacent bright lines begin to merge."),
        ("photon_reconstruction_pilot_scale",
         "Sets how far to search for neighborhood statistics. Larger searches can visit more photons each round. Maximum Radius Multiplier still bounds the final expansion.",
         "In Local Scalar, 1 keeps statistics local to the footprint. To sample a wider neighborhood while preserving fine lines, hold the maximum multiplier at 2 and compare statistical multipliers of 2 and 4."),
        ("use_photon_cell_statistics",
         "Available for Adaptive Ellipse Multiplier above a statistical multiplier of 1. Cell density narrows the search in dense regions. The search expands again when it finds too few compatible photons."),
        ("use_photon_neighbor_selection",
         "Adaptive Ellipse Multiplier can sample neighboring photons to estimate overlap, weighting by selection probability. Dense regions need fewer detailed evaluations, with some variation in estimated radius.",
         "Selection only affects reconstruction statistics. Each photon keeps its original path for caustics, LPEs and light groups."),
        ("photon_neighbor_samples",
         "With compatible neighbor selection enabled, this controls how many overlap evaluations each photon receives. Higher values usually stabilize the estimate. Small neighborhoods are evaluated in full."),
        ("use_photon_beam_continuity",
         "Compares the previous reflection or refraction position, normal variation and event type. This helps keep statistics separate where two beams reach the same receiver.",
         "Useful for curved metal and glass with multiple interfaces. It uses extra memory to store data about the previous interface."),
        ("photon_reconstruction_filter",
         "Same Light and Receiver groups photons by source and receiving object. Last Interface Normal also compares caster orientation, helping separate beams arriving from opposite sides of a curved surface.",
         "Full Caster Path requires the same sequence of objects, materials and scattering events. Caster Path and Normal checks both the path and the orientation.",
         "Stricter matching separates crossing beams, but leaves fewer eligible neighbors in sparse regions."),
        ("photon_reconstruction_normal_angle",
         "Sets the angular tolerance for normal matching at the previous interface. Small angles separate sharp facets. Larger angles can include neighbors from a continuous curved surface.",
         "Available with normal filtering or previous-interface continuity."),
        ("use_photon_gradient_guard",
         "Uses reliable changes in local brightness, color or direction to limit spreading at edges and intersections. It accounts for sampling variation when identifying edges.",
         "Regional Coverage expands according to effective coverage. Choose Regional Anisotropic for protection along a gradient direction."),
        ("photon_gradient_strength",
         "At 0, coverage alone controls smoothing. Increase the strength if bright edges spread. If grain persists beside an edge, compare a lower setting."),
    )


class GZGUIDE_PT_footprint_help(PhotonHelpPanel, bpy.types.Panel):
    bl_parent_id = "GZGUIDE_PT_photon_footprint"
    topics = (
        ("photon_caustics_footprint",
         "Fixed Radius starts with a circular footprint. Differential Ellipse carries beam width and orientation through specular paths, preserving the direction of thin caustics from lenses or curved surfaces."),
        ("photon_caustics_radius",
         "Sets the base gathering radius in scene units. Smaller values can separate fine lines but expose gaps. Increase emission or enable adaptive radius if coverage becomes patchy."),
        ("photon_caustics_min_radius",
         "Sets the minimum differential semi-axis before adaptive reconstruction. At 0, it uses 5% of Gather Radius. Raise the floor to reduce tiny bright spikes, checking that narrow lines keep their shape."),
        ("photon_caustics_roughness_spread",
         "Rough reflection and transmission already scatter directions. This setting adds footprint spread at the landing point for finishes such as lightly frosted glass or metal. Set it to 0 to disable the extra widening."),
        ("photon_caustics_gather_max",
         "Limits the photons evaluated across all lights at each surface shading point. At 0, every candidate is evaluated. A cap can speed up dense gathers at the cost of some sampling noise."),
        ("photon_caustics_scatter",
         "Jitters the lookup within the tangent plane to disperse fixed-radius surface spots. The unit is the footprint radius. Set to 0 to disable."),
        ("photon_caustics_differential_scatter",
         "Controls scatter independently in differential mode, including circular fallback footprints. Lower values keep sharply outlined ellipses cleaner."),
        ("photon_caustics_noise_strength",
         "Adds blue-noise grain to fixed-radius surface and volume footprints. At 0 the kernel is smooth. Higher values show more grain at low sample counts and average out over many samples."),
        ("photon_caustics_ellipse_noise_strength",
         "Controls grain inside differential surface footprints, including circular fallbacks. Set both this and ellipse scatter to 0 to inspect the original footprint."),
    )


class GZGUIDE_PT_guiding_help(PhotonHelpPanel, bpy.types.Panel):
    bl_parent_id = "GZGUIDE_PT_photon_guiding"
    topics = (
        ("use_photon_guiding",
         "Emission starts with the ordinary sampler. When a camera path gathers caustic energy, it records where that photon first hit a caster. Each light fits eight 3D Gaussian components to these positions.",
         "The model is updated every 32 emission rounds when there is enough feedback. A trained light guides half its emission samples and continues exploring with the rest. Photon weights use the combined probability.",
         "Area lights and zero-radius point or spot lights can guide direction. Sun lights guide the origin in a plane across the beam and retain the sun angle. Finite-radius point and spot lights use ordinary emission.",
         "Caustics seen through mirrors or glass can train the guide. Contributions after a diffuse camera bounce still illuminate the image but are excluded from training. Allow emission rounds after the first update when comparing settings."),
        ("use_photon_area_guiding",
         "A patterned area light can learn where to start photons on its surface. The sampler measures textured emission by region, then incorporates visible caustic contributions and retains uniform exploration.",
         "Use this alongside 3D directional guiding. At zero Spread the direction is fixed, so selecting better emission positions is especially useful."),
    )


class GZGUIDE_PT_photon_schedule(PhotonPanel, bpy.types.Panel):
    bl_label = "Photon Cache and Calculation Order"

    def draw(self, context):
        from cycles.ui import draw_photon_cache_settings, photon_caustics_uses_deep_output
        scene = context.scene
        draw_photon_cache_settings(self.layout, scene)
        if scene.cycles.deep_simple_mode and photon_caustics_uses_deep_output(scene):
            self.layout.prop(scene.cycles, "deep_photon_samples")


class GZGUIDE_PT_lpe(GuidePanel, bpy.types.Panel):
    bl_label = "LPE Light Paths"
    bl_order = 1

    def draw(self, context):
        layout = self.layout
        _help(layout, "editor", "Create a Light Path",
              "Switch an area to the LPE Node Editor.", "Create a node tree and connect a light to an output.")
        _help(layout, "viewer", "Preview a Light Path",
              "Ctrl-Shift-click a node to connect the viewer.", "The rendered viewport shows the contribution from the selected socket.")
        _help(layout, "edit", "Edit Lights Live",
              "Choose Add > Output > Light Edit.", "Connect the light through the adjustment node. Click Apply to write the changes to the light.")
        _help(layout, "cache", "Cache a Pass",
              "Set the output to Always to switch to its cached preview during interactive rendering.")
        _help(layout, "query", "Inspect a Region",
              "Press N in the LPE or Image Editor and open Contribution Query.",
              "Select a region and start sampling. Add a result as an object path or a material path.",
              "Start with the combined region thumbnail, then inspect the highest contributions. Total Light groups events along the same object or material sequence. Detailed Paths separates reflection, transmission and other events.")


def _is_procedural(ob):
    return (ob.instance_type in {'VERTS', 'FACES'} or bool(ob.particle_systems) or
            any(m.type == 'NODES' for m in ob.modifiers))


def _instance_members(root, context):
    # all_objects follows child collections. Collection instances need a separate descent.
    seen_objects = set()
    seen_collections = set()
    procedural = set()
    pending = [root]
    while pending:
        ob = pending.pop()
        if ob in seen_objects:
            continue
        seen_objects.add(ob)
        yield ob
        collection = ob.instance_collection if ob.instance_type == 'COLLECTION' else None
        if collection and collection not in seen_collections:
            seen_collections.add(collection)
            pending.extend(collection.all_objects)
        if _is_procedural(ob):
            procedural.add(ob)
    if procedural:
        generating_roots = procedural | {root}
        for inst in context.evaluated_depsgraph_get().object_instances:
            if inst.is_instance and inst.parent and inst.parent.original in generating_roots:
                source = inst.object.original
                if source not in seen_objects:
                    seen_objects.add(source)
                    yield source


def instance_issues(root, context):
    scene = context.scene.cycles
    material_override = context.view_layer.material_override
    found_geometry = False
    for ob in _instance_members(root, context):
        is_instancer = ob == root or ob.instance_type == 'COLLECTION' or _is_procedural(ob)
        if not is_instancer and ob.type not in _GEOMETRY:
            continue
        if not ob.cycles.is_caustics_caster:
            yield (ob.name, "Cast Shadow Caustics is disabled")
        if not ob.cycles.use_nested_dielectric:
            yield (ob.name, "Nested Dielectrics is disabled")
        if ob.type not in _GEOMETRY:
            continue
        found_geometry = True
        materials = [material_override] if material_override else [s.material for s in ob.material_slots]
        for mat in dict.fromkeys(materials or [None]):
            if mat is None:
                yield (ob.name, "Missing material")
                continue
            label = f"{ob.name} / {mat.name}"
            cm = mat.cycles
            if not cm.caustics_reflective and not cm.caustics_refractive:
                yield (label, "Reflective and refractive caustics are both disabled")
            elif not ((cm.caustics_reflective and scene.caustics_reflective) or
                      (cm.caustics_refractive and scene.caustics_refractive)):
                yield (label, "The matching caustic type is disabled in the scene")
            elif scene.caustics_reflective and not cm.caustics_reflective:
                yield (label, "Refraction only, reflective caustics disabled")
            elif scene.caustics_refractive and not cm.caustics_refractive:
                yield (label, "Reflection only, refractive caustics disabled")
            if cm.nested_dielectric_priority == 0:
                yield (label, "Medium priority is zero")
    if not found_geometry:
        yield (root.name, "No geometry to inspect")


class GZGUIDE_PT_materials(GuidePanel, bpy.types.Panel):
    bl_label = "Objects, Lights and Materials"
    bl_order = 2

    def draw(self, context):
        layout = self.layout
        ob = _active_object(context)
        if not ob:
            layout.label(text="Select an object.", icon='INFO')
            return
        if ob.type == 'LIGHT':
            _draw_light(layout, ob, context.scene)
            return
        instance = ob.instance_type == 'COLLECTION' or _is_procedural(ob)
        if not instance and ob.type not in _GEOMETRY:
            layout.label(text="Select a geometry object or an instance.", icon='INFO')
            return
        layout.label(text=ob.name, translate=False, icon='OBJECT_DATA')
        layout.prop(ob.cycles, "is_caustics_caster")
        if _photon_get(context.scene):
            col = layout.column(align=True)
            col.active = ob.cycles.is_caustics_caster
            col.prop(ob.cycles, "photon_caustics_geometry_normal")
            col.prop(ob.cycles, "photon_caustics_zero_roughness")
        else:
            layout.prop(ob.cycles, "is_caustics_receiver")
        layout.prop(ob.cycles, "use_nested_dielectric")
        if ob.type in _GEOMETRY:
            layout.separator()
            if ob.material_slots:
                layout.template_list("MATERIAL_UL_matslots", "gz_guide", ob, "material_slots",
                                     ob, "active_material_index", rows=min(5, len(ob.material_slots)))
                layout.template_ID(ob, "active_material")
            mat = ob.active_material
            if mat:
                cm = mat.cycles
                row = layout.row(align=True)
                row.prop(cm, "caustics_reflective", text="Reflective Caustics")
                row.prop(cm, "caustics_refractive", text="Refractive Caustics")
                if _photon_get(context.scene):
                    layout.prop(cm, "photon_caustics_geometry_normal")
                    layout.prop(cm, "photon_caustics_zero_roughness")
                    layout.prop(cm, "photon_caustics_precise_spectrum")
                    layout.label_multiline(text="Most scenes can leave this off. Enable it when you need more accurate photon dispersion, such as saturated light passing through glass. It can slow photon rendering.")
                layout.prop(cm, "nested_dielectric_priority")
                row = layout.row()
                row.active = cm.nested_dielectric_priority > 0
                row.prop(cm, "use_nested_dielectric_union")
                layout.prop(cm, "use_disney_glossy_terminator", text="Disney Glossy Terminator")
                _help(layout, "disney_terminator", "When to Use Disney Terminator Softening",
                      "Light flaked car paint at a grazing angle. If an entire row of flakes disappears at the terminator, enable softening and compare the same angle. The flakes should fade toward the dark side.",
                      "It attenuates rough reflection according to how the smooth surface and flake normals face the light. The flake directions and roughness are preserved while highlights lose energy near the terminator.",
                      "Applies to rough GGX and Beckmann reflection, including the reflective lobe of rough glass. Ideal mirrors and transmission bypass the correction.",
                      "Both camera and photon paths honor this material setting. Photon normal or zero-roughness overrides can hide the effect on flake reflections.")
            else:
                layout.label(text="The active object or material slot has no material.", icon='INFO')
        _help(layout, "overrides", "Object and Material Overrides",
              "For photons, either the object or the material can force zero roughness or mesh normals. Clear the override on both to restore material shading. Mesh normals retain smooth shading while bypassing material normal maps and bump.",
              "Enable nested dielectrics on the object and each enclosing instance, then set a nonzero priority on each participating material. Higher priorities take precedence in overlaps, for example glass 3, ice 2 and water 1.",
              "Enable the union option for overlapping regions of the same participating material when they should behave as one medium. Leave it off to keep the internal boundaries.")
        if instance:
            # Scan only when the user opens diagnostics, including evaluated procedural instances.
            body = _section(layout, "instance", "Check Instance Participation", closed=False)
            if body:
                found = False
                for name, reason in instance_issues(ob, context):
                    body.label(text=f"{name} · {iface_(reason)}", translate=False, icon='INFO')
                    found = True
                if not found:
                    body.label(text="All instance participation switches are enabled.", icon='CHECKMARK')


class GZGUIDE_PT_deep(GuidePanel, bpy.types.Panel):
    bl_label = "Deep EXR"
    bl_order = 3

    def draw(self, context):
        layout = self.layout
        _help(layout, "output", "Write Deep EXR",
              "Add a File Output node in the Compositor and choose Deep EXR.",
              "Connect lighting passes directly from the same render layer, then render.")
        _help(layout, "contribution", "Contribution Deep",
              "Stores depth samples with a contribution to this pass. Files are usually smaller.",
              "Use it to isolate one light, a light group or an LPE.")
        _help(layout, "occlusion", "Occlusion Deep",
              "Retains occlusion from unlit fog and surfaces.", "Use Combined with Occlusion Deep when inserting an object at depth in compositing.")
        _help(layout, "passes", "Supported Passes",
              "Recorded lighting, caustics, light group and LPE passes are supported.",
              "Depth, normals, motion vectors and Cryptomatte are unavailable as Deep lighting outputs.")


class GZGUIDE_PT_motion(GuidePanel, bpy.types.Panel):
    bl_label = "Motion Blur"
    bl_order = 4

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        layout.prop(scene.render, "use_motion_blur", text="Enable Motion Blur")
        col = layout.column()
        col.active = scene.render.use_motion_blur
        col.prop(context.view_layer, "use_motion_blur", text="Current View Layer")
        settings = col.column()
        settings.active = context.view_layer.use_motion_blur
        settings.prop(scene.render, "motion_blur_position")
        settings.prop(scene.render, "motion_blur_shutter")
        settings.prop(scene.cycles, "use_relative_motion_blur")
        settings.prop(scene.cycles, "use_photon_motion_blur_correction")
        _help(layout, "relative", "Choosing a Motion Correction",
              "Enable relative motion correction for ordinary lighting when a headlight moves with the car.",
              "Photon emission and gathering have a separate correction switch.")
        _help(layout, "shutter", "Shutter Curve",
              "In Render Properties > Motion Blur > Shutter Curve, shape the exposure over time.",
              "Shutter length is measured in frames. Position places the exposure before, around or after the frame time. For a camera and a light attached to a vehicle, compare the lit edge against the body at the same shutter length.")


class GZGUIDE_PT_gpu_surface_guiding(GuidePanel, bpy.types.Panel):
    bl_label = "GPU Surface Path Guiding"
    bl_translation_context = "CyclesPhotonCaustics"
    bl_order = 5
    topics = (
        ("use_gpu_guiding",
         "At diffuse and sufficiently rough reflective surfaces, the GPU learns which bounce directions bring light back to the camera. Try a room lit through a doorway. Glass transmission, ideal mirrors, hair and volumes keep their usual sampling."),
        ("gpu_guiding_training_samples",
         "The first camera samples update the guide and also contribute to the image. With 1,024 total samples and 128 training samples, the remaining 896 use the frozen guide. Photon emission trains its own 3D Gaussian guide separately."),
        ("gpu_guiding_probability",
         "At 0.5, an eligible surface with a trained cell samples about half its directions from the guide and half from its material. Compare the same crop and sample budget. Reduce the probability if grain increases. Turn off the checkbox to disable guiding."),
    )

    def draw(self, context):
        from cycles import ui

        layout = self.layout
        settings = context.scene.cycles
        controls = layout.column()
        controls.enabled = (ui.CYCLES_RENDER_PT_sampling_gpu_guiding.poll(context) and
                            not ui.use_vcm(context.scene, context.view_layer))
        controls.prop(settings, "use_gpu_guiding")
        training = controls.column()
        training.active = settings.use_gpu_guiding
        training.prop(settings, "gpu_guiding_training_samples", text="Training Samples",
                      text_ctxt=ui.photon_caustics_context)
        training.prop(settings, "gpu_guiding_probability")
        if not controls.enabled:
            layout.label_multiline(text="Use a CUDA or OptiX render device and the Path Tracing integrator to enable GPU surface guiding.")
        _property_help(layout, settings, self.topics)


class GZGUIDE_PT_overscan(GuidePanel, bpy.types.Panel):
    bl_label = "EXR Overscan"
    bl_order = 6

    def draw(self, context):
        from bl_ui.properties_output import RENDER_PT_exr_overscan

        RENDER_PT_exr_overscan.draw(self, context)
        _help(self.layout, "overscan_padding", "Percentage or Per-Side Pixels",
              "Percentage adds the same number of pixels to every edge, using the longer side of the render as the reference and rounding up. At 1920 × 1080, 5% adds 96 pixels on each edge, giving 2112 × 1272 pixels of data.",
              "Pixels sets Left, Right, Bottom and Top independently at render resolution. For a 64-pixel margin on the left only, use 64, 0, 0, 0. Set all amounts in the active mode to zero to turn overscan off.")
        _help(self.layout, "overscan_windows", "Camera Preview and EXR Windows",
              "In camera view the outer guide shows the extra area. Interactive rendering keeps the camera frame. In the final EXR, displayWindow is the delivery frame and dataWindow includes the extra pixels for reframing or lens distortion.",
              "Use a perspective or orthographic camera, disable Render Region and use single-view output. Choose OpenEXR, Multilayer EXR or Deep EXR in Output Properties or on a connected compositor File Output. A PNG-only render uses the delivery frame.")


_CLASSES = (GZGUIDE_PT_photons, GZGUIDE_PT_lpe, GZGUIDE_PT_materials,
            GZGUIDE_PT_deep, GZGUIDE_PT_motion, GZGUIDE_PT_gpu_surface_guiding, GZGUIDE_PT_overscan)
_registered = []


def register():
    from cycles import ui

    bpy.types.Scene.gz_guide_use_photons = BoolProperty(
        name="Enable Photon Caustics", description="Enable photon caustics or switch back to MNEE",
        get=_photon_get, set=_photon_set, options={'SKIP_SAVE'})
    # Use exactly the native renderer controls, including presets and availability checks.
    classes = list(_CLASSES)
    for native in (ui.CYCLES_RENDER_PT_photon_sampling, ui.CYCLES_RENDER_PT_photon_footprint,
                   ui.CYCLES_RENDER_PT_photon_adaptive_reconstruction, ui.CYCLES_RENDER_PT_photon_guiding):
        attrs = dict(__module__=__name__, bl_label=native.bl_label,
                     bl_translation_context=native.bl_translation_context, draw=native.draw)
        if 'draw_header_preset' in native.__dict__:
            attrs['draw_header_preset'] = native.draw_header_preset
        classes.append(type(native.__name__.replace('CYCLES_RENDER_', 'GZGUIDE_'),
                            (PhotonPanel, bpy.types.Panel), attrs))
    classes.extend((GZGUIDE_PT_photon_schedule, GZGUIDE_PT_sampling_help, GZGUIDE_PT_footprint_help,
                    GZGUIDE_PT_reconstruction_help, GZGUIDE_PT_guiding_help, GZGUIDE_PT_schedule_help, GZGUIDE_PT_photon_sources))
    try:
        for cls in classes:
            bpy.utils.register_class(cls)
            _registered.append(cls)
    except Exception:
        unregister()
        raise


def unregister():
    for cls in reversed(_registered):
        bpy.utils.unregister_class(cls)
    _registered.clear()
    del bpy.types.Scene.gz_guide_use_photons
