/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: Apache-2.0 */

#include "blender/device.h"
#include "blender/util.h"

#include "BKE_collection.hh"
#include "BKE_material.hh"
#include "BKE_node_runtime.hh"
#include "DEG_depsgraph.hh"
#include "DEG_depsgraph_query.hh"
#include "DNA_camera_types.h"
#include "DNA_layer_types.h"
#include "DNA_light_types.h"
#include "DNA_material_types.h"
#include "DNA_world_types.h"

#include <unordered_set>

CCL_NAMESPACE_BEGIN

namespace {
using namespace blender;

constexpr const char *medium = "Volume transport needs the native integrator";
constexpr const char *subsurface = "Subsurface scattering needs the native integrator";
constexpr const char *portal = "Ray portals need the native integrator";
constexpr const char *path_state = "The shader uses camera or path state";
constexpr const char *surface = "The surface needs the native integrator";
constexpr const char *light = "The light configuration needs the native integrator";
constexpr const char *camera = "The camera projection or depth of field needs the native integrator";

/* Walk upstream from the selected Cycles output. Keep a visited set per group instance, so
 * two instances with different external connections cannot mask each other's dependencies. */
struct ShaderWalk {
  const bNode *group = nullptr;
  ShaderWalk *parent = nullptr;
  bool mesh_surface = false;
  std::unordered_set<const bNodeSocket *> visited;

  const char *input(const bNodeSocket *in)
  {
    if (!in || !in->is_available()) {
      return nullptr;
    }
    for (const bNodeSocket *out : in->logically_linked_sockets()) {
      if (!visited.insert(out).second) {
        continue;
      }
      const bNode &node = out->owner_node();
      if (node.is_group_input() && group) {
        if (const char *reason = parent->input(group->input_by_identifier(out->identifier_ustr())))
        {
          return reason;
        }
        continue;
      }
      if (node.is_group() && node.id) {
        auto &tree = *reinterpret_cast<bNodeTree *>(node.id);
        tree.ensure_topology_cache();
        if (const bNode *end = tree.group_output_node()) {
          ShaderWalk nested{&node, this, mesh_surface, {}};
          if (const char *reason = nested.input(end->input_by_identifier(out->identifier_ustr())))
          {
            return reason;
          }
        }
        continue;
      }
      if (node.is_type("ShaderNodeLightPath"_ustr)) {
        return path_state;
      }
      if (node.is_type("ShaderNodeBsdfRayPortal"_ustr)) {
        return portal;
      }
      if (node.is_type("ShaderNodeSubsurfaceScattering"_ustr)) {
        return subsurface;
      }
      if (node.is_type("ShaderNodeBsdfPrincipled"_ustr)) {
        const auto *weight = node.input_by_identifier("Subsurface Weight"_ustr);
        if (weight && (weight->is_logically_linked() ||
                       weight->default_value_typed<bNodeSocketValueFloat>()->value > 0.0f))
        {
          return subsurface;
        }
        const auto *alpha = node.input_by_identifier("Alpha"_ustr);
        if (alpha && (alpha->is_logically_linked() ||
                      alpha->default_value_typed<bNodeSocketValueFloat>()->value < 1.0f))
        {
          return surface;
        }
        const auto *color = node.input_by_identifier("Emission Color"_ustr);
        const auto *strength = node.input_by_identifier("Emission Strength"_ustr);
        if (mesh_surface && color && strength &&
            (strength->is_logically_linked() ||
             strength->default_value_typed<bNodeSocketValueFloat>()->value != 0.0f))
        {
          const float *rgb = color->default_value_typed<bNodeSocketValueRGBA>()->value;
          if (color->is_logically_linked() || rgb[0] || rgb[1] || rgb[2]) {
            return surface;
          }
        }
      }
      if (node.is_type("ShaderNodeBsdfTransparent"_ustr) ||
          node.is_type("ShaderNodeHoldout"_ustr) || node.is_type("ShaderNodeScript"_ustr) ||
          (mesh_surface && node.is_type("ShaderNodeEmission"_ustr)))
      {
        return surface;
      }
      for (const bNodeSocket *socket : node.input_sockets()) {
        if (const char *reason = input(socket)) {
          return reason;
        }
      }
    }
    return nullptr;
  }
};

const char *check_tree(bNodeTree *tree, const bool mesh_surface)
{
  if (!tree) {
    return nullptr;
  }
  tree->ensure_topology_cache();
  /* Match ntreeShaderOutputNode's engine and active-output preference without tagging or
   * mutating the original tree from a properties-panel draw. */
  const bNode *output = nullptr;
  int best = -1;
  for (const bNode &node : tree->nodes) {
    if (!(node.is_type("ShaderNodeOutputMaterial"_ustr) ||
          node.is_type("ShaderNodeOutputLight"_ustr) ||
          node.is_type("ShaderNodeOutputWorld"_ustr)) ||
        (node.custom1 != SHD_OUTPUT_ALL && node.custom1 != SHD_OUTPUT_CYCLES))
    {
      continue;
    }
    const int rank = 2 * int(node.custom1 == SHD_OUTPUT_CYCLES) +
                     int(bool(node.flag & NODE_DO_OUTPUT));
    if (rank > best) {
      output = &node;
      best = rank;
    }
  }
  if (!output) {
    return nullptr;
  }
  const auto *volume = output->input_by_identifier("Volume"_ustr);
  if (volume && volume->is_logically_linked()) {
    return medium;
  }
  ShaderWalk walk{nullptr, nullptr, mesh_surface, {}};
  return walk.input(output->input_by_identifier("Surface"_ustr));
}

struct SceneCheck {
  std::unordered_set<const blender::Material *> materials;

  const char *object(blender::Object &obj, const bool evaluated, blender::Material *override_mat)
  {
    if (obj.type == OB_VOLUME) {
      return medium;
    }
    if (obj.light_linking &&
        (obj.light_linking->receiver_collection || obj.light_linking->blocker_collection))
    {
      return light;
    }
    if (obj.type == OB_LAMP) {
      const blender::Light &lamp = *blender::id_cast<blender::Light *>(obj.data);
      if (lamp.energy == 0.0f) {
        return nullptr;
      }
      if (lamp.type == LA_SUN ||
          ((lamp.type == LA_LOCAL || lamp.type == LA_SPOT) && lamp.radius > 0) ||
          (lamp.type == LA_AREA && lamp.area_spread < float(M_PI)) || !(lamp.mode & LA_SHADOW) ||
          (obj.visibility_flag & (OB_HIDE_DIFFUSE | OB_HIDE_GLOSSY | OB_HIDE_TRANSMISSION)))
      {
        return light;
      }
      return check_tree(lamp.nodetree, false);
    }
    if (obj.type == OB_EMPTY || obj.type == OB_CAMERA || obj.type == OB_ARMATURE) {
      return nullptr;
    }
    if (obj.type == OB_CURVES || obj.type == OB_POINTCLOUD) {
      return surface;
    }
    if (obj.visibility_flag & (OB_HOLDOUT | OB_SHADOW_CATCHER)) {
      return surface;
    }
    blender::PointerRNA obj_ptr = RNA_id_pointer_create(&obj.id);
    blender::PointerRNA settings = RNA_pointer_get(&obj_ptr, "cycles");
    const bool nested = get_boolean(settings, "use_nested_dielectric");
    const int count = override_mat ?
                          1 :
                          (evaluated ? BKE_object_material_count_eval(&obj) : obj.totcol);
    for (int i = 1; i <= count; ++i) {
      blender::Material *mat = override_mat ? override_mat :
                                              (evaluated ? BKE_object_material_get_eval(&obj, i) :
                                                           BKE_object_material_get(&obj, i));
      if (nested && mat) {
        blender::PointerRNA mat_ptr = RNA_id_pointer_create(&mat->id);
        blender::PointerRNA settings = RNA_pointer_get(&mat_ptr, "cycles");
        if (get_int(settings, "nested_dielectric_priority") > 0) {
          return surface;
        }
      }
      if (mat && materials.insert(mat).second) {
        if (const char *reason = check_tree(mat->nodetree, true)) {
          return reason;
        }
      }
    }
    return nullptr;
  }
};

const char *classify(blender::Scene &scene, Depsgraph *graph)
{
  blender::PointerRNA ptr = RNA_id_pointer_create(&scene.id);
  blender::PointerRNA cycles = RNA_pointer_get(&ptr, "cycles");
  if (get_boolean(cycles, "shading_system") || (scene.r.alphamode & R_ALPHAPREMUL)) {
    return surface;
  }
  if (scene.camera && scene.camera->type == OB_CAMERA) {
    const auto &cam = *blender::id_cast<blender::Camera *>(scene.camera->data);
    if (cam.type != CAM_PERSP || (cam.dof.flag & CAM_DOF_ENABLED)) {
      return camera;
    }
  }
  const auto *layer = graph ? DEG_get_evaluated_view_layer(graph) : nullptr;
  const auto *world = layer && layer->world_override ? layer->world_override : scene.world;
  if (world) {
    if (const char *reason = check_tree(world->nodetree, false)) {
      return reason;
    }
  }
  SceneCheck check;
  if (graph) {
    DEGObjectIterSettings settings{};
    settings.depsgraph = graph;
    settings.flags = DEG_OBJECT_ITER_FOR_RENDER_ENGINE_FLAGS;
    DEGObjectIterData iter{};
    iter.settings = &settings;
    iter.graph = graph;
    iter.flag = settings.flags;
    blender::Material *override_mat = layer->mat_override;
    const char *reason = nullptr;
    ITER_BEGIN (DEG_iterator_objects_begin,
                DEG_iterator_objects_next,
                DEG_iterator_objects_end,
                &iter,
                blender::Object *,
                obj)
    {
      if (BKE_object_visibility(obj, iter.eval_mode) & OB_VISIBLE_SELF) {
        reason = check.object(*obj, true, override_mat);
        if (reason) {
          break;
        }
      }
    }
    ITER_END;
    return reason;
  }
  if (scene.master_collection) {
    for (Base &base : BKE_collection_object_cache_instanced_get(scene.master_collection)) {
      if ((base.flag & BASE_ENABLED_RENDER) && !(base.object->visibility_flag & OB_HIDE_RENDER)) {
        if (const char *reason = check.object(*base.object, false, nullptr)) {
          return reason;
        }
      }
    }
  }
  return nullptr;
}
}  // namespace

const char *blender_vcm_fallback_reason(blender::Scene &scene)
{
  auto *graph = DEG_get_depsgraph_by_id(scene.id);
  const uint64_t revision = graph ? blender::DEG_get_update_count(graph) :
                                    blender::DEG_get_render_data_update_count();
  struct Cache {
    const blender::Scene *scene = nullptr;
    unsigned int uid = 0;
    const blender::Depsgraph *graph = nullptr;
    uint64_t revision = 0;
    const char *reason = nullptr;
  };
  static thread_local Cache cache;
  if (cache.scene != &scene || cache.uid != scene.id.session_uid || cache.graph != graph ||
      cache.revision != revision)
  {
    cache = {&scene, scene.id.session_uid, graph, revision, classify(scene, graph)};
  }
  return cache.reason;
}

CCL_NAMESPACE_END
