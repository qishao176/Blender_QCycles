/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: Apache-2.0 */

#include "blender/lpe.h"
#include "blender/sync.h"
#include "scene/film.h"
#include "scene/background.h"
#include "scene/object.h"
#include "scene/light.h"
#include "scene/shader.h"

#include "BKE_node_runtime.hh"
#include "BKE_global.hh"
#include "BKE_lib_id.hh"
#include "BKE_main.hh"
#include "BLI_listbase.hh"
#include "BLT_translation.hh"
#include "util/string.h"
#include "DNA_anim_types.h"
#include "DEG_depsgraph_query.hh"
#include "DNA_collection_types.h"
#include "DNA_layer_types.h"
#include "DNA_light_types.h"
#include "DNA_material_types.h"
#include "DNA_node_types.h"
#include "DNA_object_types.h"
#include "DNA_scene_types.h"
#include "RE_lpe.hh"
#include "RE_pipeline.h"
#include "RNA_access.hh"
#include "scene/integrator.h"

#include <cstdlib>
#include <cstdio>
#include <algorithm>
#include <type_traits>

CCL_NAMESPACE_BEGIN

namespace {

int event_for_socket(const blender::bNodeSocket &socket)
{
  const string name = socket.identifier;
  if (name == "Total Light" || name == "Mask" || name == "Emission" || name == "Photon")
    return LPE_ALL;
  if (name == "Diffuse")
    return LPE_DIFFUSE;
  if (name == "SSS")
    return LPE_SUBSURFACE;
  if (name == "Glossy")
    return LPE_GLOSSY;
  if (name == "Caustics")
    return LPE_GLOSSY | LPE_TRANSMISSION;
  if (name == "Transmission")
    return LPE_TRANSMISSION;
  if (name == "Transparent")
    return LPE_TRANSPARENT;
  if (name == "Volume")
    return LPE_VOLUME;
  return 0;
}

const blender::ID *original_id(const void *data)
{
  return data ? blender::DEG_get_original_id(
                    const_cast<blender::ID *>(static_cast<const blender::ID *>(data))) :
                nullptr;
}

std::map<const blender::ID *, const blender::Object *> selected_objects(
    const blender::bNode &node, const bool lights_only, string &error)
{
  using enum blender::ObjectType;
  std::map<const blender::ID *, const blender::Object *> result;
  const auto add_object = [&](const blender::Object *object, const bool collection_member) {
    if (!object)
      return;
    const bool accepted = lights_only ? (object->type == blender::OB_LAMP ||
        (collection_member && object->type == blender::OB_EMPTY && object->instance_collection)) :
        (object->type == blender::OB_EMPTY || OB_TYPE_IS_GEOMETRY(object->type));
    if (accepted) {
      result[original_id(object)] = object;
    }
    else if (!collection_member) {
      error = lights_only ? "Light nodes accept light objects only" :
                            "Object nodes accept geometry and instance objects only";
    }
  };
  for (const blender::bNodeSocket *input : node.input_sockets()) {
    if (input->type != blender::SOCK_OBJECT)
      continue;
    const blender::bNodeSocket *socket = input;
    std::set<const blender::bNodeSocket *> visited_sockets;
    const blender::bNodeSocket *linked = nullptr;
    while (visited_sockets.insert(socket).second) {
      linked = nullptr;
      for (const auto *link : socket->directly_linked_links()) {
        if (!(link->flag & blender::NODE_LINK_MUTED)) {
          linked = link->fromsock;
          break;
        }
      }
      if (!linked || !linked->owner_node().is_reroute())
        break;
      socket = &linked->owner_node().input_socket(0);
    }
    if (!linked) {
      if (socket->type == blender::SOCK_OBJECT && socket->default_value)
        add_object(socket->default_value_typed<blender::bNodeSocketValueObject>()->value, false);
      continue;
    }
    const blender::bNode &source = linked->owner_node();
    if (!STREQ(source.idname, "LPENodeCollection") ||
        !STREQ(linked->identifier, lights_only ? "Lights" : "Objects")) {
      error = lights_only ? "Connect Collection Lights to a Light selection" :
                            "Connect Collection Objects to an Object selection";
      continue;
    }
    if (source.is_muted())
      continue;
    const auto *collection = source.input_socket(0)
                                 .default_value_typed<blender::bNodeSocketValueCollection>()->value;
    std::set<const blender::Collection *> visited;
    const auto visit = [&](auto &&self, const blender::Collection *current) -> void {
      if (!current || !visited.insert(current).second)
        return;
      for (const blender::CollectionObject &member : current->gobject)
        add_object(member.ob, true);
      for (const blender::CollectionChild &child : current->children)
        self(self, child.collection);
    };
    visit(visit, collection);
  }
  return result;
}

template<typename Selection>
bool selection_matches_object(const Selection &selected, const ObjectKey &key,
                              const Object &object, const bool lights)
{
  /* Object/Collection Objects selects geometry descendants, not light objects
   * that happen to share the same instance host. Mesh emitters remain geometry. */
  if (!lights && object.get_geometry() && object.get_geometry()->is_light()) return false;
  const auto *source = original_id(key.ob);
  if (selected.contains(source))
    return true;
  const auto *parent = original_id(key.parent);
  if (lights && (!source || source->id_type() != blender::ID_OB ||
                 reinterpret_cast<const blender::Object *>(source)->type != blender::OB_LAMP))
    return false;
  if (parent && selected.contains(parent)) return true;
  for (const auto &entry : selected) {
    const blender::ID *id;
    if constexpr (std::is_convertible_v<decltype(entry), const blender::ID *>) id = entry;
    else id = entry.first;
    if (std::find(object.lpe_instance_ancestor_ids.begin(), object.lpe_instance_ancestor_ids.end(),
                  id->session_uid) != object.lpe_instance_ancestor_ids.end()) return true;
  }
  return false;
}

bool shadow_caustics(const blender::bNodeSocket &socket)
{
  const auto &node = socket.owner_node();
  return STREQ(node.idname, "LPENodeObject") && node.storage &&
         (static_cast<const blender::NodeLPE *>(node.storage)->mode & 1);
}

int caustics_caster_index(const std::vector<const blender::bNodeSocket *> &path)
{
  for (size_t i = 0; i < path.size(); ++i) {
    if (STREQ(path[i]->identifier, "Caustics")) {
      return int(i);
    }
  }
  return -1;
}

int caustics_last_caster_index(const std::vector<const blender::bNodeSocket *> &path)
{
  for (int i = int(path.size()) - 1; i >= 0; --i) {
    if (STREQ(path[i]->identifier, "Caustics")) {
      return i;
    }
  }
  return -1;
}

bool path_total_caustic_source(const std::vector<const blender::bNodeSocket *> &path)
{
  return path.size() > 1 && caustics_caster_index(path) < 0 && !shadow_caustics(*path.back()) &&
         (STREQ(path[0]->owner_node().idname, "LPENodeLight") ||
          STREQ(path[0]->owner_node().idname, "LPENodeWorld")) &&
         (STREQ(path[0]->identifier, "Total Light") || STREQ(path[0]->identifier, "Photon")) &&
         STREQ(path[1]->identifier, "Total Light");
}

/* Emit the same matcher states for UI budgets and device synchronization.
 * Selection binding is deferred: it changes IDs, never the state topology. */
template<typename Resolve>
vector<KernelLPEState> compile_path_states(
    const std::vector<const blender::bNodeSocket *> &path,
    Resolve &&resolve, const bool photon, string &error)
{
  vector<KernelLPEState> steps;
  KernelLPEState terminal = {};
  terminal.pass_offset = 0;
  terminal.light_count = -1;
  terminal.flags = LPE_LOOP;
  const int caustics_caster = caustics_caster_index(path);
  const int last_caster = caustics_last_caster_index(path);
  if (last_caster > caustics_caster && !photon) {
    error =
        "Multiple ordered Caustics outputs require Photon; MNEE supports one caster selection";
    return {};
  }
  const bool caustics = caustics_caster < 0 && !path.empty() && shadow_caustics(*path.back());
  if (last_caster >= 0 && last_caster + 1 == int(path.size())) {
    /* A caster output can go straight to Output: match any actual caustic receiver,
     * including receivers visible after camera-side reflection/transmission. */
    KernelLPEState receiver = {};
    receiver.events = LPE_DIFFUSE | LPE_GLOSSY | LPE_TRANSMISSION;
    if (photon) {
      receiver.events |= LPE_VOLUME;
    }
    receiver.flags = LPE_MNEE_RECEIVER | LPE_LOOP;
    receiver.pass_offset = PASS_UNUSED;
    steps.push_back(receiver);
  }
  bool weighted = false;
  for (auto it = path.rbegin(); it != path.rend(); ++it) {
    const int path_index = int(path.size() - 1 - size_t(it - path.rbegin()));
    const blender::bNodeSocket &socket = **it;
    const blender::bNode &node = socket.owner_node();
    const int events = event_for_socket(socket);
    const bool world = STREQ(node.idname, "LPENodeWorld");
    const bool emission = STREQ(socket.identifier, "Emission");
    if (world || STREQ(node.idname, "LPENodeLight") || emission) {
      if (caustics && path.size() == 2) {
        KernelLPEState caster = {};
        caster.events = LPE_GLOSSY | LPE_TRANSMISSION;
        caster.flags = LPE_REPEAT_MATCH;
        caster.pass_offset = PASS_UNUSED;
        steps.push_back(caster);
      }
      resolve(socket, terminal, true);
      if (node.is_muted())
        terminal.light_count = 0;
      if (!steps.empty()) {
        steps.back().events &= events;
        if (!steps.back().events)
          error = "Source and receiving selection have incompatible event types";
        terminal.flags = 0;
      }
      else if (events != LPE_ALL) {
        KernelLPEState prefix = {};
        prefix.events = events;
        prefix.pass_offset = PASS_UNUSED;
        prefix.flags = LPE_LOOP;
        steps.push_back(prefix);
        terminal.flags = 0;
      }
      /* Total Light includes the implicit specular connection to its source.
       * Only the caustic replay may retain this terminal through extra vertices;
       * ordinary explicit path adjacency and caster edges keep their semantics. */
      if (path_total_caustic_source(path)) {
        terminal.flags |= LPE_CAUSTIC_SOURCE;
      }
      if (STREQ(socket.identifier, "Photon")) {
        terminal.flags |= LPE_PHOTON_SOURCE;
      }
      weighted |= events != LPE_ALL;
      if (emission && STREQ(node.idname, "LPENodeMaterial"))
        terminal.flags |= LPE_SOURCE_MATERIAL;
      break;
    }
    const bool material = STREQ(node.idname, "LPENodeMaterial");
    KernelLPEState step = {};
    step.events = events;
    step.selector = material ? 2 : 1;
    step.pass_offset = PASS_UNUSED;
    if (caustics || caustics_caster >= 0) {
      const bool is_receiver = caustics ? steps.empty() : path_index == last_caster + 1;
      const bool is_caster = caustics ?
                                 !steps.empty() :
                                 (path_index >= caustics_caster && path_index <= last_caster);
      if (is_receiver) {
        if (!photon && events == LPE_VOLUME) {
          error = "MNEE caustics require a surface receiver; use Photon for volume caustics";
        }
        step.flags |= LPE_MNEE_RECEIVER;
      }
      else if (is_caster) {
        step.events &= LPE_GLOSSY | LPE_TRANSMISSION;
        step.flags |= LPE_REPEAT_MATCH;
      }
    }
    resolve(socket, step, false);
    steps.push_back(step);
    weighted |= events != LPE_ALL;
  }
  if (weighted)
    terminal.flags |= LPE_WEIGHTED;
  steps.push_back(terminal);
  return steps;
}


template<typename Resolve>
vector<KernelLPEState> compile_mask_states(
    const std::vector<const blender::bNodeSocket *> &full_path,
    Resolve &&resolve, string &error)
{
  /* Coverage stops at the first selected surface on the light side. A source
   * seed controls radiance, not opacity. Keep the seed in RGB paths, but allow
   * the same authored path to expose illumination-independent alpha. */
  std::vector<const blender::bNodeSocket *> path = full_path;
  if (!path.empty() && (STREQ(path.front()->owner_node().idname, "LPENodeLight") ||
                        STREQ(path.front()->owner_node().idname, "LPENodeWorld") ||
                        STREQ(path.front()->identifier, "Emission")))
  {
    path.erase(path.begin());
  }
  if (path.empty()) {
    error = "Mask needs an Object or Material target after the Light/World source";
    return {};
  }
  for (const blender::bNodeSocket *socket : path) {
    if (!STREQ(socket->owner_node().idname, "LPENodeObject") &&
        !STREQ(socket->owner_node().idname, "LPENodeMaterial"))
    {
      error = "Mask requires an ordered Object/Material coverage path";
      return {};
    }
  }
  if (!error.empty())
    return {};
  vector<KernelLPEState> states;
  bool weighted = false;
  for (size_t i = 0; i < path.size(); i++) {
    const blender::bNodeSocket &socket = *path[path.size() - 1 - i];
    const blender::bNode &node = socket.owner_node();
    const bool material = STREQ(node.idname, "LPENodeMaterial");
    KernelLPEState step = {};
    step.selector = material ? 2 : 1;
    resolve(socket, step);
    if (node.is_muted()) step.selection_count = 0;
    step.light_count = -1;
    if (i + 1 == path.size()) {
      step.flags = LPE_MASK_TARGET | (weighted ? LPE_WEIGHTED : 0);
      if (path.size() == 1)
        step.flags |= LPE_LOOP;
      step.pass_offset = 0;
    }
    else {
      step.events = event_for_socket(socket);
      weighted |= step.events != LPE_ALL;
      /* Transitions are relocated when the path is appended to the film. */
      step.pass_offset = PASS_UNUSED;
    }
    states.push_back(step);
  }
  return states;
}


string sampling_key(const std::vector<const blender::bNodeSocket *> &path)
{
  string key;
  for (const auto *socket : path) {
    const auto &node = socket->owner_node();
    key += std::to_string(node.identifier) + ":" + socket->identifier + ":" +
           std::to_string(node.is_muted()) + ":" + std::to_string(shadow_caustics(*socket));
    if (STREQ(node.idname, "LPENodeObject") || STREQ(node.idname, "LPENodeLight")) {
      string error;
      for (const auto &[id, object] : selected_objects(node, STREQ(node.idname, "LPENodeLight"), error))
        key += ":" + std::to_string(id->session_uid);
      key += error;
    }
    for (const auto *input : node.input_sockets()) {
      if (input->type == blender::SOCK_MATERIAL) {
        const auto *id = original_id(input->default_value_typed<blender::bNodeSocketValueMaterial>()->value);
        key += ":M" + std::to_string(id ? id->session_uid : 0);
      }
    }
    key += ";";
  }
  return key;
}

std::map<string, string> graph_pass_keys(const LPEGraph &graph)
{
  std::map<string, string> keys;
  if (graph.shadow_catcher)
    keys["Shadow Catcher"] = "N:ShadowCatcher";
  for (size_t i = 0; i < graph.shadows.size(); i++) {
    const string base = "__LPE_shadow_" + std::to_string(i);
    keys[base + "_reference"] = "SR" + sampling_key(graph.shadows[i]);
    keys[base + "_blocked"] = "SB" + sampling_key(graph.shadows[i]);
  }
  for (size_t i = 0; i < graph.paths.size(); i++)
    keys["__LPE_raw_" + std::to_string(i)] = "P" + sampling_key(graph.paths[i]);
  for (const auto &[path, expression] : graph.masks)
    keys[graph.program->expressions[expression].pass] = "M" + sampling_key(path);
  for (size_t i = 0; i < graph.unions.size(); i++) {
    std::set<string> parts;
    for (const auto &path : graph.unions[i])
      parts.insert(sampling_key(path));
    string key = "U";
    for (const auto &part : parts)
      key += "[" + part + "]";
    keys["__LPE_union_" + std::to_string(i)] = key;
  }
  return keys;
}

bool coalesce_light_sources(const std::vector<std::vector<const blender::bNodeSocket *>> &paths)
{
  if (paths.empty()) return false;
  for (const auto &path : paths) {
    if (path.size() != 1 || !STREQ(path[0]->owner_node().idname, "LPENodeLight") ||
        !STREQ(path[0]->identifier, paths[0][0]->identifier)) return false;
  }
  return true;
}

bool weighted_path(const std::vector<const blender::bNodeSocket *> &path)
{
  for (const auto *socket : path)
    if (event_for_socket(*socket) != LPE_ALL) return true;
  return false;
}

void record_sampling_bindings(LPEGraph &graph)
{
  const auto keys = graph_pass_keys(graph);
  for (const auto &expression : graph.program->expressions) {
    if (const auto found = keys.find(expression.pass); found != keys.end())
      graph.program->sampling_bindings[found->second] = expression;
  }
  for (const auto &edit : graph.edits)
    graph.program->sampling_bindings["E" + sampling_key(edit.path)] = graph.program->expressions[edit.expression];
}

}  // namespace

bool rebind_lpe_program(LPEGraph &graph, const LPEProgram &sampled,
                       const blender::bNodeTree &tree, string &error, const bool clamp_missing)
{
  const auto keys = graph_pass_keys(graph);
  for (auto &expression : graph.program->expressions) {
    if (expression.operation != LPEOperation::PASS && expression.operation != LPEOperation::MASK)
      continue;
    const auto key = keys.find(expression.pass);
    if (key == keys.end())
      return false;
    const auto bound = sampled.sampling_bindings.find(key->second);
    if (bound == sampled.sampling_bindings.end()) {
      /* A source already split into editable buckets also supplies its original
       * (unmodified) Viewer. Do not demand a second sample pass for that alias. */
      const auto editable = sampled.sampling_bindings.find("E" + key->second.substr(1));
      if (key->second[0] != 'P' || editable == sampled.sampling_bindings.end()) return false;
      expression = editable->second;
      for (auto &term : expression.edit_terms) {
        term.manager_node = term.output_node = 0;
        term.manager_item.clear();
        term.output_item.clear();
        term.operation_nodes.clear();
        term.scene_operation_nodes.clear();
        term.source_parameters = term.sampling_parameters;
      }
      continue;
    }
    expression.pass = bound->second.pass;
  }
  for (const auto &edit : graph.edits) {
    const auto bound = sampled.sampling_bindings.find("E" + sampling_key(edit.path));
    if (bound == sampled.sampling_bindings.end()) {
      return false;
    }
    auto &expression = graph.program->expressions[edit.expression];
    expression.edit_terms = bound->second.edit_terms;
    for (auto &term : expression.edit_terms) {
      if (edit.manager_node && term.global_pass.empty()) {
        if (edit.path.size() != 1 || !STREQ(edit.path.front()->identifier, "Total Light"))
          return false;
        term.global_pass = term.pass;
      }
      term.manager_node = edit.manager_node;
      term.manager_item = edit.manager_item;
      term.operation_nodes = edit.operation_nodes;
      term.scene_operation_nodes = edit.scene_operation_nodes;
      term.output_node = edit.output_node;
      term.output_item = edit.output_item;
    }
  }
  graph.program->sampling_bindings = sampled.sampling_bindings;
  graph.program->sample_clamped = sampled.sample_clamped;
  return update_lpe_edit_terms(*graph.program, tree, error, clamp_missing);
}

std::set<string> lpe_resident_keys(const LPEGraph &graph)
{
  std::set<string> result;
  const auto keys = graph_pass_keys(graph);
  for (int i = 0; i < graph.resident_expressions; i++) {
    const auto &expression = graph.program->expressions[i];
    if (const auto found = keys.find(expression.pass); found != keys.end())
      result.insert(found->second);
  }
  for (const auto &edit : graph.edits) {
    if (edit.expression >= graph.resident_expressions) continue;
    string ignored;
    if (!edit.path.empty() && STREQ(edit.path.front()->owner_node().idname, "LPENodeLight") &&
        selected_objects(edit.path.front()->owner_node(), true, ignored).empty())
      continue;
    result.insert("E" + sampling_key(edit.path));
  }
  return result;
}

std::set<string> lpe_sampling_keys(const LPEGraph &graph)
{
  LPEGraph all = graph;
  all.resident_expressions = int(graph.program->expressions.size());
  return lpe_resident_keys(all);
}

bool lpe_can_reuse_sampling(const LPEGraph &graph, const std::set<string> &sampled_keys)
{
  const auto required = lpe_sampling_keys(graph);
  /* Removing a Viewer only drops consumers. Keep its buffers until a genuine sampling
   * change. A raw light path can also be reconstructed from the resident edit buckets. */
  return std::all_of(required.begin(), required.end(), [&](const string &key) {
    return sampled_keys.contains(key) ||
           (key.starts_with("P") && sampled_keys.contains("E" + key.substr(1)));
  });
}

bool lpe_paths_disjoint(const std::vector<const blender::bNodeSocket *> &a,
                        const std::vector<const blender::bNodeSocket *> &b)
{
  if (a.empty() || b.empty() || !STREQ(a.front()->owner_node().idname, "LPENodeLight") ||
      !STREQ(b.front()->owner_node().idname, "LPENodeLight"))
    return false;
  if (a.size() == 1 && b.size() == 1 &&
      !(event_for_socket(*a.front()) & event_for_socket(*b.front())))
    return true;
  string error;
  const auto left = selected_objects(a.front()->owner_node(), true, error);
  const auto right = selected_objects(b.front()->owner_node(), true, error);
  if (!error.empty()) return false;
  /* An instance-parent selector can overlap a directly selected child light.
   * Keep that case conservative until an instance-aware union estimator exists. */
  for (const auto &[id, object] : left)
    if (object->type == blender::OB_EMPTY || right.contains(id)) return false;
  for (const auto &[id, object] : right)
    if (object->type == blender::OB_EMPTY) return false;
  return true;
}

bool apply_lpe_lights(blender::Main &bmain, blender::bNodeTree &tree,
                      const blender::render::LPEApplyScope &scope,
                      blender::render::LPEApplyChanges &changes, string &error)
{
  tree.ensure_topology_cache();
  const LPEGraph graph = compile_lpe_graph(tree, error);
  if (!error.empty()) return false;
  struct Write {
    blender::Light *light;
    LPELightParameters target;
    std::set<unsigned int> direct_objects;
    std::set<unsigned int> instance_roots;
  };
  std::map<unsigned int, Write> writes;
  std::set<int> mute_nodes;
  const auto selected = [&](const LPEEditRequest &request) {
    return scope.output ? request.output_node == scope.node && request.output_item == scope.item :
                          request.manager_node == scope.node && request.manager_item == scope.item;
  };
  bool found_scope = false;
  for (const auto &request : graph.edits) {
    if (!selected(request)) continue;
    found_scope = true;
    const auto &operations = scope.output ? request.operation_nodes : request.scene_operation_nodes;
    /* A pass-through or already-applied path has no light data to write. */
    if (std::none_of(operations.begin(), operations.end(), [&](const int identifier) {
          const auto *node = tree.node_by_id(identifier);
          return node && !node->is_muted();
        })) {
      continue;
    }
    std::map<const blender::ID *, const blender::Object *> objects;
    std::set<unsigned int> direct_objects, instance_roots;
    std::set<const blender::Collection *> visited;
    const auto collection_lights = [&](auto &&self, const blender::Collection *collection) -> void {
      if (!collection || !visited.insert(collection).second) return;
      for (const auto &member : collection->gobject) {
        if (member.ob->type == blender::OB_LAMP) objects[original_id(member.ob)] = member.ob;
        if (member.ob->instance_collection) self(self, member.ob->instance_collection);
      }
      for (const auto &child : collection->children) self(self, child.collection);
    };
    for (const auto &[id, object] : selected_objects(request.path.front()->owner_node(), true, error)) {
      if (object->type == blender::OB_LAMP) {
        objects[id] = object;
        direct_objects.insert(id->session_uid);
      }
      else {
        instance_roots.insert(id->session_uid);
        collection_lights(collection_lights, object->instance_collection);
      }
    }
    if (!error.empty()) return false;
    for (const auto &[id, object] : objects) {
      auto *light = reinterpret_cast<blender::Light *>(
          const_cast<blender::ID *>(original_id(object->data)));
      if (!light) continue;
      if (!blender::BKE_id_is_editable(&bmain, &light->id)) {
        error = "Create an editable library override for the source light data";
        return false;
      }
      if (light->adt && (light->adt->action || !light->adt->drivers.is_empty() ||
                         !light->adt->nla_tracks.is_empty())) {
        error = "Apply does not overwrite animated or driven light data";
        return false;
      }
      blender::PointerRNA light_ptr = blender::RNA_id_pointer_create(&light->id);
      for (const char *property_name : {"color", "energy", "exposure", "temperature", "use_temperature"}) {
        auto *property = blender::RNA_struct_find_property(&light_ptr, property_name);
        if (!property || !blender::RNA_property_editable(&light_ptr, property)) {
          error = "The light property is not editable; create an editable library override";
          return false;
        }
      }
      LPEEditTerm term;
      term.source_parameters.color = make_float3(light->r, light->g, light->b);
      term.source_parameters.power = light->energy;
      term.source_parameters.exposure = light->exposure;
      term.source_parameters.temperature = light->temperature;
      term.source_parameters.use_temperature = (light->mode & blender::LA_USE_TEMPERATURE) != 0;
      term.sampling_parameters = term.source_parameters;
      term.operation_nodes = operations;
      LPELightParameters target;
      if (!evaluate_lpe_light_target(term, tree, target, error)) return false;
      if (!isfinite_safe(lpe_light_emission(target))) {
        error = "Non-finite light settings cannot be applied";
        return false;
      }
      const auto [it, inserted] = writes.emplace(light->id.session_uid, Write{light, target, {}, {}});
      if (!inserted && len_squared(lpe_light_emission(it->second.target) -
                                    lpe_light_emission(target)) > 1e-12f) {
        error = "Selected paths request conflicting settings for the same light";
        return false;
      }
      if (direct_objects.contains(id->session_uid)) it->second.direct_objects.insert(id->session_uid);
      it->second.instance_roots.insert(instance_roots.begin(), instance_roots.end());
    }
    if (!objects.empty()) mute_nodes.insert(operations.begin(), operations.end());
  }
  if (!found_scope) {
    error = "Connect a Light path to this Light Edit slot before applying";
    return false;
  }
  for (const auto &[uid, write] : writes) {
    const auto contains_indirect_light = [&](auto &&self, const blender::Collection *collection,
                                             std::set<const blender::Collection *> &visited) -> bool {
      if (!collection || !visited.insert(collection).second) return false;
      for (const auto &entry : collection->gobject) {
        if (entry.ob->type == blender::OB_LAMP && entry.ob->data == &write.light->id &&
            !write.direct_objects.contains(original_id(entry.ob)->session_uid)) return true;
        if (self(self, entry.ob->instance_collection, visited)) return true;
      }
      for (const auto &child : collection->children)
        if (self(self, child.collection, visited)) return true;
      return false;
    };
    const blender::Object *first_user = nullptr;
    for (const blender::Object &object : bmain.objects) {
      /* Count object owners, not ID users (fake users and collection links are harmless). */
      if (object.type == blender::OB_LAMP && object.data == &write.light->id) {
        if (first_user) {
          error = string_printf(blender::RPT_("Light data '%s' is shared by '%s' and '%s'. Make it single-user before applying."),
                                write.light->id.name + 2, first_user->id.name + 2, object.id.name + 2);
          return false;
        }
        first_user = &object;
      }
      if (object.instance_collection && !write.instance_roots.contains(object.id.session_uid)) {
        std::set<const blender::Collection *> visited;
        if (contains_indirect_light(contains_indirect_light, object.instance_collection, visited)) {
          error = "This selection covers only some light instances; select all instances before Apply";
          return false;
        }
      }
    }
  }
  for (const auto &request : graph.edits) {
    if (selected(request) || (!request.manager_node && !request.output_node)) continue;
    for (const int identifier : request.operation_nodes) {
      if (mute_nodes.contains(identifier)) {
        error = "An operation is shared with another edit scope; apply the shared Light Edit input";
        return false;
      }
    }
  }
  /* Validation is complete. Apply parameters once; no image capture or pixel work. */
  for (auto &[uid, write] : writes) {
    auto &light = *write.light;
    const auto &target = write.target;
    light.r = target.color.x; light.g = target.color.y; light.b = target.color.z;
    light.energy = target.power;
    light.exposure = target.exposure;
    light.temperature = target.temperature;
    if (target.use_temperature) light.mode |= blender::LA_USE_TEMPERATURE;
    else light.mode &= ~blender::LA_USE_TEMPERATURE;
    changes.ids.push_back(&light.id);
  }
  for (int identifier : mute_nodes) {
    if (auto *node = tree.node_by_id(identifier)) node->flag |= blender::NODE_MUTED;
  }
  changes.node_tree_changed = !mute_nodes.empty();
  return true;
}

size_t stateful_edit_count(const LPEGraph &graph)
{
  /* Count authored edit scopes, independently of temporary Viewer expressions.
   * The resident sampling keys and rebind determine whether sampling is needed. */
  size_t count = 0;
  for (const auto &edit : graph.edits) {
    if (edit.manager_node || edit.output_node) count++;
  }
  return count;
}
static vector<KernelLPEState> compile_budget_path(
    const std::vector<const blender::bNodeSocket *> &path)
{
  string ignored;
  return compile_path_states(
      path, [](const blender::bNodeSocket &, KernelLPEState &, bool) {}, true, ignored);
}

/* Expand collection instances by occurrence, not just by their shared source ID.
 * Procedural instances require the evaluated scene; don't claim an exact budget
 * for those before synchronization. */
static std::set<string> budget_light_instances(const blender::bNode &source,
                                               bool &dynamic)
{
  std::set<string> result;
  string ignored;
  const auto selected = selected_objects(source, true, ignored);
  const auto visit = [&](auto &&self, const blender::Object &object, const string &prefix,
                         std::set<const blender::Collection *> stack) -> void {
    if (object.type == blender::OB_LAMP && object.data) {
      result.insert(prefix + std::to_string(original_id(&object)->session_uid));
    }
    if (object.type != blender::OB_EMPTY || !object.instance_collection) return;
    dynamic = true;
    if (!stack.insert(object.instance_collection).second) return;
    const string instance = prefix + std::to_string(original_id(&object)->session_uid) + "/";
    const auto collection = [&](auto &&walk, const blender::Collection &group) -> void {
      for (const auto &member : group.gobject) self(self, *member.ob, instance, stack);
      for (const auto &child : group.children) walk(walk, *child.collection);
    };
    collection(collection, *object.instance_collection);
  };
  if (!source.is_muted()) {
    for (const auto &[id, object] : selected) visit(visit, *object, "", {});
  }
  return result;
}

bool lpe_independent_photon_budget(const blender::Scene &scene, const bool preview, const bool deep_output)
{
  if (preview || deep_output)
    return false;
  blender::PointerRNA ptr = RNA_id_pointer_create(const_cast<blender::ID *>(&scene.id));
  if (!RNA_struct_find_property(&ptr, "cycles"))
    return false;
  blender::PointerRNA cycles = RNA_pointer_get(&ptr, "cycles");
  if (RNA_enum_get(&cycles, "caustics_method") != 1 ||
      !RNA_boolean_get(&cycles, "photon_caustics_separate_sampling") ||
      (RNA_enum_get(&cycles, "photon_cache_order") == 2 &&
       !RNA_boolean_get(&cycles, "use_auto_tile")))
    return false;
  return true;
}

/* -1: coupled matcher, 0: ordinary estimator, 1: photon estimator. Generic
 * lighting rules participate in both programs; explicit caustic rules only in
 * the photon program. Path masks and selected shadows stay ordinary-only. */
static bool lpe_path_in_phase(const std::vector<const blender::bNodeSocket *> &path,
                              const int phase)
{
  return phase != 0 ||
         (caustics_caster_index(path) < 0 &&
          (path.empty() || (!shadow_caustics(*path.back()) &&
                            !STREQ(path.front()->identifier, "Photon"))));
}

static blender::render::LPEGraphStatus graph_state_counts(const LPEGraph &graph,
                                                          const int phase = -1)
{
  blender::render::LPEGraphStatus status;
  status.available = true;
  status.limit = LPE_MAX_STATES;
  status.viewer_node = graph.program->viewer_node;
  if (graph.program->viewer_expression >= 0 &&
      size_t(graph.program->viewer_expression) < graph.program->expressions.size()) {
    const auto operation = graph.program->expressions[graph.program->viewer_expression].operation;
    status.viewer_direct = operation == LPEOperation::PASS || operation == LPEOperation::MASK;
  }
  vector<KernelLPEState> lighting, masks, edits;
  const auto compile = [&](const std::vector<const blender::bNodeSocket *> &path) {
    if (!lpe_path_in_phase(path, phase))
      return vector<KernelLPEState>();
    return compile_path_states(
        path, [](const blender::bNodeSocket &, KernelLPEState &, bool) {}, true, status.error);
  };
  const auto append = [](auto &target, const auto &states) {
    target.insert(target.end(), states.begin(), states.end());
  };
  for (const auto &path : graph.paths) append(lighting, compile(path));
  for (const auto &group : graph.unions) {
    if (coalesce_light_sources(group)) {
      append(lighting, compile(group.front()));
      continue;
    }
    std::set<std::tuple<const blender::bNodeSocket *, bool, bool>> sources;
    for (const auto &path : group) {
      if (!lpe_path_in_phase(path, phase))
        continue;
      const auto *first = path.front();
      const auto *source = STREQ(first->owner_node().idname, "LPENodeLight") ||
                                   STREQ(first->owner_node().idname, "LPENodeWorld") ||
                                   STREQ(first->identifier, "Emission") ? first : nullptr;
      auto states = compile(path);
      if (!sources.emplace(source, weighted_path(path), path_total_caustic_source(path)).second &&
          !states.empty()) states.pop_back();
      append(lighting, states);
    }
  }
  for (size_t i = 0; phase != 1 && i < graph.shadows.size(); i++) {
    KernelLPEState rule = {};
    rule.flags = LPE_SHADOW;
    lighting.push_back(rule);
  }
  for (const auto &[path, expression] : graph.masks) {
    if (phase == 1)
      continue;
    append(masks, compile_mask_states(
        path, [](const blender::bNodeSocket &, KernelLPEState &) {}, status.error));
  }
  /* Exactly the same bucket sharing as sync_lpe(), for both render and IPR.
   * Offline light overrides also retain per-instance edit passes. */
  std::set<std::pair<string, string>> buckets;
  std::set<string> global_buckets;
  for (const auto &request : graph.edits) {
    if (request.path.empty() ||
        !STREQ(request.path.front()->owner_node().idname, "LPENodeLight")) continue;
    const auto &source = request.path.front()->owner_node();
    const string path_key = sampling_key(request.path);
    for (const string &instance : budget_light_instances(source, status.has_dynamic_edits)) {
      if (buckets.emplace(path_key, instance).second) {
        append(edits, compile(request.path));
      }
      if (request.manager_node && global_buckets.insert(instance).second &&
          !(request.path.size() == 1 && STREQ(request.path.front()->identifier, "Total Light"))) {
        append(edits, compile({&source.output_socket(0)}));
      }
    }
  }
  status.lighting_states = int(lighting.size());
  status.mask_states = int(masks.size());
  status.render_edit_states = int(edits.size());
  status.viewport_lighting_states = status.lighting_states;
  status.viewport_mask_states = status.mask_states;
  status.viewport_edit_states = status.render_edit_states;
  return status;
}

blender::render::LPEGraphStatus lpe_graph_status(const blender::bNodeTree &tree,
                                                 const blender::Scene *scene)
{
  tree.ensure_topology_cache();
  string error;
  const LPEGraph graph = compile_lpe_graph(tree, error);
  blender::render::LPEGraphStatus status = graph_state_counts(graph);
  if (scene) {
    const blender::Scene *source = DEG_get_original(scene);
    bool deep_output = scene->r.im_format.imtype == blender::R_IMF_IMTYPE_DEEP_EXR;
    for (const blender::ViewLayer &layer : source->view_layers) {
      if (layer.lpe_node_tree == DEG_get_original(&tree) &&
          !blender::RE_scene_deep_exr_passes(*source, *source, layer).empty()) {
        deep_output = true;
        break;
      }
    }
    status.independent_photon = lpe_independent_photon_budget(*scene, false, deep_output);
  }
  const auto native = graph_state_counts(graph, 0);
  const auto photon = graph_state_counts(graph, 1);
  status.native_states = native.states();
  status.photon_states = photon.states();
  string viewport_error;
  const LPEGraph viewport_graph = compile_lpe_graph(tree, viewport_error, true);
  const auto viewport_counts = graph_state_counts(viewport_graph);
  status.viewport_lighting_states = viewport_counts.lighting_states;
  status.viewport_mask_states = viewport_counts.mask_states;
  if (error.empty()) error = status.error;
  if (viewport_error.empty()) viewport_error = viewport_counts.error;
  status.viewport_edit_states = viewport_counts.render_edit_states;
  status.has_dynamic_edits |= viewport_counts.has_dynamic_edits;
  for (const auto &request : graph.edits) {
    if (request.path.empty() || !STREQ(request.path.front()->owner_node().idname, "LPENodeLight"))
      continue;
    bool readonly = false;
    std::set<const blender::Collection *> seen;
    const auto inspect_collection = [&](auto &&self, const blender::Collection *collection) -> void {
      if (!collection || !seen.insert(collection).second)
        return;
      for (const auto &entry : collection->gobject) {
        if (entry.ob->type == blender::OB_LAMP && entry.ob->data &&
            !blender::BKE_id_is_editable(blender::G.main, static_cast<const blender::ID *>(entry.ob->data)))
          readonly = true;
        if (entry.ob->type == blender::OB_EMPTY)
          self(self, entry.ob->instance_collection);
      }
      for (const auto &child : collection->children)
        self(self, child.collection);
    };
    for (const auto &[id, object] : selected_objects(request.path.front()->owner_node(), true, error)) {
      if (object->type == blender::OB_EMPTY)
        inspect_collection(inspect_collection, object->instance_collection);
      else if (object->data && !blender::BKE_id_is_editable(blender::G.main, static_cast<const blender::ID *>(object->data)))
        readonly = true;
    }
    /* A color channel near zero cannot be recovered from the cached
     * contribution by scaling, so the user has to fix the light itself. Use the
     * color rather than the absolute emission: a dim white light scales fine. */
    std::string weak_message;
    for (const auto &[id, object] : selected_objects(request.path.front()->owner_node(), true, error)) {
      if (object->type != blender::OB_LAMP || !object->data)
        continue;
      const auto &light = *reinterpret_cast<const blender::Light *>(object->data);
      const float channels[3] = {light.r, light.g, light.b};
      std::string parts;
      for (int c = 0; c < 3; c++) {
        if (channels[c] >= 0.2f)
          continue;
        char value[16];
        snprintf(value, sizeof(value), "%.2f", channels[c]);
        if (!parts.empty())
          parts += " / ";
        parts += std::string(1, "RGB"[c]) + "=" + value;
      }
      if (parts.empty())
        continue;
      if (!weak_message.empty())
        weak_message += ", ";
      weak_message += std::string(object->id.name + 2) + " " + parts;
    }
    const bool weak = !weak_message.empty();
    if (readonly) {
      status.readonly_light_scopes.push_back("M:" + std::to_string(request.manager_node) + ":" + request.manager_item);
      status.readonly_light_scopes.push_back("O:" + std::to_string(request.output_node) + ":" + request.output_item);
    }
    if (weak) {
      status.weak_light_entries.emplace_back(
          "M:" + std::to_string(request.manager_node) + ":" + request.manager_item, weak_message);
      status.weak_light_entries.emplace_back(
          "O:" + std::to_string(request.output_node) + ":" + request.output_item, weak_message);
    }
  }
  status.error = error.empty() ? viewport_error : error;
  return status;
}

bool validate_lpe_state_budget(const LPEGraph &graph,
                               const char *view_layer_name,
                               string &error,
                               const size_t edit_states,
                               const bool independent_photon)
{
  /* Each sampling phase has its own 64-bit active-state mask in the kernel. */
  if (independent_photon) {
    for (const int phase : {0, 1}) {
      const auto counts = graph_state_counts(graph, phase);
      if (!counts.error.empty()) {
        error = counts.error;
        return false;
      }
      if (counts.states() > int(LPE_MAX_STATES)) {
        error = "View Layer \"" + string(view_layer_name) + "\" " + (phase ? "photon" : "native") +
                " phase requires " + std::to_string(counts.states()) +
                " matcher states; limit is 64 per phase. "
                "Reduce Repeat iterations or connected outputs.";
        return false;
      }
    }
    return true;
  }
  /* edit_states already covers every connected edit path; do not count them again. */
  const auto counts = graph_state_counts(graph);
  if (!counts.error.empty()) {
    error = counts.error;
    return false;
  }
  const size_t lighting = counts.lighting_states;
  const size_t masks = counts.mask_states;
  const size_t total = lighting + masks + edit_states;
  if (total <= LPE_MAX_STATES) {
    return true;
  }
  error = "View Layer \"" + string(view_layer_name) + "\" requires " + std::to_string(total) +
          " matcher states; limit is " + std::to_string(LPE_MAX_STATES) + " (lighting " +
          std::to_string(lighting) + ", masks " + std::to_string(masks) + ", light edits " +
          std::to_string(edit_states) + "). Reduce Repeat iterations or connected outputs.";
  return false;
}

void configure_lpe_preview(Film &film, std::shared_ptr<const LPEProgram> program, const bool enabled)
{
  const string selected = film.get_lpe_display_output();
  if (enabled && program && selected != "LPE Viewer") {
    auto display = std::make_shared<LPEProgram>(*program);
    display->viewer_expression = -1;
    display->viewer_denoising = false;
    for (const auto &output : program->outputs) {
      if (output.name == selected) display->viewer_expression = output.expression;
    }
    if (const auto edit = program->edit_previews.find(selected); edit != program->edit_previews.end())
      display->viewer_expression = edit->second;
    program = std::move(display);
  }
  if (enabled && string_startswith(selected, "N:") && selected != "N:COMBINED") {
    film.lpe_preview_cache.reset();
    film.set_lpe_preview(ustring());
    return;
  }
  if (!enabled || !program || program->viewer_expression < 0) {
    film.lpe_preview_cache.reset();
    const bool managed = program && std::any_of(program->expressions.begin(), program->expressions.end(),
                                                [](const LPEExpression &expression) {
      return std::any_of(expression.edit_terms.begin(), expression.edit_terms.end(),
                         [](const LPEEditTerm &term) { return term.manager_node != 0; });
    });
    if (enabled && program && (managed || !program->combined_terms.empty())) {
      vector<Film::LPEPreview::Term> terms;
      for (const auto &term : program->combined_terms)
        terms.push_back({ustring(term.global_pass), term.multiplier - one_float3()});
      if (terms.size() <= 64) {
        film.set_lpe_linear_preview(std::move(terms), true);
        return;
      }
    }
    film.set_lpe_preview(ustring());
    return;
  }
  const auto &expression = program->expressions[program->viewer_expression];
  if (expression.operation == LPEOperation::MIX) {
    /* Merge of disjoint edited paths is a linear radiance union. Reuse the
     * native GPU pass mixer during interaction instead of reading pixels back
     * and evaluating the expression on the CPU for every slider event. */
    vector<Film::LPEPreview::Term> terms;
    const auto collect = [&](auto &&self, const int index) -> bool {
      if (index < 0 || terms.size() > 64) return false;
      const auto &part = program->expressions[index];
      if (part.operation == LPEOperation::EDIT) {
        for (const auto &term : part.edit_terms)
          terms.push_back({ustring(term.pass), term.multiplier});
        return true;
      }
      if (part.operation == LPEOperation::PASS && !part.scalar &&
          string_startswith(part.pass, "__LPE_")) {
        terms.push_back({ustring(part.pass), one_float3()});
        return true;
      }
      if (part.operation != LPEOperation::MIX || part.mode != blender::MA_RAMP_ADD ||
          part.clamp || part.inputs[0] < 0) return false;
      const auto &factor = program->expressions[part.inputs[0]];
      return factor.operation == LPEOperation::CONSTANT && factor.value.x == 1.0f &&
             self(self, part.inputs[1]) && self(self, part.inputs[2]);
    };
    if (collect(collect, program->viewer_expression) && !terms.empty() && terms.size() <= 64) {
      film.set_lpe_linear_preview(std::move(terms), false, program->viewer_denoising);
      return;
    }
  }
  if (expression.operation == LPEOperation::PASS || expression.operation == LPEOperation::MASK)
    film.set_lpe_preview(ustring(expression.pass), {}, false, program->viewer_denoising);
  else if (expression.operation == LPEOperation::EDIT && expression.edit_terms.size() <= 64) {
    vector<Film::LPEPreview::Term> terms;
    for (const auto &term : expression.edit_terms)
      terms.push_back({ustring(term.pass), term.multiplier});
    if (!terms.empty()) film.set_lpe_linear_preview(std::move(terms), false, program->viewer_denoising);
    else film.set_lpe_preview(ustring(), [program](const OutputDriver::Tile &tile, std::vector<float> &pixels) {
      return evaluate_lpe_preview(*program, tile, pixels);
    }, false, program->viewer_denoising);
  }
  else {
    if (!film.lpe_preview_cache) film.lpe_preview_cache = create_lpe_preview_cache();
    film.set_lpe_preview(ustring(), [program, cache = film.lpe_preview_cache](const OutputDriver::Tile &tile, std::vector<float> &pixels) {
      return evaluate_lpe_preview(*program, tile, pixels, cache.get());
    }, false, program->viewer_denoising);
  }
}

bool update_lpe_preview_plan(Film &film, LPEGraph graph, const blender::bNodeTree &tree,
                             bool *complete)
{
  if (complete) *complete = false;
  const auto sampled = film.get_lpe_program();
  if (!sampled) return false;
  LPEGraph full = graph;
  full.program = std::make_shared<LPEProgram>(*graph.program);
  string error;
  if (rebind_lpe_program(full, *sampled, tree, error, true)) {
    if (complete) *complete = true;
    film.set_lpe_program(full.program);
    configure_lpe_preview(film, full.program, true);
    return true;
  }
  /* New Viewer requirements need real samples. Never accept an empty/partial
   * plan as if the requested path were already available. */
  return false;
}

void BlenderSync::sync_lpe(blender::ViewLayer &b_view_layer, LPEGraph graph)
{
  /* Render depsgraphs do not flush animation back to original IDs. Animated
   * parameters must come from the evaluated tree; static edits retain their
   * direct UI source so display-only updates need no scene reevaluation. */
  const blender::bNodeTree *parameter_tree = b_view_layer.lpe_node_tree;
  if (parameter_tree &&
      !(parameter_tree->adt && (parameter_tree->adt->action ||
                               !parameter_tree->adt->drivers.is_empty() ||
                               !parameter_tree->adt->nla_tracks.is_empty()))) {
    parameter_tree = blender::DEG_get_original(parameter_tree);
  }
  if (getenv("LPE_TRACE")) fprintf(stderr, "[LPE] sync preview=%d reset=%d program=%d edits=%zu\n", int(preview), int(scene->need_reset()), int(bool(scene->film->get_lpe_program())), graph.edits.size());
  Film &film = *scene->film;
  /* A Light Edit entry that does not resolve to any light is not an active edge:
   * drop it before comparing the sampled plan, otherwise merely wiring an empty
   * slot would restart the preview. Selecting a light in that slot later makes
   * the edge real and triggers the refresh. */
  {
    string ignored;
    std::erase_if(graph.edits, [&](const LPEEditRequest &edit) {
      return !edit.path.empty() && STREQ(edit.path.front()->owner_node().idname, "LPENodeLight") &&
             selected_objects(edit.path.front()->owner_node(), true, ignored).empty();
    });
  }
  if (preview && b_view_layer.lpe_node_tree && film.get_lpe_program() &&
      !scene->need_reset() && lpe_resident_keys(graph) == lpe_synced_resident_keys &&
      lpe_can_reuse_sampling(graph, lpe_synced_sampling_keys)) {
    LPEGraph candidate = graph;
    candidate.program = std::make_shared<LPEProgram>(*graph.program);
    if (update_lpe_preview_plan(film, std::move(candidate), *parameter_tree)) {
      if (getenv("LPE_TRACE")) fprintf(stderr, "[LPE] fast ok\n");
      return;
    }
  }
  if (getenv("LPE_TRACE")) fprintf(stderr, "[LPE] full rebuild\n");
  /* Adding/removing a World shadow may change whether native background NEE
   * is needed. Also refresh on removal and persistent view-layer switches. */
  if (!graph.shadows.empty() ||
      std::any_of(film.lpe_states.begin(), film.lpe_states.end(), [](const KernelLPEState &state) {
        return (state.flags & LPE_SHADOW) != 0;
      })) {
    scene->light_manager->tag_update(scene, LightManager::LIGHT_MODIFIED);
  }
  /* Persistent sessions may switch view layers or remove the LPE tree. */
  for (Object *object : scene->objects) {
    if (object->has_lpe_light_strength) {
      object->has_lpe_light_strength = false;
      scene->light_manager->tag_update(scene, LightManager::LIGHT_MODIFIED);
    }
  }
  film.lpe_states.clear();
  film.photon_lpe_states.clear();
  film.photon_lpe_pass_names.clear();
  film.photon_lpe_initial = 0;
  film.lpe_independent_photon = lpe_independent_photon_budget(*b_scene, preview, film.get_use_deep_output());
  film.lpe_selections.clear();
  film.lpe_volume_selections.clear();
  film.lpe_pass_names.clear();
  film.lpe_initial = 0;
  film.set_lpe_program(nullptr);
  configure_lpe_preview(film, nullptr, false);
  film.tag_modified();
  set<Pass *> old_raw;
  for (Pass *pass : scene->passes) {
    if (string_startswith(pass->get_name().string(), "__LPE_raw_") ||
        string_startswith(pass->get_name().string(), "__LPE_union_") ||
        string_startswith(pass->get_name().string(), "__LPE_mask_") ||
        string_startswith(pass->get_name().string(), "__LPE_edit_") ||
        string_startswith(pass->get_name().string(), "__LPE_shadow_"))
      old_raw.insert(pass);
  }
  scene->delete_nodes(old_raw);
  blender::bNodeTree *tree = b_view_layer.lpe_node_tree;
  if (!tree)
    return;
  if (graph.shadow_catcher) {
    /* Use the native split-path estimator and pass accessor, including its
     * normalization and transparent-film edges. No extra matcher states. */
    if (!Pass::find(scene->passes, "Shadow Catcher")) {
      Pass *pass = scene->create_node<Pass>();
      pass->set_type(PASS_SHADOW_CATCHER);
      pass->set_name(ustring("Shadow Catcher"));
      pass->set_mode(PassMode::NOISY);
    }
    film.set_use_approximate_shadow_catcher(false);
  }
  string error;
  /* Cycles merges phase closures for overlapping media. Disjoint volumes keep
   * unambiguous event provenance; validate selected bounds during device update. */
  const auto has_volume = [](const Object &object) {
    if (!object.get_geometry())
      return false;
    for (const Node *shader : object.get_geometry()->get_used_shaders()) {
      if (static_cast<const Shader *>(shader)->has_volume_connected)
        return true;
    }
    return false;
  };
  size_t volume_media = 0;
  if (const Shader *world = scene->background->get_shader(scene))
    volume_media += world->has_volume_connected ? 1 : 0;
  for (const Object *object : scene->objects)
    volume_media += has_volume(*object) ? 1 : 0;
  const auto validate_volume = [&](const bool mask, const LPESelection &selection) {
    if (mask) {
      error =
          "Volume influence Mask requires a separate extinction estimator; use Volume color "
          "output";
    }
    else {
      film.lpe_volume_selections.push_back(selection);
    }
  };
  const auto select_ids = [&](const blender::bNode &node,
                              const bool material,
                              const bool lights_only,
                              const bool mask = false) {
    const int offset = int(film.lpe_selections.size());
    set<const blender::ID *> selected;
    for (const blender::bNodeSocket *socket : node.input_sockets()) {
      if (material && socket->type == blender::SOCK_MATERIAL) {
        selected.insert(
            original_id(socket->default_value_typed<blender::bNodeSocketValueMaterial>()->value));
      }
    }
    if (!material) {
      for (const auto &[id, object] : selected_objects(node, lights_only, error)) {
        selected.insert(id);
      }
    }
    selected.erase(nullptr);
    if (material) {
      for (const auto &[id, shader] : shader_map.key_to_scene_data()) {
        if (selected.contains(original_id(id))) {
          if (shader->has_volume_connected)
            validate_volume(mask, {nullptr, shader});
          film.lpe_selections.push_back({nullptr, shader});
        }
      }
    }
    else {
      for (const auto &[key, object] : object_map.key_to_scene_data()) {
        if (selection_matches_object(selected, key, *object, lights_only)) {
          if (!lights_only && has_volume(*object))
            validate_volume(mask, {object, nullptr});
          film.lpe_selections.push_back({object, nullptr});
        }
      }
    }
    return std::pair(offset, int(film.lpe_selections.size()) - offset);
  };

  int matcher_phase = -1;
  const auto add_path = [&](const std::vector<const blender::bNodeSocket *> &path,
                            const string &pass_name,
                            Object *light_filter = nullptr,
                            const int shared_terminal = -1,
                            const bool create_pass = true) -> int {
    auto steps = compile_path_states(path, [&](const blender::bNodeSocket &socket,
                                               KernelLPEState &state, const bool source) {
      const blender::bNode &node = socket.owner_node();
      if (source) {
        auto &terminal = state;
        const bool world = STREQ(node.idname, "LPENodeWorld");
        const bool emission = STREQ(socket.identifier, "Emission");
        if (world) {
          const blender::World *b_world = view_layer.world_override ? view_layer.world_override :
                                                                      b_scene->world;
          terminal.light_offset = int(film.lpe_selections.size());
          for (const auto &[key, object] : object_map.key_to_scene_data()) {
            if (b_world && original_id(key.ob) == original_id(b_world))
              film.lpe_selections.push_back({object, nullptr});
          }
          terminal.light_count = int(film.lpe_selections.size()) - terminal.light_offset;
        }
        else {
          const auto [offset, count] = select_ids(
              node, emission && STREQ(node.idname, "LPENodeMaterial"), !emission);
          terminal.light_offset = offset;
          terminal.light_count = count;
          if (emission) {
            for (int i = offset; i < offset + count; i++) {
              const auto &selection = film.lpe_selections[i];
              if ((selection.object && has_volume(*selection.object)) ||
                  (selection.shader && selection.shader->has_volume_connected)) {
                error = "Emission source selection currently requires surface emitters";
              }
            }
          }
        }
        if (light_filter) {
          terminal.light_offset = int(film.lpe_selections.size());
          terminal.light_count = 1;
          film.lpe_selections.push_back({light_filter, nullptr});
        }
      }
      else {
        const bool material = STREQ(node.idname, "LPENodeMaterial");
        const auto [offset, count] = select_ids(node, material, false);
        state.selection_offset = offset;
        state.selection_count = count;
        if ((state.flags & LPE_MNEE_RECEIVER) &&
            !scene->integrator->get_use_photon_caustics()) {
          for (int i = offset; i < offset + count; i++) {
            const LPESelection &selection = film.lpe_selections[i];
            if (selection.object) {
              if (!selection.object->get_is_caustics_receiver()) {
                error = "Enable native Receive Caustics on the selected receiver objects";
              }
            }
            else if (selection.shader) {
              /* Material selectors bind shaders, not object pointers. Validate
               * every object using the selected receiver material for MNEE. */
              for (const Object *object : scene->objects) {
                if (!object->get_geometry()) {
                  continue;
                }
                const auto &shaders = object->get_geometry()->get_used_shaders();
                if (std::find(shaders.begin(), shaders.end(), selection.shader) != shaders.end() &&
                    !object->get_is_caustics_receiver())
                {
                  error =
                      "Enable native Receive Caustics on objects using the selected receiver "
                      "materials";
                }
              }
            }
          }
        }
      }
    }, scene->integrator->get_use_photon_caustics(), error);
    if (!error.empty() || steps.empty()) return -1;
    const KernelLPEState terminal = steps.back();
    const bool weighted = (terminal.flags & LPE_WEIGHTED) != 0;
    steps.pop_back();
    const size_t base = film.lpe_states.size();
    if (base + steps.size() + (shared_terminal < 0 ? 1 : 0) > LPE_MAX_STATES) {
      error = string("LPE ") + (matcher_phase == 1 ? "photon phase" : "native phase") +
              " exceeds the 64 matcher-state limit; reduce outputs or repeat iterations";
      return -1;
    }
    const int terminal_index = shared_terminal < 0 ? int(base + steps.size()) : shared_terminal;
    film.lpe_initial |= uint64_t(1) << (steps.empty() ? terminal_index : int(base));
    for (size_t i = 0; i < steps.size(); i++) {
      steps[i].next = uint64_t(1) << (i + 1 < steps.size() ? int(base + i + 1) : terminal_index);
      film.lpe_states.push_back(steps[i]);
      film.lpe_pass_names.push_back(ustring());
    }
    if (shared_terminal < 0) {
      film.lpe_states.push_back(terminal);
      film.lpe_pass_names.push_back(ustring(pass_name));
    }
    else if (weighted) {
      film.lpe_states[shared_terminal].flags |= LPE_WEIGHTED;
    }
    if (create_pass && !Pass::find(scene->passes, pass_name)) {
      Pass *pass = scene->create_node<Pass>();
      pass->set_type(PASS_LPE);
      pass->set_name(ustring(pass_name));
      pass->set_mode(PassMode::NOISY);
    }
    return terminal_index;
  };

  struct EditPath {
    std::vector<const blender::bNodeSocket *> path;
    string name;
    Object *light;
  };
  vector<EditPath> edit_paths;
  size_t edit_states = 0;
  std::map<unsigned int, int> light_instances;
  for (const auto &[key, object] : object_map.key_to_scene_data()) {
    const auto *id = original_id(key.ob);
    if (!id || id->id_type() != blender::ID_OB)
      continue;
    const auto *source = static_cast<const blender::Object *>(key.ob);
    if (source->type == blender::OB_LAMP && source->data)
      light_instances[original_id(source->data)->session_uid]++;
  }
  std::map<std::pair<std::vector<std::pair<int, string>>, uintptr_t>, string> edit_buckets;
  std::map<Object *, string> global_buckets;
  std::map<string, Object *> edit_objects;
  graph.program->sample_clamped = scene->integrator->get_sample_clamp_direct() != 0.0f ||
                                  scene->integrator->get_sample_clamp_indirect() != 0.0f;
  for (const LPEEditRequest &request : graph.edits) {
    if (!error.empty())
      break;
    if (request.path.empty() || !STREQ(request.path.front()->owner_node().idname, "LPENodeLight"))
    {
      error = "Light Edit currently requires an explicit Light source";
      break;
    }
    const blender::bNode &source = request.path.front()->owner_node();
    const auto selected = selected_objects(source, true, error);
    std::vector<std::pair<int, string>> path_key;
    for (const auto *socket : request.path)
      path_key.emplace_back(socket->owner_node().identifier, socket->identifier);
    for (const auto &[key, object] : object_map.key_to_scene_data()) {
      if (source.is_muted() || !selection_matches_object(selected, key, *object, true))
        continue;
      const auto *light_object = static_cast<const blender::Object *>(key.ob);
      if (light_object->type != blender::OB_LAMP || !light_object->data) {
        error = "Light Edit Light source must contain valid lights";
        break;
      }
      const auto &light = *reinterpret_cast<const blender::Light *>(light_object->data);
      LPELightParameters params;
      params.color = make_float3(light.r, light.g, light.b);
      params.power = light.energy;
      params.exposure = light.exposure;
      params.temperature = light.temperature;
      params.use_temperature = (light.mode & blender::LA_USE_TEMPERATURE) != 0;
      const auto bucket_key = std::make_pair(path_key, reinterpret_cast<uintptr_t>(object));
      auto bucket = edit_buckets.find(bucket_key);
      if (bucket == edit_buckets.end()) {
        const string name = "__LPE_edit_" + std::to_string(edit_buckets.size());
        edit_paths.push_back({request.path, name, object});
        edit_states += compile_budget_path(request.path).size();
        bucket = edit_buckets.emplace(bucket_key, name).first;
      }
      LPEEditTerm term;
      term.pass = bucket->second;
      if (request.manager_node) {
        auto global = global_buckets.find(object);
        if (global == global_buckets.end()) {
          const bool total_source = request.path.size() == 1 &&
                                    STREQ(request.path.front()->identifier, "Total Light");
          const string global_name = total_source ? term.pass :
              "__LPE_edit_global_" + std::to_string(global_buckets.size());
          if (!total_source) {
            edit_paths.push_back({{&source.output_socket(0)}, global_name, object});
            edit_states += compile_budget_path({&source.output_socket(0)}).size();
          }
          global = global_buckets.emplace(object, global_name).first;
        }
        term.global_pass = global->second;
      }
      term.source_name = object->name.string();
      if (!object->get_lightgroup().empty()) {
        /* The public name can be encoded to fit RenderPass or avoid collisions. */
        for (const Pass *pass : scene->passes) {
          if (pass->get_type() == PASS_COMBINED &&
              pass->get_lightgroup() == object->get_lightgroup()) {
            term.lightgroup_pass = pass->get_name().string();
            break;
          }
        }
      }
      term.source_parameters = params;
      term.sampling_parameters = params;
      term.source_object_uid = original_id(light_object)->session_uid;
      term.source_light_uid = original_id(&light)->session_uid;
      term.light_instance_count = light_instances[term.source_light_uid];
      const auto *parent = original_id(key.parent);
      term.source_instance = std::to_string(term.source_object_uid) + ":" +
                             std::to_string(parent ? parent->session_uid : 0);
      for (const int part : key.id)
        term.source_instance += ":" + std::to_string(part);
      edit_objects[term.source_instance] = object;
      term.manager_node = request.manager_node;
      term.manager_item = request.manager_item;
      term.operation_nodes = request.operation_nodes;
      term.scene_operation_nodes = request.scene_operation_nodes;
      term.output_node = request.output_node;
      term.output_item = request.output_item;
      graph.program->expressions[request.expression].edit_terms.push_back(std::move(term));
    }
  }
  /* Light edits split into one bucket per actual synchronized light instance.
   * Check their exact combined budget before creating any matcher states/passes. */
  if (error.empty()) {
    validate_lpe_state_budget(
        graph, b_view_layer.name, error, edit_states, film.lpe_independent_photon);
  }
  if (error.empty()) {
    if (!preview) {
      /* Offline renders sample the edited emission, including native passes,
       * light groups and the light-tree PDF. IPR keeps its immutable samples. */
      std::map<string, LPELightParameters> targets;
      for (const auto &expression : graph.program->expressions) {
        for (const auto &term : expression.edit_terms) {
          if (!term.manager_node || term.scene_operation_nodes.empty())
            continue;
          LPEEditTerm scene_term = term;
          scene_term.operation_nodes = term.scene_operation_nodes;
          scene_term.output_node = 0;
          LPELightParameters target;
          if (!evaluate_lpe_light_target(scene_term, *parameter_tree, target, error))
            break;
          const float3 strength = lpe_light_emission(target);
          if (!isfinite_safe(strength)) {
            error = "Non-finite LPE render light strength";
            break;
          }
          const auto [found, inserted] = targets.emplace(term.source_instance, target);
          if (!inserted && len_squared(lpe_light_emission(found->second) - strength) > 1e-12f) {
            error = "Light Edit outputs request conflicting settings for the same scene light";
            break;
          }
        }
        if (!error.empty()) break;
      }
      if (error.empty()) {
        for (const auto &[instance, target] : targets) {
          Object *object = edit_objects.at(instance);
          object->has_lpe_light_strength = true;
          object->lpe_light_strength = lpe_light_emission(target);
          scene->light_manager->tag_update(scene, LightManager::LIGHT_MODIFIED);
        }
        for (auto &expression : graph.program->expressions) {
          for (auto &term : expression.edit_terms) {
            const auto found = targets.find(term.source_instance);
            if (found != targets.end()) term.sampling_parameters = found->second;
          }
        }
      }
    }
    if (error.empty())
      update_lpe_edit_terms(*graph.program, *parameter_tree, error, preview);
  }

  for (int program = 0; program < (film.lpe_independent_photon ? 2 : 1); ++program) {
    matcher_phase = film.lpe_independent_photon ? program : -1;
    if (program == 1) {
      film.lpe_states.swap(film.photon_lpe_states);
      film.lpe_pass_names.swap(film.photon_lpe_pass_names);
      std::swap(film.lpe_initial, film.photon_lpe_initial);
    }
    for (size_t i = 0; i < graph.paths.size() && error.empty(); i++) {
      if (!lpe_path_in_phase(graph.paths[i], matcher_phase)) {
        /* Keep the raw pass addressable even when its contribution is zero in this phase. */
        continue;
      }
      add_path(graph.paths[i], "__LPE_raw_" + std::to_string(i));
    }
    for (size_t i = 0; i < graph.unions.size() && error.empty(); i++) {
      std::map<std::tuple<const blender::bNodeSocket *, bool, bool>, int> terminals;
      const string name = "__LPE_union_" + std::to_string(i);
      if (coalesce_light_sources(graph.unions[i])) {
        if (!lpe_path_in_phase(graph.unions[i].front(), matcher_phase))
          continue;
        const int terminal = add_path(graph.unions[i].front(), name);
        if (terminal < 0)
          break;
        std::set<Object *> lights;
        for (const auto &path : graph.unions[i]) {
          const auto &node = path.front()->owner_node();
          if (node.is_muted())
            continue;
          const auto [offset, count] = select_ids(node, false, true);
          for (int j = offset; j < offset + count; j++)
            lights.insert(film.lpe_selections[j].object);
        }
        auto &state = film.lpe_states[terminal];
        state.light_offset = int(film.lpe_selections.size());
        state.light_count = int(lights.size());
        for (Object *light : lights)
          film.lpe_selections.push_back({light, nullptr});
        continue;
      }
      for (const auto &path : graph.unions[i]) {
        if (!lpe_path_in_phase(path, matcher_phase))
          continue;
        const auto *first = path.front();
        const auto *source = STREQ(first->owner_node().idname, "LPENodeLight") ||
                                     STREQ(first->owner_node().idname, "LPENodeWorld") ||
                                     STREQ(first->identifier, "Emission") ?
                                 first :
                                 nullptr;
        auto [entry, inserted] = terminals.try_emplace(
            std::make_tuple(source, weighted_path(path), path_total_caustic_source(path)), -1);
        const bool create_pass = terminals.size() == 1 && inserted;
        entry->second = add_path(path, name, nullptr, entry->second, create_pass);
        if (entry->second >= 0)
          film.lpe_states[entry->second].flags |= LPE_UNION;
        if (!error.empty())
          break;
      }
    }
    for (const EditPath &edit : edit_paths) {
      if (!error.empty())
        break;
      if (lpe_path_in_phase(edit.path, matcher_phase))
        add_path(edit.path, edit.name, edit.light);
    }

    if (matcher_phase != 1 && !graph.shadows.empty()) {
      if (volume_media)
        error = "Selected caster shadows currently require a scene without participating media";
      for (const Object *object : scene->objects) {
        if (object->get_blocker_shadow_set() ||
            object->get_shadow_set_membership() != LIGHT_LINK_MASK_ALL)
          error = "Selected caster shadows do not yet support shadow linking";
        if (object->get_is_shadow_catcher() && object->get_geometry() &&
            object->get_geometry()->is_mesh())
          error =
              "Selected caster shadows must be evaluated independently of native Shadow Catcher "
              "objects";
        if (!scene->integrator->get_use_photon_caustics() &&
            (object->get_is_caustics_caster() || object->get_is_caustics_receiver()))
          error = "Selected caster shadows do not yet support native MNEE caustic objects";
        if (!object->get_geometry())
          continue;
        if (object->get_geometry()->is_hair() || object->get_geometry()->is_pointcloud())
          error = "Selected caster shadows currently support mesh occluders";
        for (const Node *node : object->get_geometry()->get_used_shaders()) {
          const auto *shader = static_cast<const Shader *>(node);
          if (shader->has_surface_transparent)
            error =
                "Selected caster shadows currently require opaque shadow surfaces; transparent "
                "shadow attribution is not supported";
          if (shader->has_surface_bssrdf)
            error = "Selected caster shadows do not yet support subsurface receivers";
        }
      }
    }
    for (size_t i = 0; matcher_phase != 1 && i < graph.shadows.size() && error.empty(); i++) {
      const auto &path = graph.shadows[i];
      KernelLPEState rule = {};
      rule.flags = LPE_SHADOW;
      rule.pass_offset = 0;
      const int caster_offset = int(film.lpe_selections.size());
      const bool material_casters = STREQ(path[1]->owner_node().idname, "LPENodeMaterial");
      std::set<std::pair<Object *, Shader *>> casters;
      for (size_t j = 1; j + 1 < path.size(); j++) {
        const auto [offset, count] = select_ids(path[j]->owner_node(), material_casters, false);
        for (int k = offset; k < offset + count; k++)
          casters.emplace(film.lpe_selections[k].object, film.lpe_selections[k].shader);
      }
      film.lpe_selections.resize(caster_offset);
      for (const auto &[object, shader] : casters)
        film.lpe_selections.push_back({object, shader});
      const int caster_count = int(casters.size());
      const bool material_receiver = STREQ(path.back()->owner_node().idname, "LPENodeMaterial");
      const auto [receiver_offset, receiver_count] = select_ids(
          path.back()->owner_node(), material_receiver, false);
      int light_offset, light_count;
      if (STREQ(path[0]->owner_node().idname, "LPENodeWorld")) {
        const blender::World *world = view_layer.world_override ? view_layer.world_override :
                                                                  b_scene->world;
        light_offset = int(film.lpe_selections.size());
        for (const auto &[key, object] : object_map.key_to_scene_data()) {
          if (world && original_id(key.ob) == original_id(world)) {
            const auto *background = dynamic_cast<const BackgroundLight *>(object->get_geometry());
            if (background && !background->get_use_mis()) {
              error = "World shadow selection requires Automatic or Manual World sampling";
            }
            film.lpe_selections.push_back({object, nullptr});
          }
        }
        light_count = int(film.lpe_selections.size()) - light_offset;
      }
      else {
        const auto selection = select_ids(path[0]->owner_node(), false, true);
        light_offset = selection.first;
        light_count = selection.second;
      }
      rule.selection_offset = caster_offset;
      rule.selection_count = caster_count;
      rule.selector = material_casters ? 2 : 1;
      if (material_receiver)
        rule.flags |= LPE_SHADOW_RECEIVER_MATERIAL;
      rule.pad[0] = receiver_offset;
      rule.pad[1] = receiver_count;
      rule.light_offset = light_offset;
      rule.light_count = light_count;
      if (film.lpe_states.size() >= LPE_MAX_STATES) {
        error = "Selected caster shadows exceed the 64 LPE state budget";
        break;
      }
      film.lpe_states.push_back(rule);
      const string base = "__LPE_shadow_" + std::to_string(i);
      film.lpe_pass_names.push_back(ustring(base + "_blocked"));
      for (const char *suffix : {"_reference", "_blocked"}) {
        Pass *pass = scene->create_node<Pass>();
        pass->set_type(PASS_LPE);
        pass->set_name(ustring(base + suffix));
        pass->set_mode(PassMode::NOISY);
      }
    }
    for (const auto &[full_path, expression_index] : graph.masks) {
      if (matcher_phase == 1)
        continue;
      if (!error.empty())
        break;
      auto states = compile_mask_states(
          full_path,
          [&](const blender::bNodeSocket &socket, KernelLPEState &state) {
            const auto [offset,
                        count] = select_ids(socket.owner_node(),
                                            STREQ(socket.owner_node().idname, "LPENodeMaterial"),
                                            false,
                                            true);
            state.selection_offset = offset;
            state.selection_count = count;
          },
          error);
      if (!error.empty())
        break;
      const size_t base = film.lpe_states.size();
      if (base + states.size() > LPE_MAX_STATES) {
        error = "LPE graph exceeds the 64 matcher-state limit";
        break;
      }
      const string &name = graph.program->expressions[expression_index].pass;
      film.lpe_initial |= uint64_t(1) << base;
      for (size_t i = 0; i < states.size(); i++) {
        if (i + 1 < states.size())
          states[i].next = uint64_t(1) << (base + i + 1);
        film.lpe_states.push_back(states[i]);
        film.lpe_pass_names.push_back(i + 1 == states.size() ? ustring(name) : ustring());
      }
      Pass *pass = scene->create_node<Pass>();
      pass->set_type(PASS_AOV_VALUE);
      pass->set_name(ustring(name));
      pass->set_mode(PassMode::NOISY);
    }
    if (program == 1) {
      film.lpe_states.swap(film.photon_lpe_states);
      film.lpe_pass_names.swap(film.photon_lpe_pass_names);
      std::swap(film.lpe_initial, film.photon_lpe_initial);
    }
  }
  if (!error.empty()) {
    film.photon_lpe_states.clear();
    film.photon_lpe_pass_names.clear();
    film.photon_lpe_initial = 0;
    film.lpe_states.clear();
    film.lpe_selections.clear();
    film.lpe_pass_names.clear();
    film.lpe_initial = 0;
    progress.set_error("LPE: " + error);
    return;
  }
  configure_lpe_preview(film, graph.program, preview);
  if (!preview) {
    graph.program->display_evaluate = [cache = create_lpe_preview_cache()](
        const LPEProgram &program, const int expression, const OutputDriver::Tile &tile,
        std::vector<float> &pixels) {
      return evaluate_lpe_preview(program, tile, pixels, cache.get(), expression);
    };
  }
  record_sampling_bindings(graph);
  lpe_synced_edits = stateful_edit_count(graph);
  lpe_synced_resident_keys = lpe_resident_keys(graph);
  lpe_synced_sampling_keys = lpe_sampling_keys(graph);
  film.set_lpe_program(std::move(graph.program));
}

void BlenderSync::sync_deep_lpe(const std::vector<string> &passes)
{
  Film &film = *scene->film;
  film.deep_lpe_passes.clear();
  film.deep_lpe_outputs.clear();
  film.deep_native_sources.clear();
  if (preview || !film.get_use_deep_output()) return;
  const auto program = film.get_lpe_program();
  for (const string &pass_name : passes) {
    if (pass_name == "Image" || pass_name == "Alpha" || pass_name == "Noisy Image" ||
        pass_name == "Combined") continue;
    if (!string_startswith(pass_name, "LPE_")) {
      if (std::any_of(film.deep_lpe_outputs.begin(), film.deep_lpe_outputs.end(),
                      [&](const auto &o) { return o.name == pass_name; })) continue;
      const Pass *pass = Pass::find(scene->passes, pass_name);
      const auto supported = [](PassType type) {
        switch (type) {
          case PASS_COMBINED: case PASS_EMISSION: case PASS_BACKGROUND: case PASS_CAUSTICS:
          case PASS_DIFFUSE: case PASS_DIFFUSE_DIRECT: case PASS_DIFFUSE_INDIRECT:
          case PASS_GLOSSY: case PASS_GLOSSY_DIRECT: case PASS_GLOSSY_INDIRECT:
          case PASS_TRANSMISSION: case PASS_TRANSMISSION_DIRECT: case PASS_TRANSMISSION_INDIRECT:
          case PASS_SUBSURFACE: case PASS_SUBSURFACE_DIRECT: case PASS_SUBSURFACE_INDIRECT:
          case PASS_VOLUME: case PASS_VOLUME_DIRECT: case PASS_VOLUME_INDIRECT: case PASS_AO:
            return true;
          default: return false;
        }
      };
      if (!pass || !supported(pass->get_type())) {
        progress.set_error("Deep: unsupported data or derived pass " + pass_name +
            ". Connect a lighting pass; Z, normals, vectors and Cryptomatte are unsupported.");
        return;
      }
      const auto info = pass->get_info();
      Film::DeepLPEOutput output;
      output.name = pass_name;
      output.divisor = info.divide_type == PASS_DIFFUSE_COLOR ? 0 :
                       info.divide_type == PASS_SUBSURFACE_COLOR ? 1 :
                       info.divide_type == PASS_GLOSSY_COLOR ? 2 :
                       info.divide_type == PASS_TRANSMISSION_COLOR ? 3 : -1;
      output.use_exposure = info.use_exposure;
      const auto add_source = [&](PassType type) {
        for (const auto &source : film.deep_native_sources)
          if (source.type == type && source.lightgroup == pass->get_lightgroup()) {
            output.terms.push_back({source.channel, one_float3()}); return;
          }
        const int channel = int(film.deep_lpe_passes.size());
        film.deep_lpe_passes.push_back("__deep_native_" + std::to_string(channel));
        film.deep_native_sources.push_back({channel, type, pass->get_lightgroup()});
        output.terms.push_back({channel, one_float3()});
      };
      if (info.direct_type != PASS_NONE) {
        add_source(info.direct_type); add_source(info.indirect_type);
      }
      else add_source(pass->get_type());
      film.deep_lpe_outputs.push_back(std::move(output));
      continue;
    }
    if (!program) { progress.set_error("Deep LPE: no compiled LPE program"); return; }
    if (std::any_of(film.deep_lpe_outputs.begin(), film.deep_lpe_outputs.end(),
                    [&](const auto &o) { return o.name == pass_name; })) continue;
    const auto found = std::find_if(program->outputs.begin(), program->outputs.end(),
                                   [&](const auto &o) { return o.name == pass_name; });
    if (found == program->outputs.end()) {
      progress.set_error("Deep LPE: output not found: " + pass_name); return;
    }
    std::map<string, float3> terms;
    const auto collect = [&](auto &&self, int index, float3 weight, int depth) -> bool {
      if (depth > 256 || index < 0 || index >= int(program->expressions.size())) return false;
      const auto &part = program->expressions[index];
      if (part.clamp) return false;
      if (part.operation == LPEOperation::PASS && !part.scalar) {
        terms[part.pass] += weight; return true;
      }
      if (part.operation == LPEOperation::EDIT) {
        for (const auto &term : part.edit_terms) terms[term.pass] += weight * term.multiplier;
        return true;
      }
      if (part.operation == LPEOperation::CONSTANT) return is_zero(part.value);
      if (part.operation == LPEOperation::MIX && part.inputs[0] >= 0) {
        const auto &f = program->expressions[part.inputs[0]];
        if (f.operation != LPEOperation::CONSTANT) return false;
        const float factor = part.clamp_factor ? saturatef(f.value.x) : f.value.x;
        if (part.mode == blender::MA_RAMP_ADD)
          return self(self, part.inputs[1], weight, depth + 1) &&
                 self(self, part.inputs[2], weight * factor, depth + 1);
        if (part.mode == blender::MA_RAMP_BLEND)
          return self(self, part.inputs[1], weight * (1.0f - factor), depth + 1) &&
                 self(self, part.inputs[2], weight * factor, depth + 1);
      }
      return false;
    };
    if (!collect(collect, found->expression, one_float3(), 0)) {
      progress.set_error("Deep LPE: " + pass_name +
          " requires linear radiance selection or light editing; masks and nonlinear operations are unsupported");
      return;
    }
    Film::DeepLPEOutput output;
    output.name = pass_name;
    for (const auto &[name, weight] : terms) {
      if (is_zero(weight)) continue;
      /* Only terminal illumination passes have a Physical capture hook. */
      if (std::none_of(film.lpe_pass_names.begin(), film.lpe_pass_names.end(),
                       [&](const auto &p) { return p.string() == name; })) {
        progress.set_error("Deep LPE: unsupported source pass " + name); return;
      }
      auto p = std::find(film.deep_lpe_passes.begin(), film.deep_lpe_passes.end(), name);
      const int channel = int(p - film.deep_lpe_passes.begin());
      if (p == film.deep_lpe_passes.end()) film.deep_lpe_passes.push_back(name);
      output.terms.push_back({channel, weight});
    }
    film.deep_lpe_outputs.push_back(std::move(output));
  }
  if (!film.deep_lpe_outputs.empty()) film.set_deep_simple_mode(false);
}

CCL_NAMESPACE_END
