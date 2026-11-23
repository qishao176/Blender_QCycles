/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: Apache-2.0 */

#include "blender/lpe.h"

#include <set>
#include "util/scoped_defer.h"
#include "util/tbb.h"

#include "BKE_node.hh"
#include "BKE_node_runtime.hh"
#include "BKE_node_tree_zones.hh"
#include "DNA_material_types.h"
#include "IMB_colormanagement.hh"

#include "kernel/svm/color_util.h"
#include "kernel/svm/math_util.h"

CCL_NAMESPACE_BEGIN

namespace {

using Socket = blender::bNodeSocket;
using Node = blender::bNode;

bool is_geometry_selection(const Node &node)
{
  return STREQ(node.idname, "LPENodeObject") || STREQ(node.idname, "LPENodeMaterial");
}

const Socket *source(const Socket &input)
{
  for (const blender::bNodeLink *link : input.directly_linked_links()) {
    if (!(link->flag & blender::NODE_LINK_MUTED)) {
      return link->fromsock;
    }
  }
  return nullptr;
}

const Socket *input(const Node &node, const string &identifier)
{
  for (const Socket *socket : node.input_sockets()) {
    if (socket->identifier == identifier) {
      return socket;
    }
  }
  return nullptr;
}

string edit_preview_key(const Node &node, const Socket &socket)
{
  return "E:" + std::to_string(node.identifier) + ":" + socket.identifier;
}

float3 evaluate_operation(const LPEExpression &expression,
                          const float3 a,
                          const float3 b,
                          const float3 c)
{
  float3 result = expression.value;
  if (expression.operation == LPEOperation::SHADOW_RATIO) {
    result = saturate(make_float3(a.x > 0 ? 1 - b.x / a.x : 1,
                         a.y > 0 ? 1 - b.y / a.y : 1,
                         a.z > 0 ? 1 - b.z / a.z : 1));
  }
  else if (expression.operation == LPEOperation::SHADOW_MASK) {
    result = make_float3(1.0f - saturatef(average(a)));
  }
  else if (expression.operation == LPEOperation::TO_FLOAT) {
    const float color[3] = {a.x, a.y, a.z};
    result = make_float3(blender::IMB_colormanagement_get_luminance(color));
  }
  else if (expression.operation == LPEOperation::MATH) {
    result = make_float3(svm_math(NodeMathType(expression.mode), a.x, b.x, c.x));
  }
  else if (expression.operation == LPEOperation::MIX) {
    result = svm_mix(
        NodeMix(expression.mode), expression.clamp_factor ? saturatef(a.x) : a.x, b, c);
  }
  return expression.clamp ? saturate(result) : result;
}

struct Symbol {
  LPEExpression expression;
  std::shared_ptr<Symbol> inputs[3];
  bool is_path = false;
  std::vector<const Socket *> path;
  const Node *mask = nullptr;
  int manager_node = 0;
  string manager_item;
  std::vector<int> operation_nodes;
  std::vector<int> scene_operation_nodes;
  int output_node = 0;
  string output_item;
  /* Carry the next path frontier separately from the explicitly accumulated image.
   * Only repeat-input bindings have history; ordinary numerical values do not. */
  std::shared_ptr<Symbol> frontier;
  std::shared_ptr<Symbol> history;
  std::vector<std::shared_ptr<Symbol>> alternatives;
};
using Value = std::shared_ptr<Symbol>;

struct Context {
  const blender::bke::bNodeTreeZone *zone = nullptr;
  Context *parent = nullptr;
  std::map<const Socket *, Value> values;
  std::set<const Socket *> active;
};

class Compiler {
 public:
  const blender::bNodeTree &tree;
  const blender::bke::bNodeTreeZones *zones;
  string &error;
  LPEGraph graph;
  int symbol_count = 0;
  int evaluation_depth = 0;
  std::map<const Symbol *, int> lowered;
  std::map<std::vector<std::pair<int, string>>, int> lowered_paths;
  /* Output-local clones must outlive the raw-pointer lowering cache. */
  std::vector<Value> output_roots;

  Compiler(const blender::bNodeTree &tree, string &error)
      : tree(tree), zones(tree.zones()), error(error)
  {
  }

  Value make_value()
  {
    if (++symbol_count > 4096 && error.empty()) {
      error = "LPE expression expansion exceeds 4096 values; reduce repeat iterations";
    }
    return std::make_shared<Symbol>();
  }

  Value constant(const float3 value, const bool scalar = false)
  {
    Value result = make_value();
    result->expression.value = value;
    result->expression.scalar = scalar;
    return result;
  }

  bool is_constant(const Value &value) const
  {
    return !value->history && !value->is_path && value->alternatives.empty() &&
           value->expression.operation == LPEOperation::CONSTANT;
  }

  Value path_frontier(const Value &value) const
  {
    return value->frontier ? value->frontier :
           (value->is_path || !value->alternatives.empty()) ? value : Value();
  }

  Value merge_paths(const std::vector<Value> &inputs)
  {
    std::vector<Value> paths;
    std::vector<Value> frontier;
    bool has_history = false;
    std::map<std::vector<std::pair<int, string>>, Value> seen;
    const auto append = [&](auto &&self, const Value &value) -> void {
      if (!value) return;
      if (!value->alternatives.empty()) {
        for (const Value &part : value->alternatives)
          self(self, part);
        return;
      }
      if (is_constant(value) && is_zero(value->expression.value))
        return;
      if (!value->is_path || value->path.empty()) {
        error = "Merge Paths requires path selections, not arbitrary image operations";
        return;
      }
      std::vector<std::pair<int, string>> key;
      for (const Socket *socket : value->path)
        key.emplace_back(socket->owner_node().identifier, socket->identifier);
      const auto [found, inserted] = seen.emplace(key, value);
      if (inserted)
        paths.push_back(value);
      else if (found->second->manager_node != value->manager_node ||
               found->second->manager_item != value->manager_item ||
               found->second->operation_nodes != value->operation_nodes) {
        error = "Merge Paths received the same path with different light edits; use one edit scope";
      }
    };
    for (const Value &value : inputs) {
      if (!value) continue;
      has_history |= bool(value->history);
      append(append, value->history ? value->history : value);
      if (!value->history) frontier.push_back(value);
    }
    if (paths.empty())
      return constant(make_float3(0.0f));
    if (paths.size() == 1) {
      if (!has_history) return paths.front();
      Value result = make_value();
      *result = *paths.front();
      result->frontier = frontier.empty() ? paths.front() : merge_paths(frontier);
      return result;
    }
    const bool shadow = std::any_of(paths.begin(), paths.end(), [](const Value &part) {
      return std::any_of(part->path.begin(), part->path.end(), [](const Socket *socket) {
        return STREQ(socket->identifier, "Shadow");
      });
    });
    if (shadow) {
      /* Merge caster selections before evaluating their shared shadow ratio.
       * Adding already normalized shadow images would double-count overlap. */
      const Socket *source = paths.front()->path.front();
      const Socket *receiver = STREQ(paths.front()->path.back()->identifier, "Shadow") ?
                                   nullptr : paths.front()->path.back();
      std::vector<const Socket *> casters;
      for (const Value &part : paths) {
        const auto &path = part->path;
        const Socket *part_receiver = STREQ(path.back()->identifier, "Shadow") ? nullptr : path.back();
        const size_t end = path.size() - (part_receiver ? 1 : 0);
        if (path.front() != source || part_receiver != receiver || end <= 1 ||
            part->manager_node || !part->operation_nodes.empty() || has_history ||
            !std::all_of(path.begin() + 1, path.begin() + end, [](const Socket *socket) {
              return is_geometry_selection(socket->owner_node()) &&
                     STREQ(socket->identifier, "Shadow");
            })) {
          error = "Merge shadow casters from the same Light or World output and use one shared receiver; "
                  "do not mix shadow factors with lighting or edited paths";
          return {};
        }
        casters.insert(casters.end(), path.begin() + 1, path.begin() + end);
      }
      Value result = make_value();
      *result = *paths.front();
      result->path = {source};
      result->path.insert(result->path.end(), casters.begin(), casters.end());
      if (receiver) result->path.push_back(receiver);
      return result;
    }
    const bool edited = std::any_of(paths.begin(), paths.end(), [](const Value &path) {
      return path->manager_node || !path->operation_nodes.empty();
    });
    if (edited) {
      for (size_t i = 0; i < paths.size(); i++)
        for (size_t j = i + 1; j < paths.size(); j++)
          if (!lpe_paths_disjoint(paths[i]->path, paths[j]->path)) {
            error = "Merge Paths after light edits requires disjoint light selections or source channels; "
                    "overlapping edited paths need a shared union estimator";
          }
    }
    Value result = make_value();
    result->alternatives = std::move(paths);
    if (has_history && !frontier.empty()) result->frontier = merge_paths(frontier);
    return result;
  }

  Value operation(const LPEOperation op,
                  Value a,
                  Value b = {},
                  Value c = {},
                  const int mode = 0,
                  const bool clamp = false,
                  const bool clamp_factor = true)
  {
    if (a && a->history)
      a = a->history;
    if (b && b->history)
      b = b->history;
    if (c && c->history)
      c = c->history;
    Value result = make_value();
    result->expression.operation = op;
    result->expression.mode = mode;
    result->expression.clamp = clamp;
    result->expression.clamp_factor = clamp_factor;
    result->expression.scalar = op != LPEOperation::MIX;
    result->inputs[0] = a;
    result->inputs[1] = b;
    result->inputs[2] = c;
    if (is_constant(a) && (!b || is_constant(b)) && (!c || is_constant(c))) {
      result->expression.value = evaluate_operation(result->expression,
                                                    a->expression.value,
                                                    b ? b->expression.value : make_float3(0.0f),
                                                    c ? c->expression.value : make_float3(0.0f));
      result->expression.operation = LPEOperation::CONSTANT;
      result->inputs[0].reset();
      result->inputs[1].reset();
      result->inputs[2].reset();
    }

    return result;
  }

  Value as_float(Value value)
  {
    return value->expression.scalar ? value : operation(LPEOperation::TO_FLOAT, value);
  }

  Value default_value(const Socket &socket)
  {
    if (socket.type == blender::SOCK_RGBA) {
      const float *v = socket.default_value_typed<blender::bNodeSocketValueRGBA>()->value;
      return constant(make_float3(v[0], v[1], v[2]));
    }
    if (socket.type == blender::SOCK_FLOAT) {
      return constant(
          make_float3(socket.default_value_typed<blender::bNodeSocketValueFloat>()->value), true);
    }
    if (socket.type == blender::SOCK_INT) {
      return constant(
          make_float3(float(socket.default_value_typed<blender::bNodeSocketValueInt>()->value)),
          true);
    }
    error = "Unsupported repeat item or numerical socket type";
    return constant(make_float3(0.0f));
  }

  Value read(const Socket &socket, Context &context)
  {
    Value value = source(socket) ? evaluate(*source(socket), context) : default_value(socket);
    return ELEM(socket.type, blender::SOCK_FLOAT, blender::SOCK_INT) ? as_float(value) : value;
  }

  Value extend(Value upstream, const Socket &socket)
  {
    if (upstream->history)
      upstream = upstream->frontier;
    if (!upstream->alternatives.empty()) {
      std::vector<Value> next;
      for (const Value &part : upstream->alternatives)
        next.push_back(extend(part, socket));
      return merge_paths(next);
    }
    if (upstream->is_path) {
      Value result = make_value();
      result->is_path = true;
      result->path = upstream->path;
      result->manager_node = upstream->manager_node;
      result->manager_item = upstream->manager_item;
      result->operation_nodes = upstream->operation_nodes;
      result->scene_operation_nodes = upstream->scene_operation_nodes;
      result->output_node = upstream->output_node;
      result->output_item = upstream->output_item;
      result->path.push_back(&socket);
      return result;
    }
    /* Within an iteration, explicit addition keeps every authored term. At the
     * repeat
     * boundary only the new frontier is fed back into path selection. */

    if (is_constant(upstream) && is_zero(upstream->expression.value)) {
      return upstream;
    }
    error =
        "An ordered Light Path needs a selection or Merge Paths; "
        "nonlinear image operations cannot generate scattering paths";
    return constant(make_float3(0.0f));
  }

  Value as_mask(const Value &value)
  {
    if (value->history)
      return as_mask(value->frontier);
    if (value->expression.operation == LPEOperation::MASK)
      return value;
    if (value->is_path && !value->path.empty()) {
      Value result = make_value();
      result->path = value->path;
      result->mask = &value->path.back()->owner_node();
      result->expression.operation = LPEOperation::MASK;
      result->expression.scalar = true;
      return result;
    }

    if (is_constant(value) && is_zero(value->expression.value)) {
      return constant(make_float3(0.0f), true);
    }
    error = "Mask requires an object/material light path selection";
    return constant(make_float3(0.0f), true);
  }

  Value edit_scope(const Value &value,
                   const int manager,
                   const string &item,
                   const int operation_node = 0)
  {
    if (!value->alternatives.empty()) {
      error = "Light Edit on merged paths is not supported yet; edit separate path outputs";
      return constant(make_float3(0.0f));
    }
    if (value->history)
      return edit_scope(value->frontier, manager, item, operation_node);
    if (value->is_path) {
      Value result = make_value();
      *result = *value;
      if (manager) {
        if (value->manager_node && value->manager_node != manager) {
          error = "Nested Light Edit managers are not yet supported";
        }
        result->manager_node = manager;
        result->manager_item = item;
        result->scene_operation_nodes = result->operation_nodes;
      }
      if (operation_node) {
        result->operation_nodes.push_back(operation_node);
      }
      return result;
    }

    if (is_constant(value) && is_zero(value->expression.value))
      return value;
    error = "Light Edit requires light path selections";
    return constant(make_float3(0.0f));
  }

  void repeat(const Node &output, Context &parent)
  {
    const auto *zone = zones ? zones->get_zone_by_node(output.identifier) : nullptr;
    if (!zone || !zone->input_node()) {
      error = "Repeat zone has no valid paired input";
      return;
    }
    const Node &entry = *zone->input_node();
    Value count = read(entry.input_socket(0), parent);
    if (!is_constant(count) || !std::isfinite(count->expression.value.x)) {
      error = "Repeat iterations must be a finite uniform value, not a rendered image";
      return;
    }
    const float iterations_value = count->expression.value.x;
    if (iterations_value > 1024) {
      error = "LPE repeat expansion is limited to 1024 iterations";
      return;
    }
    const int iterations = iterations_value <= 0 ? 0 : int(iterations_value);
    const auto &storage = *static_cast<const blender::NodeGeometryRepeatOutput *>(output.storage);
    std::vector<Value> state;
    std::vector<Value> frontiers;
    std::vector<const Socket *> entry_outputs;
    std::vector<const Socket *> exit_outputs;
    for (const blender::NodeRepeatItem &item : blender::Span(storage.items, storage.items_num)) {
      const string identifier = "Item_" + std::to_string(item.identifier);
      const Socket *initial = input(entry, identifier);
      if (!initial || !input(output, identifier)) {
        error = "Invalid repeat item declaration";
        return;
      }
      state.push_back(read(*initial, parent));
      frontiers.push_back(path_frontier(state.back()));
      for (const Socket *socket : entry.output_sockets()) {
        if (socket->identifier == identifier)
          entry_outputs.push_back(socket);
      }
      for (const Socket *socket : output.output_sockets()) {
        if (socket->identifier == identifier)
          exit_outputs.push_back(socket);
      }
    }
    if (entry_outputs.size() != state.size() || exit_outputs.size() != state.size()) {
      error = "Invalid repeat zone output declaration";
      return;
    }
    for (int iteration = 0; iteration < iterations && error.empty(); iteration++) {
      Context body;
      body.zone = zone;
      body.parent = &parent;
      body.values[&entry.output_socket(0)] = constant(make_float3(float(iteration)), true);
      for (size_t i = 0; i < state.size(); i++) {
        Value binding = state[i];
        if (frontiers[i]) {
          binding = make_value();
          binding->frontier = frontiers[i];
          binding->history = iteration == 0 ? constant(make_float3(0.0f)) : state[i];
        }
        body.values[entry_outputs[i]] = binding;
      }
      std::vector<Value> next;
      for (size_t i = 0; i < exit_outputs.size(); i++) {
        Value value = read(*input(output, exit_outputs[i]->identifier), body);
        frontiers[i] = path_frontier(value);
        /* A direct pass-through carries the initial value unchanged. Only image
         *
         * operations consume the (initially empty) accumulated history. */
        next.push_back(value->history ? state[i] : value);
      }
      state = std::move(next);
    }
    for (size_t i = 0; i < state.size(); i++)
      parent.values[exit_outputs[i]] = state[i];
  }

  bool within_traversal_limit(const int depth)
  {
    /* Reroutes do not allocate symbols; Repeat can create new contexts and expressions. */
    if (error.empty() && depth >= 256) {
      error = "LPE graph exceeds the maximum traversal depth of 256";
    }
    return error.empty();
  }

  Value evaluate(const Socket &socket, Context &context)
  {
    if (!within_traversal_limit(evaluation_depth))
      return constant(make_float3(0.0f));
    ++evaluation_depth;
    SCOPED_DEFER(--evaluation_depth);
    const auto *zone = zones ? zones->get_zone_by_socket(socket) : nullptr;
    if (context.zone != zone) {
      if (context.parent)
        return evaluate(socket, *context.parent);
      error = "A repeat body cannot be read outside its paired output";
      return constant(make_float3(0.0f));
    }
    if (const auto found = context.values.find(&socket); found != context.values.end())
      return found->second;
    if (!context.active.insert(&socket).second) {
      error = "Cycle in LPE expression graph";
      return constant(make_float3(0.0f));
    }
    const Node &node = socket.owner_node();
    const string type = node.idname;
    Value result;
    if (node.is_reroute())
      result = read(node.input_socket(0), context);
    else if (type == "GeometryNodeRepeatOutput") {
      repeat(node, context);
      const auto found = context.values.find(&socket);
      if (found != context.values.end())
        result = found->second;
    }
    else if (type == "LPENodeShadowCatcher") {
      graph.shadow_catcher = true;
      result = make_value();
      result->expression.operation = LPEOperation::PASS;
      result->expression.pass = "Shadow Catcher";
      if (socket.identifier == string("Shadow Mask"))
        result = operation(LPEOperation::SHADOW_MASK, result);
    }
    else if (type == "LPENodeMask") {
      result = as_mask(read(node.input_socket(0), context));
    }
    else if (type == "LPENodeMerge") {
      std::vector<Value> parts;
      for (const auto *link : node.input_socket(0).directly_linked_links()) {
        if (!(link->flag & blender::NODE_LINK_MUTED))
          parts.push_back(evaluate(*link->fromsock, context));
      }
      result = node.is_muted() ? (parts.empty() ? constant(make_float3(0.0f)) : parts.front()) :
                                merge_paths(parts);
    }
    else if (type == "LPENodeSceneEdit") {
      const Socket *paired = input(node, socket.identifier);
      if (paired && source(*paired)) {
        result = read(*paired, context);
        if (!node.is_muted()) result = edit_scope(result, node.identifier, socket.identifier);
      }
      else {
        /* An empty Light Edit slot is simply not part of the graph. Failing here
         * would abort the whole viewport preview for a dangling entry. */
        result = Value();
      }
    }
    else if (type == "LPENodeLightOverride" || type == "LPENodeLightOffset" ||
             type == "LPENodeLightHSL" || type == "LPENodeLightBrightnessContrast") {
      /* Keep the source bucket when bypassing an operation, so muting and Apply
       * change only the edit plan rather than removing sampled requirements. */
      result = edit_scope(read(node.input_socket(0), context), 0, "", node.identifier);
    }
    else if (type == "LPENodeLight" || type == "LPENodeWorld") {
      result = make_value();
      result->is_path = true;
      result->path = {&socket};
    }
    else if (type == "LPENodeObject" || type == "LPENodeMaterial") {
      const Socket *path = input(node, "Light Path");
      Value upstream;
      if (path && source(*path))
        upstream = read(*path, context);
      else {
        upstream = make_value();
        upstream->is_path = true;
      }
      result = node.is_muted() ? upstream : extend(upstream, socket);
      if (STREQ(socket.identifier, "Mask")) {
        result = node.is_muted() ? constant(make_float3(0.0f), true) : as_mask(result);
      }
    }
    else if (type == "ShaderNodeRGB" || type == "ShaderNodeValue")
      result = default_value(socket);
    else if (type == "ShaderNodeMix" || type == "ShaderNodeMixRGB") {
      error = "Mix Color is no longer supported in LPE; replace it with Merge Paths";
    }
    else if (type == "ShaderNodeMath") {
      result = node.is_muted() ? read(node.input_socket(0), context) :
          operation(LPEOperation::MATH, read(node.input_socket(0), context),
                    read(node.input_socket(1), context), read(node.input_socket(2), context),
                    node.custom1, node.custom2 & blender::SHD_MATH_CLAMP);
    }
    else
      error = "Unsupported node in LPE expression: " + type;
    if (!result)
      result = constant(make_float3(0.0f));
    context.active.erase(&socket);
    context.values[&socket] = result;
    return result;
  }

  Value output_scope(const Value &value, const int node, const string &item,
                     std::map<const Symbol *, Value> &scoped_values, const int depth = 0)
  {
    if (const auto found = scoped_values.find(value.get()); found != scoped_values.end())
      return found->second;
    if (!within_traversal_limit(depth))
      return value;
    if (value->history) {
      Value result = output_scope(value->history, node, item, scoped_values, depth + 1);
      scoped_values.emplace(value.get(), result);
      return result;
    }
    if (value->is_path && (value->manager_node || !value->operation_nodes.empty())) {
      Value result = make_value();
      *result = *value;
      result->output_node = node;
      result->output_item = item;
      scoped_values.emplace(value.get(), result);
      return result;
    }
    Value result;
    for (size_t i = 0; i < value->alternatives.size(); i++) {
      Value scoped = output_scope(value->alternatives[i], node, item, scoped_values, depth + 1);
      if (scoped != value->alternatives[i]) {
        if (!result) {
          result = make_value();
          *result = *value;
        }
        result->alternatives[i] = scoped;
      }
    }
    for (int i = 0; i < 3; i++) {
      if (!value->inputs[i])
        continue;
      Value scoped = output_scope(value->inputs[i], node, item, scoped_values, depth + 1);
      if (scoped != value->inputs[i]) {
        if (!result) {
          result = make_value();
          *result = *value;
        }
        result->inputs[i] = scoped;
      }
    }
    if (!result) result = value;
    scoped_values.emplace(value.get(), result);
    return result;
  }

  /* Light Edit acts as an output of its own: a connected path is edited even
   * when no Output node consumes it. The offline render bakes the overrides into
   * the lights, so requiring an Output would only duplicate the light group. */
  void lower_edit_roots(Context &context)
  {
    for (const Node *node : tree.all_nodes()) {
      if (!STREQ(node->idname, "LPENodeSceneEdit") || node->is_muted())
        continue;
      for (const Socket *socket : node->output_sockets()) {
        const Socket *paired = input(*node, socket->identifier);
        if (!paired || !source(*paired))
          continue;
        /* Final output scopes already apply the edit. An IPR observer also needs the
         * socket's own expression, before any downstream output operations. Both
         * expressions bind to the same per-light sampling bucket. */
        const bool collected = std::any_of(
            graph.edits.begin(), graph.edits.end(), [&](const LPEEditRequest &edit) {
              return edit.manager_node == node->identifier &&
                     edit.manager_item == socket->identifier;
            });
        if (collected && !graph.program->viewport)
          continue;
        const Value value = evaluate(*socket, context);
        if (!value || !error.empty())
          return;
        output_roots.push_back(value);
        const int index = lower(value);
        if (!error.empty())
          return;
        if (graph.program->viewport)
          graph.program->edit_previews.emplace(edit_preview_key(*node, *socket), index);
      }
    }
  }

  int lower(const Value &value, const int depth = 0)
  {
    if (!value)
      return -1;
    if (const auto found = lowered.find(value.get()); found != lowered.end())
      return found->second;
    if (!within_traversal_limit(depth))
      return -1;
    if (value->history)
      return lower(value->history, depth + 1);
    const auto validate_caustics = [&](const std::vector<const Socket *> &path) {
      int caster = -1, last_caster = -1;
      for (size_t i = 0; i < path.size(); ++i) {
        if (STREQ(path[i]->identifier, "Caustics")) {
          if (!is_geometry_selection(path[i]->owner_node()) ||
              (last_caster >= 0 && int(i) != last_caster + 1))
          {
            error = "Connect consecutive Object or Material Caustics outputs before the receiving path";
            return;
          }
          if (caster < 0) {
            caster = int(i);
          }
          last_caster = int(i);
        }
      }
      if (caster >= 0) {
        if (value->mask) {
          error = "Caustics is a lighting contribution, not a coverage Mask";
          return;
        }
        if (!is_geometry_selection(path[caster]->owner_node()) || caster > 1 ||
            (caster == 1 && !STREQ(path[0]->owner_node().idname, "LPENodeLight") &&
             !STREQ(path[0]->owner_node().idname, "LPENodeWorld")))
        {
          error = "Start with an optional Light/World, followed by an Object or Material Caustics output";
          return;
        }
        if (last_caster + 1 < int(path.size())) {
          const Socket &receiver = *path[last_caster + 1];
          if (!is_geometry_selection(receiver.owner_node()) ||
              (!STREQ(receiver.identifier, "Total Light") &&
               !STREQ(receiver.identifier, "Diffuse") && !STREQ(receiver.identifier, "Glossy") &&
               !STREQ(receiver.identifier, "Transmission") &&
               !STREQ(receiver.identifier, "Volume")))
          {
            error = "Connect Caustics to an Object or Material surface/volume receiver, or directly to Output";
          }
        }
        return;
      }
      /* Compatibility for receiver-side flags stored in older files. */
      for (size_t i = 0; i < path.size(); i++) {
        const Node &node = path[i]->owner_node();
        if (!STREQ(node.idname, "LPENodeObject") || !node.storage ||
            !(static_cast<const blender::NodeLPE *>(node.storage)->mode & 1))
          continue;
        if (value->mask) {
          error = "Shadow Caustics Mask is not supported; select the caustic color contribution";
          return;
        }
        if (i + 1 != path.size() || (path.size() != 2 && path.size() != 3) ||
            (!STREQ(path.front()->owner_node().idname, "LPENodeLight") &&
             !STREQ(path.front()->owner_node().idname, "LPENodeWorld")))
        {
          error = "Shadow Caustics needs Light/World -> optional caster -> receiver, without an expanded Repeat";
          return;
        }
        if (!STREQ(path[i]->identifier, "Total Light") && !STREQ(path[i]->identifier, "Diffuse") &&
            !STREQ(path[i]->identifier, "Glossy") && !STREQ(path[i]->identifier, "Transmission"))
        {
          error = "Shadow Caustics requires a surface receiver channel";
          return;
        }
        if (path.size() == 3 && !STREQ(path[1]->identifier, "Transmission") &&
            !STREQ(path[1]->identifier, "Glossy") &&
            !STREQ(path[1]->identifier, "Total Light"))
        {
          error = "Shadow Caustics requires Glossy, Transmission or Total Light on the caster";
          return;
        }
      }
    };
    validate_caustics(value->path);
    for (const Value &part : value->alternatives)
      validate_caustics(part->path);
    const auto validate_emission = [&](const std::vector<const Socket *> &path) {
      for (size_t i = 0; i < path.size(); i++) {
        if (STREQ(path[i]->identifier, "Photon") && (i != 0 || value->mask)) {
          error = "Photon is a Light/World source contribution, not a coverage Mask";
        }
        if (STREQ(path[i]->identifier, "Emission") && i != 0)
          error = "Emission is a source endpoint; start the path with Object or Material Emission";
      }
    };
    validate_emission(value->path);
    for (const Value &part : value->alternatives) validate_emission(part->path);
    if (!error.empty()) return 0;
    LPEExpression expression = value->expression;
    const bool shadow = std::any_of(value->path.begin(), value->path.end(), [](const Socket *socket) {
      return STREQ(socket->identifier, "Shadow");
    });
    if (shadow) {
      const auto &path = value->path;
      if (path.size() < 3 ||
          (!STREQ(path[0]->owner_node().idname, "LPENodeLight") &&
           !STREQ(path[0]->owner_node().idname, "LPENodeWorld")) ||
          !STREQ(path[0]->identifier, "Total Light") ||
          !std::all_of(path.begin() + 1, path.end() - 1, [](const Socket *socket) {
            return is_geometry_selection(socket->owner_node()) &&
                   STREQ(socket->identifier, "Shadow");
          }) ||
          !is_geometry_selection(path.back()->owner_node()) ||
          (!STREQ(path.back()->identifier, "Total Light") && !STREQ(path.back()->identifier, "Mask")) ||
          value->manager_node || !value->operation_nodes.empty()) {
        error = "Shadow requires Light/World Total Light -> Object/Material Shadow caster(s) -> Object/Material Total Light (receiver), without edits or Repeat";
        return 0;
      }
      const bool material_casters = STREQ(path[1]->owner_node().idname, "LPENodeMaterial");
      for (size_t i = 1; i + 1 < path.size(); i++) {
        if (STREQ(path[i]->owner_node().idname, "LPENodeMaterial") != material_casters) {
          error = "One shadow caster union must use either Object selections or Material selections";
          return 0;
        }
      }
      auto canonical = path;
      canonical.back() = &path.back()->owner_node().output_socket(0);
      std::sort(canonical.begin() + 1, canonical.end() - 1, [](const Socket *a, const Socket *b) {
        return a->owner_node().identifier < b->owner_node().identifier;
      });
      canonical.erase(std::unique(canonical.begin() + 1, canonical.end() - 1), canonical.end() - 1);
      auto found = std::find(graph.shadows.begin(), graph.shadows.end(), canonical);
      const size_t shadow_index = size_t(found - graph.shadows.begin());
      if (found == graph.shadows.end()) graph.shadows.push_back(std::move(canonical));
      const string base = "__LPE_shadow_" + std::to_string(shadow_index);
      expression.operation = LPEOperation::SHADOW_RATIO;
      expression.scalar = false;
      for (int i = 0; i < 2; i++) {
        LPEExpression raw;
        raw.operation = LPEOperation::PASS;
        raw.pass = base + (i == 0 ? "_reference" : "_blocked");
        expression.inputs[i] = int(graph.program->expressions.size());
        graph.program->expressions.push_back(std::move(raw));
      }
    }
    else if (!value->alternatives.empty()) {
      const bool edited = std::any_of(value->alternatives.begin(), value->alternatives.end(),
                                     [](const Value &part) {
                                       return part->manager_node || !part->operation_nodes.empty();
                                     });
      if (edited) {
        /* Disjoint paths form an exact union by addition. Lower the original
         * edit scopes so their per-light buffers, Apply and native render
         * overrides survive. The merge itself allocates no matcher states. */
        LPEExpression one;
        one.value = one_float3();
        const int factor = int(graph.program->expressions.size());
        graph.program->expressions.push_back(one);
        int index = lower(value->alternatives.front());
        for (size_t i = 1; i < value->alternatives.size(); i++) {
          const int next = lower(value->alternatives[i]);
          LPEExpression sum;
          sum.operation = LPEOperation::MIX;
          sum.mode = blender::MA_RAMP_ADD;
          sum.inputs[0] = factor;
          sum.inputs[1] = index;
          sum.inputs[2] = next;
          index = int(graph.program->expressions.size());
          graph.program->expressions.push_back(sum);
        }
        lowered[value.get()] = index;
        return index;
      }
      expression.operation = LPEOperation::PASS;
      expression.pass = "__LPE_union_" + std::to_string(graph.unions.size());
      std::vector<std::vector<const Socket *>> paths;
      for (const Value &part : value->alternatives)
        paths.push_back(part->path);
      graph.unions.push_back(std::move(paths));
    }
    else if (value->is_path && (value->manager_node || !value->operation_nodes.empty())) {
      expression.operation = LPEOperation::EDIT;
    }
    else if (value->is_path) {
      std::vector<std::pair<int, string>> key;
      for (const Socket *socket : value->path)
        key.emplace_back(socket->owner_node().identifier, socket->identifier);
      if (const auto found = lowered_paths.find(key); found != lowered_paths.end())
        return found->second;
      if (key.empty()) {
        error = "Output has no connected light path selection";
        return 0;
      }
      expression.operation = LPEOperation::PASS;
      expression.pass = "__LPE_raw_" + std::to_string(graph.paths.size());
      graph.paths.push_back(value->path);
      lowered_paths[key] = int(graph.program->expressions.size());
    }
    for (int i = 0; i < 3; i++)
      if (value->inputs[i])
        expression.inputs[i] = lower(value->inputs[i], depth + 1);
    if (!error.empty())
      return -1;
    int index = int(graph.program->expressions.size());
    graph.program->expressions.push_back(expression);
    if (shadow && value->mask) {
      LPEExpression mask;
      mask.operation = LPEOperation::SHADOW_MASK;
      mask.scalar = true;
      mask.inputs[0] = index;
      index = int(graph.program->expressions.size());
      graph.program->expressions.push_back(std::move(mask));
    }
    lowered[value.get()] = index;
    if (value->is_path && (value->manager_node || !value->operation_nodes.empty())) {
      graph.edits.push_back(
          {value->path, index, value->manager_node, value->manager_item, value->operation_nodes,
           value->output_node, value->output_item, value->scene_operation_nodes});
    }
    if (value->mask && !shadow) {
      graph.program->expressions[index].pass = "__LPE_mask_" + std::to_string(graph.masks.size());
      graph.masks.emplace_back(value->path, index);
    }
    return index;
  }
};

}  // namespace

std::vector<std::pair<string, bool>> lpe_viewport_catalog(const blender::bNodeTree &tree)
{
  tree.ensure_topology_cache();
  std::vector<std::pair<string, bool>> outputs;
  for (const Node *node : tree.all_nodes()) {
    if (STREQ(node->idname, "LPENodeSceneEdit") && !node->is_muted()) {
      for (const Socket *socket : node->output_sockets()) {
        const Socket *paired = input(*node, socket->identifier);
        if (paired && source(*paired))
          outputs.emplace_back(edit_preview_key(*node, *socket), true);
      }
      continue;
    }
    if (node->is_muted() || !node->storage ||
        !(STREQ(node->idname, "LPENodeViewer") || STREQ(node->idname, "LPENodeOutput"))) continue;
    const int mode = static_cast<const blender::NodeLPE *>(node->storage)->mode;
    const bool viewer = STREQ(node->idname, "LPENodeViewer") && mode == 1;
    if (!viewer && !STREQ(node->idname, "LPENodeOutput")) continue;
    for (const Socket *socket : node->input_sockets()) {
      if (ELEM(socket->type, blender::SOCK_RGBA, blender::SOCK_FLOAT) && source(*socket)) {
        outputs.emplace_back(viewer ? "LPE Viewer" : "LPE_" + string(socket->name),
                             viewer || mode == 0);
      }
    }
  }
  return outputs;
}

LPEGraph compile_lpe_graph(const blender::bNodeTree &tree, string &error, const bool preview,
                          const string &preview_output, const bool vcm_sources_only)
{
  Compiler compiler(tree, error);
  compiler.graph.program->viewport = preview;
  Context root;
  std::set<string> names;
  const Node *viewer = nullptr;
  for (const Node *node : tree.all_nodes()) {
    if (STREQ(node->idname, "LPENodeViewer") && !node->is_muted() && node->storage &&
        static_cast<const blender::NodeLPE *>(node->storage)->mode == 1)
      viewer = node;
  }
  compiler.graph.program->viewer_node = viewer ? viewer->identifier : 0;
  compiler.graph.program->viewer_denoising =
      viewer && static_cast<const blender::NodeLPE *>(viewer->storage)->use_denoising;
  const auto lower_output = [&](const Node *node, const string &only_output = "") {
    for (const Socket *socket : node->input_sockets()) {
      const Socket *upstream = source(*socket);
      if (!upstream || !ELEM(socket->type, blender::SOCK_RGBA, blender::SOCK_FLOAT))
        continue;
      const string name = node == viewer ? "LPE Viewer" : "LPE_" + string(socket->name);
      if (!only_output.empty() && only_output != name) continue;
      if (vcm_sources_only &&
          (!(STREQ(upstream->owner_node().idname, "LPENodeLight") ||
             STREQ(upstream->owner_node().idname, "LPENodeWorld")) ||
           !STREQ(upstream->identifier, "Total Light"))) {
        continue;
      }
      if (!names.insert(name).second) {
        error = "Duplicate LPE output: " + name;
        break;
      }
      Value value = compiler.evaluate(*upstream, root);
      if (!error.empty())
        return;
      if (!value)
        continue;
      if (node != viewer) {
        std::map<const Symbol *, Value> scoped_values;
        value = compiler.output_scope(value, node->identifier, socket->identifier, scoped_values);
      }
      compiler.output_roots.push_back(value);
      const int index = compiler.lower(value);
      /* A failed lowering can return a placeholder index without emitting an
       * expression (for example an incomplete caster/receiver shadow path).
       * Never publish that index to the Viewer or output table. */
      if (!error.empty()) {
        return;
      }
      if (node == viewer)
        compiler.graph.program->viewer_expression = index;
      compiler.graph.program->outputs.push_back(
          {name, index, socket->type == blender::SOCK_FLOAT ? 1 : 3});
      if (!error.empty())
        break;
    }
  };
  for (const Node *node : tree.all_nodes()) {
    if (!STREQ(node->idname, "LPENodeOutput") || node->is_muted() ||
        (preview && node->storage &&
         static_cast<const blender::NodeLPE *>(node->storage)->mode == 1))
      continue;
    lower_output(node);
    if (!error.empty()) break;
  }
  /* Light Edit roots are collected after the explicit outputs, so a path that
   * is not wired to an Output still contributes its edit. */
  if (!vcm_sources_only && error.empty())
    compiler.lower_edit_roots(root);
  compiler.graph.resident_expressions = int(compiler.graph.program->expressions.size());
  /* A Viewer is a temporary sampling root, even when the same upstream branch
   * also feeds a Render Only output. It does not own a resident label. */
  if (error.empty() && viewer)
    lower_output(viewer);
  if (preview && error.empty() && !preview_output.empty()) {
    for (const Node *node : tree.all_nodes()) {
      if (STREQ(node->idname, "LPENodeOutput") && !node->is_muted() && node->storage &&
          static_cast<const blender::NodeLPE *>(node->storage)->mode == 1) {
        lower_output(node, preview_output);
        if (!error.empty()) break;
      }
    }
  }
  /* Color operations are light authoring, not an arbitrary image compositor.
   * A Viewer of a manager's upstream operations remains a valid observation. */
  std::set<int> managed_operations;
  for (const auto &edit : compiler.graph.edits)
    if (edit.manager_node)
      managed_operations.insert(edit.operation_nodes.begin(), edit.operation_nodes.end());
  for (const auto &edit : compiler.graph.edits) {
    for (const int operation : edit.operation_nodes) {
      if (!managed_operations.contains(operation)) {
        error = "Color nodes require a Light Edit chain; connect the edited light to Light Edit";
      }
    }
  }
  /* Only a Light path can be edited: overrides and stacks write back to light
   * data, and World/Object/Material paths have no equivalent target. */
  if (error.empty()) {
    for (const auto &edit : compiler.graph.edits) {
      if (edit.path.empty() ||
          !STREQ(edit.path.front()->owner_node().idname, "LPENodeLight"))
      {
        error = "Light Edit only accepts Light paths; connect a Light node instead of World, "
                "Object or Material";
        break;
      }
    }
  }
  return std::move(compiler.graph);
}

std::map<string, std::vector<float>> evaluate_lpe_outputs(const LPEProgram &program,
                                                          const OutputDriver::Tile &tile)
{
  const size_t pixels = size_t(tile.size.x) * tile.size.y;
  std::map<string, std::vector<float>> buffers;
  const auto fetch = [&](const string &name, const int channels) -> const std::vector<float> & {
    auto [entry, inserted] = buffers.try_emplace(name);
    if (inserted) {
      entry->second.resize(pixels * channels, 0.0f);
      tile.get_pass_pixels(name, channels, entry->second.data());
    }
    return entry->second;
  };
  std::vector<const std::vector<float> *> leaves(program.expressions.size(), nullptr);
  std::vector<std::vector<const std::vector<float> *>> edit_leaves(program.expressions.size());
  for (size_t i = 0; i < program.expressions.size(); i++) {
    const auto &expression = program.expressions[i];
    if (expression.operation == LPEOperation::PASS || expression.operation == LPEOperation::MASK) {
      leaves[i] = &fetch(expression.pass, expression.operation == LPEOperation::MASK ? 1 : 3);
    }
    if (expression.operation == LPEOperation::EDIT) {
      for (const LPEEditTerm &term : expression.edit_terms) {
        edit_leaves[i].push_back(&fetch(term.pass, 3));
      }
    }
  }
  std::map<string, std::vector<float>> outputs;
  for (const LPEOutput &output : program.outputs)
    outputs[output.name].resize(pixels * output.channels);
  std::vector<float *> output_pixels;
  for (const LPEOutput &output : program.outputs)
    output_pixels.push_back(outputs.at(output.name).data());
  parallel_for(blocked_range<size_t>(0, pixels, 1024), [&](const blocked_range<size_t> &range) {
    std::vector<float3> values(program.expressions.size());
    for (size_t p = range.begin(); p < range.end(); p++) {
      for (size_t i = 0; i < program.expressions.size(); i++) {
        const auto &expression = program.expressions[i];
        if (expression.operation == LPEOperation::EDIT) {
          values[i] = make_float3(0.0f);
          for (size_t j = 0; j < expression.edit_terms.size(); j++) {
            const auto &buffer = *edit_leaves[i][j];
            values[i] += make_float3(buffer[p * 3], buffer[p * 3 + 1], buffer[p * 3 + 2]) *
                         expression.edit_terms[j].multiplier;
          }
        }
        else if (leaves[i]) {
          const auto &buffer = *leaves[i];
          values[i] = expression.operation == LPEOperation::MASK ?
                          make_float3(buffer[p]) :
                          make_float3(buffer[p * 3], buffer[p * 3 + 1], buffer[p * 3 + 2]);
        }
        else {
          float3 operands[3] = {make_float3(0.0f), make_float3(0.0f), make_float3(0.0f)};
          for (int j = 0; j < 3; j++)
            if (expression.inputs[j] >= 0)
              operands[j] = values[expression.inputs[j]];
          values[i] = evaluate_operation(expression, operands[0], operands[1], operands[2]);
        }
      }
      for (size_t o = 0; o < program.outputs.size(); o++) {
        const LPEOutput &output = program.outputs[o];
        float *buffer = output_pixels[o];
        const float3 value = values[output.expression];
        buffer[p * output.channels] = value.x;
        if (output.channels == 3) {
          buffer[p * 3 + 1] = value.y;
          buffer[p * 3 + 2] = value.z;
        }
      }
    }
  });
  std::vector<float> combined;
  if (!program.combined_terms.empty() && evaluate_lpe_combined(program, tile, combined))
    outputs["Combined"] = std::move(combined);
  return outputs;
}

CCL_NAMESPACE_END
