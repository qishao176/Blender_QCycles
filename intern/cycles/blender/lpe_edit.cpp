/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: Apache-2.0 */

#include "blender/lpe.h"

#include <set>
#include <mutex>
#include "scene/camera.h"
#include "DEG_depsgraph_query.hh"

#include "BKE_layer.hh"
#include "BKE_lib_id.hh"
#include "BKE_main.hh"
#include "BKE_node_runtime.hh"
#include "BKE_scene.hh"
#include "BLI_listbase.hh"
#include "BLT_translation.hh"
#include "DEG_depsgraph.hh"
#include "DNA_layer_types.h"
#include "DNA_anim_types.h"
#include "DNA_light_types.h"
#include "DNA_object_types.h"
#include "DNA_scene_types.h"
#include "IMB_colormanagement.hh"
#include "IMB_imbuf_types.hh"
#include "IMB_imbuf.hh"
#include "RE_lpe.hh"
#include "RE_pipeline.h"
#include "RNA_access.hh"
#include "MEM_guardedalloc.h"

#include "util/color.h"
#include "util/math.h"
#include "kernel/svm/color_util.h"

CCL_NAMESPACE_BEGIN

namespace {

float3 effective_color(const LPELightParameters &parameters)
{
  float3 color = parameters.color;
  if (parameters.use_temperature) {
    float rgb[4];
    blender::IMB_colormanagement_blackbody_temperature_to_rgb(rgb, parameters.temperature);
    color *= make_float3(rgb[0], rgb[1], rgb[2]);
  }
  return color;
}

float3 emission(const LPELightParameters &parameters)
{
  return effective_color(parameters) * parameters.power * exp2f(parameters.exposure);
}

template<typename Parameters>
bool apply_color_operation(LPELightParameters &target, const Parameters &data, const int type)
{
  if (type == 3) {
    target.color = make_float3(data.color[0], data.color[1], data.color[2]);
    target.power = data.power;
    target.exposure = data.exposure;
    target.temperature = data.temperature;
    target.use_temperature = data.use_temperature != 0;
    return true;
  }
  if (type < 0 || type > 2)
    return false;
  if (data.factor == 0)
    return true;
  const float3 color = effective_color(target);
  float3 mapped;
  if (type == 2) {
    if (data.brightness == 0 && data.contrast == 0)
      return true;
    mapped = svm_brightness_contrast(color, data.brightness, data.contrast);
  }
  else {
    if (data.hue == 0.5f && data.saturation == 1 && data.value == 1)
      return true;
    float3 components = type == 0 ? rgb_to_hsv(color) : rgb_to_hsl(color);
    components.x += data.hue - 0.5f;
    components.x -= floorf(components.x);
    components.y = saturatef(components.y * data.saturation);
    components.z = type == 0 ? components.z * data.value : saturatef(components.z * data.value);
    mapped = type == 0 ? hsv_to_rgb(components) : hsl_to_rgb(components);
  }
  target.color = interp(color, mapped, data.factor);
  target.use_temperature = false;
  return true;
}

bool update_term(LPEEditTerm &term, const blender::bNodeTree &tree, string &error,
                 LPELightParameters *r_target = nullptr, const bool compute_multiplier = true, const bool clamp_missing = false)
{
  const blender::bNode *manager = tree.node_by_id(term.manager_node);
  if (term.manager_node &&
      (!manager || !manager->storage || !STREQ(manager->idname, "LPENodeSceneEdit"))) {
    error = "Light Edit manager changed; render again";
    return false;
  }
  LPELightParameters target = term.operation_nodes.empty() ? term.sampling_parameters : term.source_parameters;
  for (const int identifier : term.operation_nodes) {
    const blender::bNode *node = tree.node_by_id(identifier);
    if (!node) {
      error = "Light operation changed; render again";
      return false;
    }
    if (node->is_muted())
      continue;
    if (!node->storage || (!STREQ(node->idname, "LPENodeLightOverride") &&
                           !STREQ(node->idname, "LPENodeLightOffset") &&
                           !STREQ(node->idname, "LPENodeLightHSL") &&
                           !STREQ(node->idname, "LPENodeLightBrightnessContrast")))
    {
      error = "Unsupported operation in the edit plan";
      return false;
    }
    const auto &data = *static_cast<const blender::NodeLPE *>(node->storage);
    const int type = STREQ(node->idname, "LPENodeLightOverride") ? 3 :
                     STREQ(node->idname, "LPENodeLightHSL") ? 1 :
                     STREQ(node->idname, "LPENodeLightBrightnessContrast") ? 2 : 0;
    apply_color_operation(target, data, type);
  }
  if (r_target)
    *r_target = target;
  if (!compute_multiplier)
    return true;
  const float3 before = emission(term.sampling_parameters);
  const float3 after = emission(target);
  float multiplier[3];
  const float old_channels[3] = {before.x, before.y, before.z};
  const float new_channels[3] = {after.x, after.y, after.z};
  for (int i = 0; i < 3; i++) {
    if (!std::isfinite(old_channels[i]) || !std::isfinite(new_channels[i])) {
      error = "Non-finite light strength: " + term.source_name;
      return false;
    }
    if (old_channels[i] == 0 && new_channels[i] != 0) {
      if (!clamp_missing) {
        error = "A zero-energy light or color channel needs resampling: " + term.source_name;
        return false;
      }
      /* Interactive previews only own what was sampled. A channel that never had
       * energy cannot be raised from the cache, so keep it as sampled instead of
       * failing the whole viewport sync. The next render samples the edited light
       * and matches again. */
      multiplier[i] = 1.0f;
      continue;
    }
    multiplier[i] = old_channels[i] == 0 ? 1.0f : new_channels[i] / old_channels[i];
    if (!std::isfinite(multiplier[i])) {
      error = "Light multiplier is out of range";
      return false;
    }
  }
  term.multiplier = make_float3(multiplier[0], multiplier[1], multiplier[2]);
  if (r_target)
    *r_target = target;
  return true;
}

struct CachedTile : OutputDriver::Tile {
  struct Buffer {
    int channels;
    std::vector<float> pixels;
  };
  std::map<string, Buffer> buffers;
  size_t bytes = 0;
  std::unique_ptr<blender::render::LPEGPUImage> gpu_image;

  explicit CachedTile(const OutputDriver::Tile &tile)
      : Tile(tile.offset, tile.size, tile.full_size, tile.layer, tile.view)
  {
  }

  bool capture(const OutputDriver::Tile &tile, const string &name, const int channels)
  {
    if (buffers.contains(name))
      return true;
    Buffer buffer;
    buffer.channels = channels;
    buffer.pixels.resize(size_t(size.x) * size.y * channels);
    if (!tile.get_pass_pixels(name, channels, buffer.pixels.data()))
      return false;
    bytes += buffer.pixels.size() * sizeof(float);
    buffers.emplace(name, std::move(buffer));
    if (gpu_image) gpu_image->tag_inputs_changed();
    return true;
  }

  bool get_pass_pixels(const string_view name, const int channels, float *pixels) const override
  {
    const auto found = buffers.find(string(name));
    if (found == buffers.end() || found->second.channels != channels)
      return false;
    std::copy(found->second.pixels.begin(), found->second.pixels.end(), pixels);
    return true;
  }
  bool set_pass_pixels(string_view, int, const float *) const override
  {
    return false;
  }

  bool evaluate_gpu(const LPEProgram &program, std::map<string, std::vector<float>> &pixels,
                    const std::function<bool()> &cancel = {},
                    const std::function<void(float)> &progress = {})
  {
    using Matrix = std::array<float, 9>;
    struct LinearValue {
      std::array<float, 3> bias = {};
      std::map<string, Matrix> terms;
    };
    if (program.expressions.size() > 512)
      return false;
    const Matrix identity = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    const Matrix scalar = {1, 0, 0, 1, 0, 0, 1, 0, 0};
    const auto combine = [](LinearValue &destination, const LinearValue &source,
                             const Matrix &transform, const float factor) {
      for (int row = 0; row < 3; row++)
        for (int k = 0; k < 3; k++)
          destination.bias[row] += factor * transform[3 * row + k] * source.bias[k];
      for (const auto &[pass, matrix] : source.terms) {
        Matrix &target = destination.terms[pass];
        for (int row = 0; row < 3; row++)
          for (int col = 0; col < 3; col++)
            for (int k = 0; k < 3; k++)
              target[3 * row + col] += factor * transform[3 * row + k] * matrix[3 * k + col];
      }
    };
    std::vector<LinearValue> values(program.expressions.size());
    for (size_t i = 0; i < program.expressions.size(); i++) {
      const auto &expression = program.expressions[i];
      auto &value = values[i];
      if (expression.operation == LPEOperation::CONSTANT) {
        value.bias = {expression.value.x, expression.value.y, expression.value.z};
      }
      else if (expression.clamp) {
        return false;
      }
      else if (expression.operation == LPEOperation::PASS || expression.operation == LPEOperation::MASK) {
        value.terms[expression.pass] = identity;
      }
      else if (expression.operation == LPEOperation::EDIT) {
        for (const auto &term : expression.edit_terms) {
          Matrix &matrix = value.terms[term.pass];
          matrix[0] += term.multiplier.x;
          matrix[4] += term.multiplier.y;
          matrix[8] += term.multiplier.z;
        }
      }
      else if (expression.operation == LPEOperation::TO_FLOAT) {
        Matrix luminance;
        for (int col = 0; col < 3; col++) {
          float basis[3] = {0, 0, 0};
          basis[col] = 1;
          const float weight = blender::IMB_colormanagement_get_luminance(basis);
          for (int row = 0; row < 3; row++)
            luminance[3 * row + col] = weight;
        }
        combine(value, values[expression.inputs[0]], luminance, 1);
      }
      else if (expression.operation == LPEOperation::MIX &&
               (expression.mode == blender::MA_RAMP_ADD || expression.mode == blender::MA_RAMP_BLEND ||
                expression.mode == blender::MA_RAMP_SUB) &&
               program.expressions[expression.inputs[0]].operation == LPEOperation::CONSTANT) {
        float factor = program.expressions[expression.inputs[0]].value.x;
        if (expression.clamp_factor)
          factor = saturatef(factor);
        combine(value, values[expression.inputs[1]], identity,
                expression.mode == blender::MA_RAMP_BLEND ? 1 - factor : 1);
        combine(value, values[expression.inputs[2]], identity,
                expression.mode == blender::MA_RAMP_SUB ? -factor : factor);
      }
      else if (expression.operation == LPEOperation::MIX &&
               expression.mode == blender::MA_RAMP_MULT &&
               program.expressions[expression.inputs[0]].operation == LPEOperation::CONSTANT &&
               (program.expressions[expression.inputs[1]].operation == LPEOperation::CONSTANT ||
                program.expressions[expression.inputs[2]].operation == LPEOperation::CONSTANT)) {
        float factor = program.expressions[expression.inputs[0]].value.x;
        if (expression.clamp_factor) factor = saturatef(factor);
        const bool left_constant = program.expressions[expression.inputs[1]].operation == LPEOperation::CONSTANT;
        const auto &color = values[expression.inputs[left_constant ? 1 : 2]].bias;
        Matrix matrix = {};
        for (int row = 0; row < 3; row++) {
          matrix[3 * row + row] = left_constant ? factor * color[row] : 1 - factor + factor * color[row];
          if (left_constant) value.bias[row] = (1 - factor) * color[row];
        }
        combine(value, values[expression.inputs[left_constant ? 2 : 1]], matrix, 1);
      }
      else if (expression.operation == LPEOperation::MATH &&
               (expression.mode == NODE_MATH_ADD || expression.mode == NODE_MATH_SUBTRACT)) {
        combine(value, values[expression.inputs[0]], scalar, 1);
        combine(value, values[expression.inputs[1]], scalar, expression.mode == NODE_MATH_ADD ? 1 : -1);
      }
      else {
        return false;
      }
    }
    std::map<string, int> indices;
    std::vector<blender::render::LPEMixInput> inputs;
    for (const auto &[name, buffer] : buffers) {
      indices[name] = int(inputs.size());
      inputs.push_back({buffer.pixels.data(), buffer.channels});
    }
    std::vector<blender::render::LPEMixOutput> outputs;
    for (const auto &output : program.outputs) {
      const auto &value = values[output.expression];
      blender::render::LPEMixOutput plan;
      plan.name = output.name;
      plan.channels = output.channels;
      plan.bias = value.bias;
      for (const float component : plan.bias)
        if (!std::isfinite(component)) return false;
      for (const auto &[name, matrix] : value.terms) {
        const auto found = indices.find(name);
        if (found == indices.end())
          return false;
        for (const float component : matrix)
          if (!std::isfinite(component)) return false;
        plan.terms.push_back({found->second, matrix});
      }
      outputs.push_back(std::move(plan));
    }
    if (indices.contains("Combined")) {
      blender::render::LPEMixOutput combined;
      combined.name = "Combined";
      combined.channels = 4;
      combined.alpha_input = indices.at("Combined");
      combined.terms.push_back({combined.alpha_input, identity});
      for (const auto &term : program.combined_terms) {
        if (!indices.contains(term.global_pass)) return false;
        const float3 delta = term.multiplier - one_float3();
        combined.terms.push_back({indices.at(term.global_pass), {delta.x, 0, 0, 0, delta.y, 0, 0, 0, delta.z}});
      }
      outputs.push_back(std::move(combined));
    }

    if (!gpu_image)
      gpu_image = blender::render::lpe_gpu_image_create(size.x, size.y);
    return gpu_image->evaluate(inputs, outputs, pixels, cancel, progress);
  }
};

}  // namespace

struct LPEPreviewCache {
  std::unique_ptr<CachedTile> tile;
  uint64_t revision = 0;
};

std::shared_ptr<LPEPreviewCache> create_lpe_preview_cache()
{
  return std::make_shared<LPEPreviewCache>();
}

bool evaluate_lpe_light_target(const LPEEditTerm &term, const blender::bNodeTree &tree,
                                LPELightParameters &target, string &error)
{
  LPEEditTerm evaluated = term;
  return update_term(evaluated, tree, error, &target, false);
}

float3 lpe_light_emission(const LPELightParameters &parameters)
{
  return emission(parameters);
}

bool update_lpe_edit_terms(LPEProgram &program, const blender::bNodeTree &tree, string &error,
                           const bool clamp_missing)
{
  tree.ensure_topology_cache();
  program.combined_terms.clear();
  std::map<string, LPEEditTerm> global;
  for (auto &expression : program.expressions) {
    for (auto &term : expression.edit_terms) {
      if (!update_term(term, tree, error, nullptr, true, clamp_missing)) return false;
      if (!term.manager_node || term.global_pass.empty()) continue;
      LPEEditTerm scene_term = term;
      scene_term.operation_nodes = term.scene_operation_nodes;
      if (!update_term(scene_term, tree, error, nullptr, true, clamp_missing)) return false;
      if (len_squared(scene_term.multiplier - one_float3()) <= 1e-12f) continue;
      const auto [found, inserted] = global.emplace(term.source_instance, scene_term);
      if (!inserted && len_squared(found->second.multiplier - scene_term.multiplier) > 1e-12f) {
        error = "Light Edit outputs request conflicting settings for the same scene light";
        return false;
      }
    }
  }
  for (const auto &[key, term] : global) program.combined_terms.push_back(term);
  return true;
}

bool evaluate_lpe_combined(const LPEProgram &program, const OutputDriver::Tile &tile,
                           std::vector<float> &pixels)
{
  const size_t count = size_t(tile.size.x) * tile.size.y;
  pixels.resize(count * 4);
  if (!tile.get_pass_pixels("Combined", 4, pixels.data())) return false;
  std::vector<float> contribution(count * 3);
  for (const auto &term : program.combined_terms) {
    if (!tile.get_pass_pixels(term.global_pass, 3, contribution.data())) return false;
    const float3 delta = term.multiplier - one_float3();
    for (size_t i = 0; i < count; i++) {
      pixels[4 * i] += contribution[3 * i] * delta.x;
      pixels[4 * i + 1] += contribution[3 * i + 1] * delta.y;
      pixels[4 * i + 2] += contribution[3 * i + 2] * delta.z;
    }
  }
  return true;
}

bool evaluate_lpe_preview(const LPEProgram &source, const OutputDriver::Tile &tile,
                          std::vector<float> &pixels, LPEPreviewCache *cache, int expression)
{
  if (expression < 0) expression = source.viewer_expression;
  if (expression < 0 || size_t(expression) >= source.expressions.size())
    return false;
  LPEProgram program;
  std::set<int> needed;
  const auto collect = [&](auto &&self, const int index) -> void {
    if (index < 0 || !needed.insert(index).second)
      return;
    for (const int input : source.expressions[index].inputs)
      self(self, input);
  };
  collect(collect, expression);
  /* Lowering is topological. Compact the live dependency slice so unrelated
   * outputs do not inflate CPU work or the GPU expression budget. */
  std::vector<int> remap(source.expressions.size(), -1);
  for (const int index : needed) {
    remap[index] = int(program.expressions.size());
    auto expression = source.expressions[index];
    for (int &input : expression.inputs)
      if (input >= 0) input = remap[input];
    program.expressions.push_back(std::move(expression));
  }
  program.outputs.push_back({"LPE Viewer", remap[expression], 3});
  LPEPreviewCache temporary;
  if (!cache) cache = &temporary;
  if (!cache->tile || cache->tile->size != tile.size || cache->tile->offset != tile.offset) {
    cache->tile = std::make_unique<CachedTile>(tile);
    cache->revision = 0;
  }
  CachedTile &captured = *cache->tile;
  std::set<string> required;
  for (const auto &expression : program.expressions) {
    if (expression.operation == LPEOperation::PASS || expression.operation == LPEOperation::MASK)
      required.insert(expression.pass);
    for (const auto &term : expression.edit_terms) required.insert(term.pass);
  }
  /* Drop obsolete inputs instead of growing a cache on every Viewer switch. */
  for (auto it = captured.buffers.begin(); it != captured.buffers.end();) {
    if (!required.contains(it->first)) {
      captured.bytes -= it->second.pixels.size() * sizeof(float);
      it = captured.buffers.erase(it);
      if (captured.gpu_image) captured.gpu_image->tag_inputs_changed();
    }
    else ++it;
  }
  const uint64_t revision = tile.get_content_revision();
  if (!revision || revision != cache->revision) {
    cache->revision = 0;
    if (captured.gpu_image) captured.gpu_image->tag_inputs_changed();
    for (auto &[name, buffer] : captured.buffers)
      if (!tile.get_pass_pixels(name, buffer.channels, buffer.pixels.data())) return false;
  }
  for (const auto &expression : program.expressions) {
    if (expression.operation == LPEOperation::PASS || expression.operation == LPEOperation::MASK) {
      if (!captured.capture(tile, expression.pass, expression.operation == LPEOperation::MASK ? 1 : 3))
        return false;
    }
    for (const auto &term : expression.edit_terms)
      if (!captured.capture(tile, term.pass, 3))
        return false;
  }
  std::map<string, std::vector<float>> outputs;
  cache->revision = revision;
  if (!captured.evaluate_gpu(program, outputs))
    outputs = evaluate_lpe_outputs(program, captured);
  pixels = std::move(outputs["LPE Viewer"]);
  return pixels.size() == size_t(tile.size.x) * tile.size.y * 3;
}

CCL_NAMESPACE_END
