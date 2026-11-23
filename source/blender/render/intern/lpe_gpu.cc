/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: GPL-2.0-or-later */

#include "RE_lpe.hh"

#include "BKE_global.hh"
#include "BLI_threads.hh"
#include "BLI_utildefines.hh"
#include "DRW_engine.hh"
#include "GPU_capabilities.hh"
#include "GPU_compute.hh"
#include "GPU_context.hh"
#include "GPU_shader.hh"
#include "GPU_state.hh"
#include "GPU_storage_buffer.hh"
#include "gpu_shader_create_info.hh"

#include <algorithm>
#include <limits>

namespace blender::render {

class CachedGPUImage : public LPEGPUImage {
  struct Chunk {
    int y = 0;
    int height = 0;
    gpu::StorageBuf *source = nullptr;
    std::vector<int> offsets;
  };
  int width_, height_;
  std::vector<int> channels_;
  std::vector<Chunk> chunks_;
  gpu::Shader *shader_ = nullptr;
  bool ready_ = false;
  bool inputs_changed_ = false;

  bool initialize(const std::vector<LPEMixInput> &inputs)
  {
    if (!shader_) {
      using namespace gpu::shader;
      /* The generated-source helper emits CREATE_INFO_RES_PASS_pyGPU_Shader. */
      ShaderCreateInfo info("pyGPU_Shader");
      info.local_group_size(16, 16);
      info.builtins(BuiltinBits::GLOBAL_INVOCATION_ID);
      info.push_constant(Type::int_t, "width");
      info.push_constant(Type::int_t, "height");
      info.push_constant(Type::int_t, "channels");
      info.push_constant(Type::int_t, "term_count");
      info.push_constant(Type::int_t, "alpha_offset");
      info.push_constant(Type::float3_t, "bias");
      info.storage_buf(0, Qualifier::read, "float", "source_values[]");
      info.storage_buf(1, Qualifier::read, "int", "term_data[]");
      info.storage_buf(2, Qualifier::read, "float", "weights[]");
      info.storage_buf(3, Qualifier::write, "float", "result_values[]");
      /* Same generated-source route as Blender's GPU Python/OCIO shaders. */
      info.compute_source("draw_colormanagement_lib.glsl");
      info.compute_source_generated = R"GLSL(
void main()
{
  ivec2 p = ivec2(gl_GlobalInvocationID.xy);
  if (p.x >= width || p.y >= height) return;
  int pixel = p.y * width + p.x;
  vec3 value = bias;
  for (int t = 0; t < term_count; t++) {
    int stride = term_data[2 * t + 1];
    int offset = term_data[2 * t] + pixel * stride;
    vec3 input_value = stride == 1 ? vec3(source_values[offset]) :
        vec3(source_values[offset], source_values[offset + 1], source_values[offset + 2]);
    for (int row = 0; row < 3; row++) {
      int m = 9 * t + 3 * row;
      value[row] += dot(vec3(weights[m], weights[m + 1], weights[m + 2]), input_value);
    }
  }
  result_values[pixel * channels] = value.x;
  if (channels >= 3) {
    result_values[pixel * channels + 1] = value.y;
    result_values[pixel * channels + 2] = value.z;
  }
  if (channels == 4) result_values[pixel * channels + 3] = source_values[alpha_offset + pixel * 4 + 3];
}
)GLSL";
      shader_ = GPU_shader_create_from_info_python(reinterpret_cast<GPUShaderCreateInfo *>(&info));
      if (!shader_)
        return false;
    }
    if (ready_) {
      bool same_layout = channels_.size() == inputs.size();
      for (size_t i = 0; same_layout && i < inputs.size(); i++)
        same_layout = channels_[i] == inputs[i].channels;
      if (same_layout) {
        if (inputs_changed_) {
          for (const auto &chunk : chunks_) {
            std::vector<float> packed;
            for (const auto &input : inputs) {
              const float *start = input.pixels + size_t(chunk.y) * width_ * input.channels;
              packed.insert(packed.end(), start, start + size_t(chunk.height) * width_ * input.channels);
            }
            if (packed.empty()) packed.resize(4, 0);
            GPU_storagebuf_update(chunk.source, packed.data());
          }
        }
        inputs_changed_ = false;
        return true;
      }
      ready_ = false;
    }
    for (const auto &chunk : chunks_)
      GPU_storagebuf_free(chunk.source);
    chunks_.clear();
    channels_.clear();
    size_t stride = 0;
    for (const auto &input : inputs) {
      if (!ELEM(input.channels, 1, 3, 4) || !input.pixels)
        return false;
      stride += input.channels;
      channels_.push_back(input.channels);
    }
    const size_t row_bytes = size_t(width_) * std::max<size_t>(stride, 3) * sizeof(float);
    const size_t limit = std::min<size_t>(GPU_max_storage_buffer_size(), 128 * 1024 * 1024);
    if (!row_bytes || row_bytes > limit)
      return false;
    const int rows = int(std::min<size_t>(height_, limit / row_bytes));
    for (int y = 0; y < height_; y += rows) {
      Chunk chunk;
      chunk.y = y;
      chunk.height = std::min(rows, height_ - y);
      std::vector<float> packed;
      packed.reserve(std::max<size_t>(4, size_t(width_) * chunk.height * stride));
      for (const auto &input : inputs) {
        chunk.offsets.push_back(int(packed.size()));
        const float *start = input.pixels + size_t(y) * width_ * input.channels;
        packed.insert(packed.end(), start, start + size_t(chunk.height) * width_ * input.channels);
      }
      if (packed.empty())
        packed.resize(4, 0);
      chunk.source = GPU_storagebuf_create_ex(packed.size() * sizeof(float), packed.data(), GPU_USAGE_STATIC, "LPE frozen contributions");
      if (!chunk.source)
        return false;
      chunks_.push_back(std::move(chunk));
    }
    ready_ = true;
    inputs_changed_ = false;
    return true;
  }

 public:
  CachedGPUImage(const int width, const int height) : width_(width), height_(height) {}
  void tag_inputs_changed() override { inputs_changed_ = true; }
  ~CachedGPUImage() override
  {
    /* These immutable input buffers never own readback fences. The GPU backend
     * defers their resource deletion when there is no active context. */
    for (const auto &chunk : chunks_)
      GPU_storagebuf_free(chunk.source);
    if (shader_)
      GPU_shader_free(shader_);
  }

  bool evaluate(const std::vector<LPEMixInput> &inputs,
                const std::vector<LPEMixOutput> &outputs,
                std::map<std::string, std::vector<float>> &pixels,
                const std::function<bool()> &cancel,
                const std::function<void(float)> &progress) override
  {
    if (!GPU_context_active_get() || !initialize(inputs))
      return false;
    size_t completed = 0;
    const size_t total = std::max<size_t>(1, outputs.size() * chunks_.size());
    for (const auto &output : outputs) {
      if (cancel && cancel()) return false;
      if (!ELEM(output.channels, 1, 3, 4) ||
          (output.channels == 4 && (output.alpha_input < 0 || size_t(output.alpha_input) >= inputs.size() || inputs[output.alpha_input].channels != 4)))
        return false;
      std::vector<float> weights;
      for (const auto &term : output.terms) {
        if (term.input < 0 || size_t(term.input) >= inputs.size())
          return false;
        weights.insert(weights.end(), term.matrix.begin(), term.matrix.end());
      }
      if (weights.empty())
        weights.resize(4, 0);
      auto *weight_buffer = GPU_storagebuf_create_ex(weights.size() * sizeof(float), weights.data(), GPU_USAGE_STREAM, "LPE light weights");
      if (!weight_buffer)
        return false;
      auto &destination = pixels[output.name];
      destination.resize(size_t(width_) * height_ * output.channels);
      for (const auto &chunk : chunks_) {
        std::vector<int> terms;
        for (const auto &term : output.terms) {
          terms.push_back(chunk.offsets[term.input]);
          terms.push_back(inputs[term.input].channels);
        }
        if (terms.empty())
          terms.resize(4, 0);
        auto *term_buffer = GPU_storagebuf_create_ex(terms.size() * sizeof(int), terms.data(), GPU_USAGE_STREAM, "LPE light terms");
        const size_t count = size_t(width_) * chunk.height * output.channels;
        auto *result_buffer = GPU_storagebuf_create_ex(count * sizeof(float), nullptr, GPU_USAGE_DYNAMIC, "LPE mixed pixels");
        if (!term_buffer || !result_buffer) {
          if (term_buffer) GPU_storagebuf_free(term_buffer);
          if (result_buffer) GPU_storagebuf_free(result_buffer);
          GPU_storagebuf_free(weight_buffer);
          return false;
        }
        GPU_shader_bind(shader_);
        GPU_shader_uniform_1i(shader_, "width", width_);
        GPU_shader_uniform_1i(shader_, "height", chunk.height);
        GPU_shader_uniform_1i(shader_, "channels", output.channels);
        GPU_shader_uniform_1i(shader_, "term_count", int(output.terms.size()));
        GPU_shader_uniform_1i(shader_, "alpha_offset", output.alpha_input >= 0 ? chunk.offsets[output.alpha_input] : 0);
        GPU_shader_uniform_3fv(shader_, "bias", output.bias.data());
        GPU_storagebuf_bind(chunk.source, 0);
        GPU_storagebuf_bind(term_buffer, 1);
        GPU_storagebuf_bind(weight_buffer, 2);
        GPU_storagebuf_bind(result_buffer, 3);
        GPU_compute_dispatch(shader_, (width_ + 15) / 16, (chunk.height + 15) / 16, 1);
        GPU_memory_barrier(GPU_BARRIER_SHADER_STORAGE | GPU_BARRIER_BUFFER_UPDATE);
        GPU_storagebuf_read(result_buffer, destination.data() + size_t(chunk.y) * width_ * output.channels);
        GPU_shader_unbind();
        GPU_storagebuf_free(result_buffer);
        GPU_storagebuf_free(term_buffer);
        if (progress) progress(float(++completed) / total);
        if (cancel && cancel()) {
          GPU_storagebuf_free(weight_buffer);
          return false;
        }
      }
      GPU_storagebuf_free(weight_buffer);
    }
    return true;
  }
};

std::unique_ptr<LPEGPUImage> lpe_gpu_image_create(const int width, const int height)
{
  return std::make_unique<CachedGPUImage>(width, height);
}

}  // namespace blender::render
