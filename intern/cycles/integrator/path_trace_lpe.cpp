/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: Apache-2.0 */

#include "integrator/path_trace.h"
#include "integrator/path_trace_display.h"
#include "integrator/path_trace_tile.h"
#include "integrator/render_scheduler.h"
#include "scene/film.h"
#include "scene/lpe.h"
#include "scene/pass.h"
#include "scene/scene.h"
#include "session/display_driver.h"
#include "util/progress.h"
#include "util/task.h"
#include "util/tbb.h"
#include "util/time.h"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <set>
#include <tuple>
#include <cmath>

CCL_NAMESPACE_BEGIN

void PathTrace::tag_lpe_interaction()
{
  lpe_last_interaction_time_.store(time_dt(), std::memory_order_relaxed);
}

void PathTrace::request_lpe_display_update()
{
  tag_lpe_interaction();
  /* Publish after the immutable preview plan so the display sees matching state. */
  {
    const thread_scoped_lock lock(lpe_batch_mutex_);
    lpe_display_revision_.fetch_add(1, std::memory_order_release);
  }
  lpe_batch_condition_.notify_one();
}

/* Only the display thread reads this snapshot. Nonlinear Viewer plans already own a cache
 * of their inputs, keyed by tile revision. A miss must wait for a completed batch. */
struct PathTrace::LPEBatchPreview : OutputDriver::Tile {
  LPEBatchPreview(const PathTraceTile &tile, string output)
      : Tile(tile.offset, tile.size, tile.full_size, tile.layer, tile.view),
        revision(tile.get_content_revision()), selection(std::move(output))
  {
  }

  uint64_t get_content_revision() const override { return revision; }
  bool get_pass_pixels(string_view, int, float *) const override { return false; }
  bool set_pass_pixels(string_view, int, const float *) const override { return false; }

  uint64_t revision;
  int samples = 0;
  const string selection;
  std::map<ustring, std::vector<float>> inputs;
  std::vector<float> combined;
};

/* The LPE expression evaluator asks its tile for one normalized image at a time.
 * Read only that image from the device, where photon films may have three raw segments. */
bool PathTrace::read_photon_preview_pass(const BufferPass &pass, int channels, float *pixels)
{
  return device_scene_->data.integrator.use_photon_caustics &&
         read_viewport_preview_pass(pass, channels, pixels);
}

bool PathTrace::read_viewport_preview_pass(const BufferPass &pass, int channels, float *pixels)
{
  if (render_scheduler_.is_background() ||
      pass.mode != PassMode::NOISY || pass.offset == PASS_UNUSED ||
      !((pass.type == PASS_LPE && channels == 3) ||
        (pass.type == PASS_COMBINED && (channels == 3 || channels == 4)) ||
        (pass.type == PASS_AOV_VALUE && channels == 1))) {
    return false;
  }
  const PassAccessor::PassAccessInfo info(pass);
  const auto extract = [&](PathTraceWork &work) {
    return work.read_preview_pass(info, get_render_tile_params().exposure,
                                  get_num_render_tile_samples(), channels, pixels);
  };
  if (photon_replicas_) {
    return extract(*photon_reduce_work_);
  }
  for (const auto &work : path_trace_works_) {
    if (!extract(*work)) {
      return false;
    }
  }
  return true;
}

void PathTrace::capture_lpe_batch_preview()
{
  const auto plan = film_->get_lpe_preview();
  if ((!plan->combined && !plan->evaluate && plan->terms.empty()) || film_->get_show_active_pixels()) {
    lpe_batch_preview_.reset();
    return;
  }
  const PathTraceTile tile(*this, true);
  const size_t pixels = size_t(tile.size.x) * tile.size.y;
  if (!pixels || !get_num_render_tile_samples()) {
    lpe_batch_preview_.reset();
    return;
  }
  const string selection = film_->get_lpe_display_output();
  auto snapshot = std::move(lpe_batch_preview_);
  if (!snapshot || snapshot->size != tile.size || snapshot->selection != selection) {
    snapshot = std::make_shared<LPEBatchPreview>(tile, selection);
  }
  snapshot->revision = tile.get_content_revision();
  snapshot->samples = get_num_render_tile_samples();
  const BufferParams &params = get_render_tile_params();
  const auto read = [&](const BufferPass *pass, const int channels, std::vector<float> &values) {
    if (!pass || pass->offset == PASS_UNUSED) {
      return false;
    }
    values.resize(pixels * channels);
    return read_viewport_preview_pass(*pass, channels, values.data());
  };

  if (!plan->combined) {
    snapshot->combined.clear();
  }
  std::set<ustring> required;
  for (const auto &term : plan->terms) {
    required.insert(term.pass);
  }
  /* A neutral managed light has no Combined correction yet. Keep its raw pass so
   * the first edit and a drag through neutral can both replay this same batch. */
  if (plan->combined) {
    if (const auto program = film_->get_lpe_program()) {
      for (const auto &expression : program->expressions) {
        for (const auto &term : expression.edit_terms) {
          if (term.manager_node && !term.global_pass.empty()) {
            required.insert(ustring(term.global_pass));
          }
        }
      }
    }
  }
  std::erase_if(snapshot->inputs,
                [&](const auto &entry) { return !required.contains(entry.first); });
  if (plan->combined) {
    const BufferPass *pass = params.get_actual_display_pass(PASS_COMBINED, PassMode::NOISY);
    /* Shadow catcher approximation uses a different film conversion. */
    if (!pass || pass->type != PASS_COMBINED || !read(pass, 4, snapshot->combined)) {
      return;
    }
  }
  for (const ustring &name : required) {
    snapshot->inputs.try_emplace(name);
  }
  for (auto &[name, values] : snapshot->inputs) {
    if (!read(params.find_pass(name.string()), 3, values)) {
      return;
    }
  }
  lpe_batch_preview_ = std::move(snapshot);
}

bool PathTrace::display_lpe_batch_preview(const uint64_t revision)
{
  const auto &snapshot = *lpe_batch_preview_;
  if (snapshot.selection != film_->get_lpe_display_output()) {
    return false;
  }
  const auto plan = film_->get_lpe_preview();
  const size_t pixels = size_t(snapshot.size.x) * snapshot.size.y;
  std::vector<float> rgb;
  std::vector<std::pair<const float *, float3>> terms;
  if (plan->evaluate) {
    if (!plan->evaluate(snapshot, rgb) || rgb.size() != pixels * 3) {
      return false;
    }
  }
  else {
    if (plan->combined ? snapshot.combined.empty() : plan->terms.empty()) {
      return false;
    }
    for (const auto &term : plan->terms) {
      const auto it = snapshot.inputs.find(term.pass);
      if (it == snapshot.inputs.end()) {
        return false;
      }
      if (!is_zero(term.weight)) {
        terms.emplace_back(it->second.data(), term.weight);
      }
    }
  }
  std::vector<half4> rgba(pixels);
  parallel_for(int64_t(0), int64_t(pixels), [&](const int64_t i) {
    float4 color = make_float4(0, 0, 0, 1);
    if (!rgb.empty()) {
      color = make_float4(rgb[i * 3], rgb[i * 3 + 1], rgb[i * 3 + 2], 1);
    }
    else {
      if (plan->combined) {
        const float *p = snapshot.combined.data() + i * 4;
        color = make_float4(p[0], p[1], p[2], p[3]);
      }
      for (const auto &[input, weight] : terms) {
        color.x += input[3 * i] * weight.x;
        color.y += input[3 * i + 1] * weight.y;
        color.z += input[3 * i + 2] * weight.z;
      }
    }
    rgba[i] = float4_to_half4_display(color);
  });
  {
    const thread_scoped_lock lock(render_cancel_.mutex);
    if (render_cancel_.is_requested) {
      return false;
    }
  }
  /* During a drag edits can arrive faster than an image can be mixed. Show completed
   * frames in order and consume the newest weights on the next iteration. Dropping every
   * superseded frame here would freeze the preview until mouse release. */
  if (snapshot.selection != film_->get_lpe_display_output() ||
      !display_->update_begin(snapshot.size.x, snapshot.size.y)) {
    return false;
  }
  display_->copy_pixels_to_texture(rgba.data(), 0, 0, snapshot.size.x, snapshot.size.y);
  display_->update_end();
  if (progress_) {
    progress_->set_update();
  }
  if (getenv("LPE_TRACE")) {
    fprintf(stderr, "[LPE] %s batch preview edit=%llu source=%llu samples=%d\n",
            device_scene_->data.integrator.use_photon_caustics ? "photon" : "native",
            (unsigned long long)revision, (unsigned long long)snapshot.revision,
            snapshot.samples);
  }
  return true;
}

void PathTrace::trace_with_lpe_preview(RenderWork &work)
{
  if (!lpe_batch_preview_ || !display_ || !work.path_trace.num_samples ||
      render_scheduler_.is_background()) {
    path_trace(work);
    return;
  }
  if (!lpe_sampling_worker_) {
    lpe_sampling_worker_ = make_unique<DedicatedTaskPool>();
  }
  bool done = false;
  lpe_sampling_worker_->push([&] {
    path_trace(work);
    const thread_scoped_lock lock(lpe_batch_mutex_);
    done = true;
    lpe_batch_condition_.notify_one();
  });
  /* Keep graphics-context ownership on the session thread. Only the sampling worker touches
   * the live film and device queues until it completes. Slider updates coalesce by revision. */
  uint64_t seen = lpe_displayed_revision_;
  thread_scoped_lock lock(lpe_batch_mutex_);
  while (!done) {
    lpe_batch_condition_.wait(lock, [&] {
      return done || lpe_display_revision_.load(std::memory_order_acquire) != seen;
    });
    if (!done) {
      seen = lpe_display_revision_.load(std::memory_order_acquire);
      lock.unlock();
      display_lpe_batch_preview(seen);
      lock.lock();
    }
  }
}

bool PathTrace::set_lpe_navigating(const bool navigating)
{
  const bool previous = lpe_navigating_.exchange(navigating, std::memory_order_relaxed);
  if (navigating || previous != navigating) tag_lpe_interaction();
  const auto plan = film_->get_lpe_preview();
  return previous != navigating &&
         (plan->combined || plan->evaluate || !plan->pass.empty() || !plan->terms.empty());
}

std::chrono::milliseconds PathTrace::lpe_denoise_wait_time() const
{
  if (!lpe_denoise_pending_ || lpe_navigating_.load(std::memory_order_relaxed))
    return std::chrono::milliseconds::max();
  constexpr double quiet_period = 0.25;
  const double remaining = quiet_period -
      (time_dt() - lpe_last_interaction_time_.load(std::memory_order_relaxed));
  return remaining > 0.0 ? std::chrono::milliseconds(int64_t(std::ceil(remaining * 1000.0))) :
                           std::chrono::milliseconds::zero();
}

bool PathTrace::update_lpe_denoised_preview(const RenderWork &render_work)
{
  const auto plan = film_->get_lpe_preview();
  const bool active = plan->combined || plan->evaluate || !plan->terms.empty() || !plan->pass.empty();
  const bool enabled = plan->combined ? lpe_denoise_params_.use : plan->denoise;
  if (!active || !enabled) {
    lpe_denoise_pending_ = false;
    lpe_denoised_plan_.reset();
    lpe_denoised_rgba_.clear();
    lpe_denoiser_.reset();
    return false;
  }
  if (get_num_render_tile_samples() <= 0) {
    lpe_denoise_pending_ = false;
    return false;
  }
  const bool was_pending = lpe_denoise_pending_;
  lpe_denoise_pending_ = true;
  if (lpe_denoise_wait_time() != std::chrono::milliseconds::zero()) {
    if (!was_pending && getenv("LPE_TRACE")) fprintf(stderr, "[LPE] denoise deferred for interaction\n");
    return false;
  }
  lpe_denoise_pending_ = false;
  if (lpe_denoised_plan_ == plan && lpe_denoised_revision_ == lpe_buffer_revision_) return true;
  /* Hold the previous filtered image while native progressive scheduling waits.
   * Explicit display edits bypass only this cadence, never the interaction gate. */
  if (!render_work.tile.lpe_denoise) return lpe_denoised_plan_ == plan;
  const double started = time_dt();
  const auto stale = [this, plan, started]() {
    return render_cancel_.is_requested || film_->get_lpe_preview() != plan ||
           lpe_navigating_.load(std::memory_order_relaxed) ||
           lpe_last_interaction_time_.load(std::memory_order_relaxed) > started;
  };

  const PathTraceTile tile(*this, true);
  const size_t pixels = size_t(tile.size.x) * tile.size.y;
  if (!pixels) return false;
  vector<float> rgba(pixels * 4, 0.0f);
  std::vector<float> rgb(pixels * 3);
  parallel_for(int64_t(0), int64_t(pixels), [&](int64_t i) { rgba[i * 4 + 3] = 1.0f; });
  if (plan->evaluate) {
    if (!plan->evaluate(tile, rgb) || rgb.size() != pixels * 3) return false;
    parallel_for(int64_t(0), int64_t(pixels), [&](int64_t i) {
      for (int c = 0; c < 3; c++) rgba[i * 4 + c] = rgb[i * 3 + c];
    });
  }
  else if (plan->combined || !plan->terms.empty()) {
    if (plan->combined && !tile.get_pass_pixels("Combined", 4, rgba.data())) return false;
    for (const auto &term : plan->terms) {
      if (!tile.get_pass_pixels(term.pass.string(), 3, rgb.data())) return false;
      const float weights[3] = {term.weight.x, term.weight.y, term.weight.z};
      parallel_for(int64_t(0), int64_t(pixels), [&](int64_t i) {
        for (int c = 0; c < 3; c++) rgba[i * 4 + c] += rgb[i * 3 + c] * weights[c];
      });
    }
  }
  else {
    const BufferPass *pass = get_render_tile_params().find_pass(plan->pass.string());
    if (!pass) return false;
    const int channels = pass->get_info().num_components;
    if (channels != 1 && channels != 3) return false;
    if (!tile.get_pass_pixels(plan->pass.string(), channels, rgb.data())) return false;
    parallel_for(int64_t(0), int64_t(pixels), [&](int64_t i) {
      for (int c = 0; c < 3; c++) rgba[i * 4 + c] = rgb[i * channels + (channels == 1 ? 0 : c)];
    });
  }
  if (stale()) return false;

  DenoiseParams params = lpe_denoise_params_;
  params.use = true;
  params.upscale_factor = 1.0f;
  params.temporally_stable = false;
  /* Arbitrary paths do not have matching material guides. Combined can reuse
   * the native guides if they were sampled; a Viewer uses color-only filtering. */
  params.passes = DenoiserPassMask(0);
  vector<float> albedo, normal;
  if (plan->combined) {
    for (const auto &[name, flag, values] : {
             std::tuple{"Denoising Albedo", DENOISER_PASS_ALBEDO, &albedo},
             std::tuple{"Denoising Normal", DENOISER_PASS_NORMAL, &normal}}) {
      if (!(lpe_denoise_params_.passes & flag)) continue;
      values->resize(pixels * 3);
      if (tile.get_pass_pixels(name, 3, values->data())) params.passes |= flag;
      else values->clear();
    }
  }
  const GraphicsInteropDevice interop = display_->graphics_interop_get_device();
  Device *denoise_device = nullptr;
  params = get_effective_denoise_params(denoise_device_, cpu_device_.get(), params, interop,
                                        denoise_device);
  if (!lpe_denoiser_ || lpe_denoiser_->get_params().type != params.type ||
      lpe_denoiser_->get_params().use_gpu != params.use_gpu) {
    lpe_denoiser_ = Denoiser::create(denoise_device, cpu_device_.get(), params, interop);
    if (!lpe_denoiser_ || !lpe_denoiser_->load_kernels(progress_)) return false;
  }
  lpe_denoiser_->set_params(params);
  lpe_denoiser_->is_cancelled_cb = stale;

  BufferParams buffers;
  buffers.width = buffers.window_width = buffers.full_width = tile.size.x;
  buffers.height = buffers.window_height = buffers.full_height = tile.size.y;
  int pass_offset = 0;
  const auto add_pass = [&](PassType type, PassMode mode) {
    BufferPass pass;
    pass.type = type; pass.mode = mode;
    pass.offset = pass_offset;
    pass_offset += pass.get_info().num_components;
    buffers.passes.push_back(pass);
  };
  add_pass(PASS_COMBINED, PassMode::NOISY);
  add_pass(PASS_COMBINED, PassMode::DENOISED);
  if (!albedo.empty()) add_pass(PASS_DENOISING_ALBEDO, PassMode::NOISY);
  if (!normal.empty()) add_pass(PASS_DENOISING_NORMAL, PassMode::NOISY);
  buffers.update_passes();
  buffers.update_offset_stride();
  RenderBuffers scratch(cpu_device_.get());
  scratch.reset(buffers);
  const int color_offset = buffers.get_pass_offset(PASS_COMBINED);
  const int albedo_offset = buffers.get_pass_offset(PASS_DENOISING_ALBEDO);
  const int normal_offset = buffers.get_pass_offset(PASS_DENOISING_NORMAL);
  if (color_offset == PASS_UNUSED || buffers.pass_stride <= 0 ||
      buffers.get_pass_offset(PASS_COMBINED, PassMode::DENOISED) == PASS_UNUSED) return false;
  parallel_for(int64_t(0), int64_t(pixels), [&](int64_t i) {
    float *p = scratch.buffer.data() + i * buffers.pass_stride;
    std::copy_n(rgba.data() + i * 4, 4, p + color_offset);
    if (!albedo.empty()) std::copy_n(albedo.data() + i * 3, 3, p + albedo_offset);
    if (!normal.empty()) std::copy_n(normal.data() + i * 3, 3, p + normal_offset);
  });
  scratch.copy_to_device();
  /* Tile accessors already normalized adaptive sampling and exposure. The
   * scratch image therefore has one sample, independent of the path buffers. */
  const bool success = lpe_denoiser_->denoise_buffer(buffers, buffers, &scratch, 1, true);
  if (!success || !scratch.copy_from_device() || stale()) return false;
  const int denoised_offset = buffers.get_pass_offset(PASS_COMBINED, PassMode::DENOISED);
  parallel_for(int64_t(0), int64_t(pixels), [&](int64_t i) {
    const float *p = scratch.buffer.data() + i * buffers.pass_stride + denoised_offset;
    std::copy_n(p, 3, rgba.data() + i * 4);  // Retain original display alpha.
  });
  lpe_denoised_rgba_ = std::move(rgba);
  lpe_denoised_plan_ = plan;
  lpe_denoised_revision_ = lpe_buffer_revision_;
  if (getenv("LPE_TRACE")) fprintf(stderr,
      "[LPE] post denoise samples=%d combined=%d backend=%s device=%s ms=%.3f\n",
      get_num_render_tile_samples(), int(plan->combined), denoiserTypeToHumanReadable(params.type),
      lpe_denoiser_->get_denoiser_device()->info.description.c_str(), (time_dt() - started) * 1000.0);
  return true;
}

CCL_NAMESPACE_END
