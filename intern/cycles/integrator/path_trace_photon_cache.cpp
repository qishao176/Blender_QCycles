/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: Apache-2.0 */

#include <algorithm>
#include <cstring>

#include "integrator/path_trace.h"
#include "integrator/render_scheduler.h"
#include "scene/lightgroup_pass.h"
#include "scene/lpe.h"
#include "scene/scene.h"
#include "session/full_buffer.h"
#include "util/log.h"
#include "util/math.h"
#include "util/progress.h"
#include "util/scoped_defer.h"
#include "util/tbb.h"

CCL_NAMESPACE_BEGIN

void PathTrace::set_photon_cache_native(const bool enabled)
{
  photon_cache_native_ = enabled;
  for (auto &work : path_trace_works_) {
    work->set_photon_cache_native(enabled);
  }
  if (photon_reduce_work_) {
    photon_reduce_work_->set_photon_cache_native(enabled);
  }
}

void PathTrace::select_photon_lpe_program(const bool photon)
{
  KernelFilm &film = device_scene_->data.film;
  if (!film.lpe_independent_photon)
    return;
  film.lpe_state_offset = photon ? film.lpe_native_count : 0;
  film.lpe_state_count = photon ? film.lpe_photon_count : film.lpe_native_count;
  film.lpe_initial_low = photon ? film.lpe_photon_initial_low : film.lpe_native_initial_low;
  film.lpe_initial_high = photon ? film.lpe_photon_initial_high : film.lpe_native_initial_high;
  film.lpe_shadow_count = photon ? 0 : film.lpe_native_shadow_count;
  device_->const_copy_to("data", &device_scene_->data, sizeof(KernelData));
}

void PathTrace::set_photon_estimator(const int estimator, const bool final_phase)
{
  select_photon_lpe_program(estimator == 1);
  photon_estimator_ = estimator;
  photon_final_phase_ = final_phase;
  for (auto &work : path_trace_works_) {
    work->set_photon_estimator(estimator);
  }
  if (photon_reduce_work_) {
    photon_reduce_work_->set_photon_estimator(estimator);
  }
}

static KernelFilm compact_photon_film(const KernelFilm &source,
                                      const vector<int> &offsets,
                                      const int stride)
{
  KernelFilm film = source;
  const auto remap = [&](const int offset) {
    return offset >= 0 && offset < int(offsets.size()) ? offsets[offset] : PASS_UNUSED;
  };
  film.pass_combined = remap(source.pass_combined);
  film.pass_depth = remap(source.pass_depth);
  film.pass_position = remap(source.pass_position);
  film.pass_normal = remap(source.pass_normal);
  film.pass_roughness = remap(source.pass_roughness);
  film.pass_motion = remap(source.pass_motion);
  film.pass_motion_weight = remap(source.pass_motion_weight);
  film.pass_uv = remap(source.pass_uv);
  film.pass_object_id = remap(source.pass_object_id);
  film.pass_material_id = remap(source.pass_material_id);
  film.pass_diffuse_color = remap(source.pass_diffuse_color);
  film.pass_subsurface_color = remap(source.pass_subsurface_color);
  film.pass_subsurface_direct = remap(source.pass_subsurface_direct);
  film.pass_subsurface_indirect = remap(source.pass_subsurface_indirect);
  film.pass_glossy_color = remap(source.pass_glossy_color);
  film.pass_transmission_color = remap(source.pass_transmission_color);
  film.pass_diffuse_indirect = remap(source.pass_diffuse_indirect);
  film.pass_glossy_indirect = remap(source.pass_glossy_indirect);
  film.pass_transmission_indirect = remap(source.pass_transmission_indirect);
  film.pass_volume_indirect = remap(source.pass_volume_indirect);
  film.pass_diffuse_direct = remap(source.pass_diffuse_direct);
  film.pass_glossy_direct = remap(source.pass_glossy_direct);
  film.pass_transmission_direct = remap(source.pass_transmission_direct);
  film.pass_volume_direct = remap(source.pass_volume_direct);
  film.pass_volume_scatter = remap(source.pass_volume_scatter);
  film.pass_volume_scatter_denoised = remap(source.pass_volume_scatter_denoised);
  film.pass_volume_transmit = remap(source.pass_volume_transmit);
  film.pass_volume_transmit_denoised = remap(source.pass_volume_transmit_denoised);
  film.pass_volume_majorant = remap(source.pass_volume_majorant);
  film.pass_volume_majorant_sample_count = remap(source.pass_volume_majorant_sample_count);
  film.pass_emission = remap(source.pass_emission);
  film.pass_background = remap(source.pass_background);
  film.pass_ao = remap(source.pass_ao);
  film.pass_shadow_catcher = remap(source.pass_shadow_catcher);
  film.pass_shadow_catcher_sample_count = remap(source.pass_shadow_catcher_sample_count);
  film.pass_shadow_catcher_matte = remap(source.pass_shadow_catcher_matte);
  film.pass_render_time = remap(source.pass_render_time);
  film.pass_cryptomatte = remap(source.pass_cryptomatte);
  film.pass_adaptive_aux_buffer = remap(source.pass_adaptive_aux_buffer);
  film.pass_sample_count = remap(source.pass_sample_count);
  film.pass_mist = remap(source.pass_mist);
  film.pass_denoising_albedo = remap(source.pass_denoising_albedo);
  film.pass_denoising_specular_albedo = remap(source.pass_denoising_specular_albedo);
  film.pass_denoising_normal = remap(source.pass_denoising_normal);
  film.pass_denoising_roughness = remap(source.pass_denoising_roughness);
  film.pass_denoising_depth = remap(source.pass_denoising_depth);
  film.pass_denoising_backward_motion = remap(source.pass_denoising_backward_motion);
  film.pass_aov_color = remap(source.pass_aov_color);
  film.pass_aov_value = remap(source.pass_aov_value);
  film.pass_lightgroup = remap(source.pass_lightgroup);
  /* The compact estimator keeps only the photon component. Component fields address
   * lookup rows, whose film offsets are remapped below. */
  for (const auto &split : lightgroup_split_pass_descriptors()) {
    if (split.pass_type != PASS_CAUSTICS) {
      *lightgroup_pass_offset(film, split.pass_type) = PASS_UNUSED;
    }
  }
  film.pass_bake_primitive = remap(source.pass_bake_primitive);
  film.pass_bake_seed = remap(source.pass_bake_seed);
  film.pass_bake_differential = remap(source.pass_bake_differential);
  film.pass_guiding_color = remap(source.pass_guiding_color);
  film.pass_guiding_probability = remap(source.pass_guiding_probability);
  film.pass_guiding_avg_roughness = remap(source.pass_guiding_avg_roughness);
  film.pass_photon_sample_count = remap(source.pass_photon_sample_count);
  film.pass_caustics = remap(source.pass_caustics);
  film.pass_stride = stride;
  film.photon_segment_stride = 0;
  film.photon_compact = 1;
  return film;
}

void PathTrace::render_photon_cache(const BufferParams &params, const string &native_filename)
{
  if (native_filename.empty() || is_cancel_requested() || device_->have_error()) {
    return;
  }
  {
    const thread_scoped_lock lock(render_cancel_.mutex);
    if (render_cancel_.is_requested) {
      return;
    }
    render_cancel_.is_rendering = true;
  }
  SCOPED_DEFER([&]() {
    const thread_scoped_lock lock(render_cancel_.mutex);
    render_cancel_.is_rendering = false;
    render_cancel_.condition.notify_one();
  }());

  const KernelFilm native_film = device_scene_->data.film;
  FullBufferState::PhotonCacheResult result;
  result.native_film = native_film;
  const int segment = result.native_film.photon_segment_stride;
  if (!segment) {
    return;
  }
  result.offsets.resize(segment, PASS_UNUSED);
  result.merge.resize(segment);
  memcpy(result.merge.data(), device_scene_->photon_caustics_merge.data(), segment * sizeof(uint));

  vector<int> zero_lpe_offsets;
  for (size_t i = 0; i < device_scene_->lpe_states.size(); ++i) {
    const KernelLPEState &state = device_scene_->lpe_states.data()[i];
    if (state.flags & LPE_SHADOW) {
      zero_lpe_offsets.push_back(state.pass_offset);
      zero_lpe_offsets.push_back(state.events);
    }
  }
  BufferParams compact = params;
  compact.passes.clear();
  int stride = 0;
  for (const BufferPass &pass : params.passes) {
    /* User-requested channels remain available. Remove the two duplicate estimator
     * films and final denoised outputs, retaining volume-guiding scratch if needed. */
    if (pass.offset == PASS_UNUSED || pass.type == PASS_PHOTON_SAMPLE_COUNT ||
        pass.type == PASS_ADAPTIVE_AUX_BUFFER ||
        (pass.mode == PassMode::DENOISED && pass.type != PASS_VOLUME_SCATTER &&
         pass.type != PASS_VOLUME_TRANSMIT))
    {
      continue;
    }
    BufferPass packed = pass;
    /* Caustics are excluded from ordinary lightgroup lobes and selected-caster
     * shadows. These channels are identically zero in the photon estimator. */
    if ((!pass.lightgroup.empty() && pass.type != PASS_COMBINED && pass.type != PASS_CAUSTICS) ||
        (pass.type == PASS_LPE &&
         std::find(zero_lpe_offsets.begin(), zero_lpe_offsets.end(), pass.offset) !=
             zero_lpe_offsets.end()))
    {
      packed.offset = PASS_UNUSED;
      compact.passes.push_back(packed);
      continue;
    }
    packed.offset = stride;
    for (int c = 0; c < pass.get_info().num_components; ++c) {
      result.offsets[pass.offset + c] = stride++;
    }
    compact.passes.push_back(packed);
  }
  compact.update_passes();
  compact.window_x = compact.window_y = 0;
  compact.window_width = compact.width;
  compact.window_height = compact.height;
  compact.update_offset_stride();

  vector<KernelLPEState> lpe_states(device_scene_->lpe_states.size());
  if (!lpe_states.empty()) {
    memcpy(lpe_states.data(),
           device_scene_->lpe_states.data(),
           lpe_states.size() * sizeof(KernelLPEState));
  }
  auto &device_split_offsets = device_scene_->lightgroup_split_index;
  vector<int> native_split_offsets(native_film.pass_lightgroup_photon != PASS_UNUSED ?
                                       device_split_offsets.size() :
                                       0);
  if (!native_split_offsets.empty()) {
    std::copy_n(
        device_split_offsets.data(), device_split_offsets.size(), native_split_offsets.begin());
  }
  const BufferParams original_tile = big_tile_params_;
  const bool native_mode = photon_cache_native_;
  /* All kernel globals and automaton offsets are restored even after cancellation.
   * The compact host result owns no scene pointers and survives device_free. */
  SCOPED_DEFER([&]() {
    device_free();
    big_tile_params_ = original_tile;
    device_scene_->data.film = native_film;
    if (!native_split_offsets.empty()) {
      std::copy(
          native_split_offsets.begin(), native_split_offsets.end(), device_split_offsets.data());
      device_split_offsets.copy_to_device();
    }
    if (!lpe_states.empty()) {
      memcpy(device_scene_->lpe_states.data(),
             lpe_states.data(),
             lpe_states.size() * sizeof(KernelLPEState));
      device_scene_->lpe_states.copy_to_device();
    }
    device_->const_copy_to("data", &device_scene_->data, sizeof(KernelData));
    set_photon_cache_native(native_mode);
    photon_preview_native_passes_.clear();
  }());

  device_free();
  device_scene_->data.film = compact_photon_film(
      result.native_film, result.offsets, compact.pass_stride);
  if (!native_split_offsets.empty()) {
    for (size_t i = 0; i < native_split_offsets.size(); i++) {
      const int offset = native_split_offsets[i];
      device_split_offsets.data()[i] = offset == PASS_UNUSED ? PASS_UNUSED : result.offsets[offset];
    }
    device_split_offsets.copy_to_device();
  }
  for (size_t i = 0; i < lpe_states.size(); ++i) {
    KernelLPEState &state = device_scene_->lpe_states.data()[i];
    if (state.pass_offset != PASS_UNUSED) {
      state.pass_offset = result.offsets[state.pass_offset];
    }
    if ((state.flags & LPE_SHADOW) && state.events != PASS_UNUSED) {
      state.events = result.offsets[state.events];
    }
  }
  if (!lpe_states.empty()) {
    device_scene_->lpe_states.copy_to_device();
  }
  device_->const_copy_to("data", &device_scene_->data, sizeof(KernelData));
  select_photon_lpe_program(true);
  set_photon_cache_native(false);
  big_tile_params_ = compact;
  render_state_.need_reset_params = true;

  const int sample_offset = render_scheduler_.get_sample_offset();
  /* The native subset may be shorter than the photon budget. Preserve its requested
   * extent through params.samples, supplied by Session specifically for this phase. */
  render_scheduler_.set_sample_params(
      params.samples, true, sample_offset, params.samples - sample_offset);
  render_scheduler_.reset(compact);
  render_scheduler_.set_limit_samples_per_update(0);
  LOG_INFO << "Photon cache: " << compact.width << "x" << compact.height
           << ", compact floats/pixel " << compact.pass_stride << ", tiled floats/pixel "
           << params.pass_stride << ", replicas " << path_trace_works_.size();
  progress_->set_status("Rendering Photon Cache");

  bool initialized = false;
  while (!is_cancel_requested() && !device_->have_error()) {
    RenderWork work = render_scheduler_.get_render_work();
    if (!work) {
      break;
    }
    if (!work.path_trace.num_samples) {
      update_photon_cache_display(work, work.tile.write);
      continue;
    }
    render_scheduler_.report_work_begin(work);
    init_render_buffers(work);
    initialized = true;
    render_init_kernel_execution();
    render_scheduler_.set_limit_samples_per_update(0);
    const bool use_guiding = device_scene_->data.integrator.use_guiding;
    if (use_guiding) {
      guiding_prepare_structures();
    }
    if (device_scene_->data.integrator.use_volumes) {
      const uint samples = render_scheduler_.get_num_rendered_samples();
      render_scheduler_.set_limit_samples_per_update(next_power_of_two(samples) - samples);
    }
    {
      SCOPED_DEFER(render_deinit_kernel_execution());
      /* The existing replica barrier freezes this epoch for all devices. No tree,
       * photon map or partial training update is exchanged between replicas here. */
      path_trace(work);
      if (is_cancel_requested() || device_->have_error()) {
        break;
      }
      if (use_guiding && device_scene_->data.integrator.train_guiding) {
        guiding_update_structures();
      }
      adaptive_sample(work);
      /* The tiled independent photon estimator reads its raw segment, whose volume
       * guide scratch remains zero. Keep the same proposal here: filtering this
       * compact raw film would silently enable a different photon guide. */
    }
    completed_buffer_samples_ = render_scheduler_.get_num_rendered_samples();
    update_photon_cache_display(work, work.tile.write);
    progress_update_if_needed(work);
    progress_->set_status("Rendering Photon Cache",
                          "Sample " + to_string(render_scheduler_.get_num_rendered_samples()) +
                              "/" + to_string(render_scheduler_.get_num_samples()));
    progress_->set_update();
  }
  if (!initialized || is_cancel_requested() || device_->have_error()) {
    return;
  }
  update_photon_cache_display(RenderWork(), true);
  PathTraceWork *aggregate = photon_replicas_ ? photon_reduce_work_.get() :
                                                path_trace_works_.front().get();
  if (device_scene_->data.film.cryptomatte_passes) {
    aggregate->init_execution();
    aggregate->cryptomatte_postproces();
    aggregate->deinit_execution();
  }
  if (!aggregate->copy_render_buffers_from_device()) {
    device_->set_error("Failed to read full-frame photon cache");
    return;
  }
  result.params = compact;
  result.pixels.resize(aggregate->get_render_buffers()->buffer.size());
  memcpy(result.pixels.data(),
         aggregate->get_render_buffers()->buffer.data(),
         result.pixels.size() * sizeof(float));
  full_buffer_state_->photons[native_filename] = std::move(result);
}

void PathTrace::apply_photon_cache(const string_view filename, RenderBuffers &native)
{
  const auto found = full_buffer_state_->photons.find(string(filename));
  if (found == full_buffer_state_->photons.end()) {
    return;
  }
  const FullBufferState::PhotonCacheResult &cache = found->second;
  const BufferParams &source = cache.params;
  const KernelFilm &film = cache.native_film;
  if (source.width != native.params.width || source.height != native.params.height ||
      film.pass_stride != native.params.pass_stride)
  {
    if (progress_) {
      progress_->set_error("Photon cache does not match native image dimensions or passes");
    }
    return;
  }
  const int segment = film.photon_segment_stride;
  const size_t pixels = size_t(source.width) * source.height;
  parallel_for(size_t(0), pixels, [&](const size_t pixel) {
    const float *compact = cache.pixels.data() + pixel * source.pass_stride;
    float *output = native.buffer.data() + pixel * native.params.pass_stride;
    float *raw_native = output + segment;
    float *raw_photon = raw_native + segment;
    for (int i = 0; i < segment; ++i) {
      if (cache.offsets[i] != PASS_UNUSED) {
        memcpy(raw_photon + i, compact + cache.offsets[i], sizeof(float));
      }
      else {
        raw_photon[i] = 0.0f;
      }
    }
    uint native_count, photon_count;
    memcpy(&native_count, raw_native + film.pass_sample_count, sizeof(uint));
    memcpy(&photon_count, raw_photon + film.pass_sample_count, sizeof(uint));
    const uint output_count = native_count ? native_count : photon_count;
    const float scale = photon_count ? float(output_count) / photon_count : 0.0f;
    const float *metadata = native_count ? raw_native : raw_photon;
    for (int i = 0; i < segment; ++i) {
      if (cache.merge[i] != 2) {
        output[i] = cache.merge[i] == 1 ? raw_native[i] + raw_photon[i] * scale : metadata[i];
      }
    }
    memcpy(output + film.pass_sample_count, &output_count, sizeof(uint));
    memcpy(output + film.pass_photon_sample_count, &photon_count, sizeof(uint));
  });
  full_buffer_state_->photons.erase(found);
}

CCL_NAMESPACE_END
