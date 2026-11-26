/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#include "integrator/path_trace.h"
#include "integrator/path_trace_work_gpu.h"
#include "util/lpe_query.h"
#include "device/photon.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "device/cpu/device.h"
#include "device/device.h"

#include "integrator/pass_accessor.h"
#include "integrator/path_trace_display.h"
#include "integrator/path_trace_tile.h"
#include "integrator/render_scheduler.h"

#include "scene/lpe.h"
#include "scene/pass.h"
#include "scene/scene.h"

#include "session/display_driver.h"
#include "session/full_buffer.h"
#include "session/tile.h"

#include "util/log.h"
#include "util/progress.h"
#include "util/scoped_defer.h"
#include "util/task.h"
#include "util/tbb.h"
#include "util/time.h"

CCL_NAMESPACE_BEGIN

namespace {

uint32_t pass_float_as_uint(const float value)
{
  uint32_t result;
  memcpy(&result, &value, sizeof(result));
  return result;
}

float combined_transparency_to_alpha(const float transparency)
{
  return min(max(1.0f - transparency, 0.0f), 1.0f);
}

}  // namespace

PathTrace::PathTrace(Device *device,
                     Device *denoise_device,
                     Film *film,
                     DeviceScene *device_scene,
                     RenderScheduler &render_scheduler,
                     TileManager &tile_manager)
    : device_(device),
      denoise_device_(denoise_device),
      film_(film),
      device_scene_(device_scene),
      render_scheduler_(render_scheduler),
      tile_manager_(tile_manager)
{
  DCHECK_NE(device_, nullptr);

  {
    vector<DeviceInfo> cpu_devices;
    device_cpu_info(cpu_devices);

    cpu_device_ = device_cpu_create(
        cpu_devices[0], device->stats, device->profiler, device_->headless);
  }

  /* Create path tracing work in advance, so that it can be reused by incremental sampling as much
   * as possible. */
  device_->foreach_device([&](Device *path_trace_device) {
    unique_ptr<PathTraceWork> work = PathTraceWork::create(
        path_trace_device, film, device_scene, &render_cancel_.is_requested);
    if (work) {
      path_trace_works_.emplace_back(std::move(work));
    }
  });

  if (const char *directory = getenv("CYCLES_VCM_CAPTURE")) {
    vcm_capture_directory_ = directory;
  }
  photon_test_replicas_ = photon_test_two_devices(device_->info);
  if (photon_test_replicas_) {
    path_trace_works_.emplace_back(PathTraceWork::create(path_trace_works_.front()->get_device(),
                                                         film,
                                                         device_scene,
                                                         &render_cancel_.is_requested));
  }
  for (size_t i = 0; i < path_trace_works_.size(); ++i) {
    path_trace_works_[i]->viewport_photon_probe = !render_scheduler_.is_background();
    if (path_trace_works_.size() > 1) {
      path_trace_works_[i]->set_photon_replica(uint(i));
    }
  }
  /* Isolated reference for the second simulated replica, used by the energy regression. */
  const char *test_replica = getenv("CYCLES_PHOTON_TEST_REPLICA");
  if (path_trace_works_.size() == 1 &&
      (device_->info.type == DEVICE_OPTIX || device_->info.type == DEVICE_CPU) && test_replica &&
      strcmp(test_replica, "1") == 0)
  {
    path_trace_works_.front()->set_photon_replica(1);
  }

  work_balance_infos_.resize(path_trace_works_.size());
  work_balance_do_initial(work_balance_infos_);

  render_scheduler.set_need_schedule_rebalance(path_trace_works_.size() > 1);
  full_buffer_state_ = std::make_shared<FullBufferState>();
}

PathTrace::~PathTrace()
{
  clear_photon_preview_tiles();
  destroy_gpu_resources();
}

bool PathTrace::supports_gpu_query() const
{
  for (const auto &work : path_trace_works_) {
    if (work->get_device()->info.type == DEVICE_OPTIX || work->get_device()->info.type == DEVICE_CUDA) {
      return static_cast<const PathTraceWorkGPU *>(work.get())->supports_gpu_query();
    }
  }
  return false;
}

void PathTrace::release_query_workspace(const uint64_t stream)
{
  for (const auto &work : path_trace_works_) {
    if (work->get_device()->info.type == DEVICE_OPTIX || work->get_device()->info.type == DEVICE_CUDA) {
      static_cast<PathTraceWorkGPU *>(work.get())->release_query_workspace(stream);
    }
  }
}

void PathTrace::query_slice(const QuerySlice &slice, QuerySliceResult &result)
{
  for (const auto &work : path_trace_works_) {
    if (work->get_device()->info.type == DEVICE_OPTIX || work->get_device()->info.type == DEVICE_CUDA) {
      static_cast<PathTraceWorkGPU *>(work.get())->query_slice(slice, result);
      return;
    }
  }
  result.error = "No compatible NVIDIA viewport";
}

void PathTrace::load_kernels()
{
  if (denoiser_) {
    /* Activate graphics interop while denoiser device is created, so that it can choose a device
     * that supports interop for faster display updates. */
    if (display_ && path_trace_works_.size() > 1) {
      display_->graphics_interop_activate();
    }

    denoiser_->load_kernels(progress_);

    if (display_ && path_trace_works_.size() > 1) {
      display_->graphics_interop_deactivate();
    }
  }
}

void PathTrace::alloc_work_memory()
{
  for (auto &path_trace_work : path_trace_works_) {
    path_trace_work->alloc_work_memory();
  }
}

bool PathTrace::ready_to_reset()
{
  /* The logic here is optimized for the best feedback in the viewport, which implies having a GPU
   * display. Of there is no such display, the logic here will break. */
  DCHECK(display_);

  /* The logic here tries to provide behavior which feels the most interactive feel to artists.
   * General idea is to be able to reset as quickly as possible, while still providing interactive
   * feel.
   *
   * If the render result was ever drawn after previous reset, consider that reset is now possible.
   * This way camera navigation gives the quickest feedback of rendered pixels, regardless of
   * whether CPU or GPU drawing pipeline is used.
   *
   * Consider reset happening after redraw "slow" enough to not clog anything. This is a bit
   * arbitrary, but seems to work very well with viewport navigation in Blender. */

  if (did_draw_after_reset_) {
    return true;
  }

  return false;
}

void PathTrace::reset(const BufferParams &full_params,
                      const BufferParams &big_tile_params,
                      const bool reset_rendering)
{
  if (reset_rendering && !render_scheduler_.is_background()) {
    const thread_scoped_lock lock(photon_probe_mutex_);
    published_photon_probe_ = {};
    for (auto &work : path_trace_works_) {
      work->photon_probe = {};
    }
  }
  if (big_tile_params_.modified(big_tile_params)) {
    big_tile_params_ = big_tile_params;
    render_state_.need_reset_params = true;
  }

  ++lpe_buffer_revision_;
  lpe_batch_preview_.reset();
  completed_buffer_samples_ = 0;
  tag_lpe_interaction();
  lpe_denoised_plan_.reset();
  lpe_denoise_pending_ = false;
  full_params_ = full_params;

  /* NOTE: GPU display checks for buffer modification and avoids unnecessary re-allocation.
   * It is requires to inform about reset whenever it happens, so that the redraw state tracking is
   * properly updated. */
  if (display_) {
    if (!photon_cache_native_) {
      display_->reset(big_tile_params, reset_rendering);
    }
    else if (reset_rendering) {
      display_->reset(full_params, true);
    }
  }

  render_state_.has_denoised_result = false;
  render_state_.tile_written = false;

  did_draw_after_reset_ = false;
}

void PathTrace::device_free()
{
  release_query_workspace();
  /* Free render buffers used by the path trace work to reduce memory peak. */
  BufferParams empty_params;
  empty_params.pass_stride = 0;
  empty_params.update_offset_stride();
  for (auto &&path_trace_work : path_trace_works_) {
    path_trace_work->get_render_buffers()->reset(empty_params);
  }
  if (photon_reduce_work_) {
    photon_reduce_work_->get_render_buffers()->reset(empty_params);
  }
  render_state_.need_reset_params = true;
}

PhotonViewportProbe PathTrace::get_photon_viewport_probe() const
{
  const thread_scoped_lock lock(photon_probe_mutex_);
  return published_photon_probe_;
}

void PathTrace::set_progress(Progress *progress)
{
  progress_ = progress;
}

void PathTrace::render(const RenderWork &render_work)
{
  /* Indicate that rendering has started and that it can be requested to cancel. */
  {
    const thread_scoped_lock lock(render_cancel_.mutex);
    if (render_cancel_.is_requested) {
      return;
    }
    render_cancel_.is_rendering = true;
  }

  render_pipeline(render_work);

  /* Indicate that rendering has finished, making it so thread which requested `cancel()` can carry
   * on. */
  {
    const thread_scoped_lock lock(render_cancel_.mutex);
    render_cancel_.is_rendering = false;
    render_cancel_.condition.notify_one();
  }
}

void PathTrace::render_pipeline(RenderWork render_work)
{
  /* NOTE: Only check for "instant" cancel here. The user-requested cancel via progress is
   * checked in Session and the work in the event of cancel is to be finished here. */

  render_scheduler_.set_need_schedule_cryptomatte(device_scene_->data.film.cryptomatte_passes !=
                                                  0);

  render_scheduler_.report_work_begin(render_work);

  init_render_buffers(render_work);

  if (!vcm_paths_) {
    rebalance(render_work);
  }
  sync_deep_output_buffers();

  /* Initialize kernel execution after deep buffer sync so that CPU thread-local kernel globals
   * have the updated deep buffer pointers. This is critical for mixed CPU/GPU rendering where
   * rebalance can change slice layouts and deep buffer allocations. Keep the matching deinit
   * scoped so early returns unwind cleanly. */
  render_init_kernel_execution();
  SCOPED_DEFER(render_deinit_kernel_execution());

  if (!vcm_paths_ && !vcm_capture_directory_.empty() && !vcm_capture_complete_ &&
      render_work.path_trace.num_samples > 0) {
    if (!render_vcm()) {
      return;
    }
    vcm_capture_complete_ = true;
  }

  /* Reset sample limit. */
  render_scheduler_.set_limit_samples_per_update(0);

  /* Prepare all per-thread guiding structures before we start with the next rendering
   * iteration/progression. */
  const bool use_guiding = device_scene_->data.integrator.use_guiding;
  if (use_guiding) {
    guiding_prepare_structures();
  }

  const bool has_volume = device_scene_->data.integrator.use_volumes;
  if (has_volume) {
    const uint num_rendered_samples = render_scheduler_.get_num_rendered_samples();
    const uint limit = next_power_of_two(num_rendered_samples) - num_rendered_samples;
    render_scheduler_.set_limit_samples_per_update(limit);
  }

  if (vcm_paths_) {
    if (render_work.path_trace.num_samples) {
      const double start = time_dt();
      const uint first = render_work.path_trace.start_sample;
      const uint count = render_work.path_trace.num_samples;
      render_scheduler_.set_limit_samples_per_update(render_scheduler_.is_background() ? 16 : 1);
      if (!render_vcm(first, count) ||
          !write_vcm_tile(first + count - render_work.path_trace.sample_offset)) {
        return;
      }
      render_scheduler_.report_path_trace_time(render_work, time_dt() - start, false);
    }
  }
  else {
    trace_with_lpe_preview(render_work);
  }
  /* Publish a replica batch only when all devices completed their snapshots. */
  if (render_cancel_.is_requested ||
      (photon_replicas_ && (is_cancel_requested() || device_->have_error())))
  {
    return;
  }
  completed_buffer_samples_ = render_scheduler_.get_num_rendered_samples();
  if (!render_scheduler_.is_background() && device_scene_->data.integrator.use_photon_caustics) {
    PhotonViewportProbe result;
    for (const auto &work : path_trace_works_) {
      const PhotonViewportProbe &p = work->photon_probe;
      if (!p.maps) {
        continue;
      }
      result.maps = max(result.maps, p.maps);
      result.device_maps += p.maps;
      result.volume_stored += p.volume_stored;
      result.completed = max(result.completed, p.completed);
      result.requested = p.requested;
      result.capacity = result.capacity ? min(result.capacity, p.capacity) : p.capacity;
      result.max_emission = result.max_emission ? min(result.max_emission, p.max_emission) :
                                                  p.max_emission;
      result.interval = p.interval;
      result.sample_limit = p.sample_limit;
      result.has_volumes |= p.has_volumes;
      result.deposit_cv = std::max(result.deposit_cv, p.deposit_cv);
      result.emitted += p.emitted;
      result.stored += p.stored;
      result.sampled += p.sampled;
      result.isolated += p.isolated;
      result.wider_helped += p.wider_helped;
      result.in_frame += p.in_frame;
      result.subpixel += p.subpixel;
      result.radius_limited += p.radius_limited;
      result.pixel_scale_sum += p.pixel_scale_sum;
      result.radius = p.radius;
      result.max_scale = p.max_scale;
    }
    const thread_scoped_lock lock(photon_probe_mutex_);
    published_photon_probe_ = result;
  }

  /* Update the guiding field using the training data/samples collected during the rendering
   * iteration/progression. */
  const bool train_guiding = device_scene_->data.integrator.train_guiding;
  if (use_guiding && train_guiding) {
    guiding_update_structures();
  }

  /* Pairs of independent global rounds estimate the noise. For sparse emission budgets
   * wait for camera coverage before treating repeated black observations as convergence. */
  const bool sufficient_vcm_coverage =
      !vcm_paths_ || double(completed_buffer_samples_) * vcm_paths_ >=
                         4.0 * device_scene_->data.cam.width * device_scene_->data.cam.height;
  if (sufficient_vcm_coverage) {
    adaptive_sample(render_work);
  }
  if (render_cancel_.is_requested) {
    return;
  }

  if (!photon_final_phase_) {
    /* Preserve both raw estimators across the per-tile phase boundary. Only
     * the second phase may finalize, denoise or publish the tile file. */
    render_work.tile.write = false;
    render_work.full.write = false;
    render_work.tile.denoise = false;
    render_work.tile.lpe_denoise = false;
    render_work.cryptomatte.postprocess = false;
    render_work.display.use_denoised_result = false;
  }

  cryptomatte_postprocess(render_work);
  if (render_cancel_.is_requested) {
    return;
  }

  /* A rejected scene update can schedule a final display-only work item before
   * film constants and buffers are uploaded. Never launch a merge against that
   * incomplete layout, or an already failed device context. */
  if (completed_buffer_samples_ != 0 && !(progress_ && progress_->get_error()) &&
      !device_->have_error())
  {
    if (photon_replicas_) {
      photon_reduce_work_->merge_photon_estimates();
    }
    else {
      parallel_for_each(path_trace_works_,
                        [&](unique_ptr<PathTraceWork> &work) { work->merge_photon_estimates(); });
    }
  }

  denoise(render_work);
  if (render_cancel_.is_requested) {
    return;
  }

  denoise_volume_guiding_buffers(render_work, has_volume);
  if (render_cancel_.is_requested) {
    return;
  }

  write_tile_buffer(render_work);
  ++lpe_buffer_revision_;
  const bool viewport_batch = display_ && !render_scheduler_.is_background() &&
                              render_work.path_trace.num_samples && completed_buffer_samples_;
  if (viewport_batch && device_scene_->data.integrator.use_photon_caustics &&
      film_->get_lpe_preview()->evaluate) {
    /* Nonlinear edits replay the evaluator's normalized input cache. Publish each complete
     * sampling batch so that its tile revision and the cached inputs always agree. */
    render_work.display.update = true;
  }
  const bool capture_preview = viewport_batch &&
                               (render_work.display.update || !lpe_batch_preview_ ||
                                film_->display_pass_name_changed() ||
                                lpe_display_revision_.load(std::memory_order_acquire) !=
                                    lpe_displayed_revision_);
  update_display(render_work);
  if (capture_preview) {
    /* The sampling batch has joined before its display snapshot is replaced here. */
    capture_lpe_batch_preview();
  }

  progress_update_if_needed(render_work);

  finalize_full_buffer_on_disk(render_work);
}

void PathTrace::render_init_kernel_execution()
{
  for (auto &&path_trace_work : path_trace_works_) {
    path_trace_work->init_execution();
  }
}

void PathTrace::render_deinit_kernel_execution()
{
  for (auto &&path_trace_work : path_trace_works_) {
    path_trace_work->deinit_execution();
  }
}

/* Scale down an Y coordinate for slicing the buffer. */
static int scale_window_y(const int window_y,
                          const int window_height,
                          const int scaled_window_height,
                          const float resolution_divider)
{
  if (window_y >= window_height) {
    return scaled_window_height;
  }

  return min(int(window_y / resolution_divider), scaled_window_height);
}

/*
 * Get the parameters of the horizontal slice of the buffer, for a given work index and
 * balancing weights. Note the result can be empty with many devices and small height.
 *
 * The split is computed at the full resolution, and then scaled down. This keeps the slices
 * filling the window exactly at every resolution and keeps a slice that is empty at full
 * resolution empty at the downscaled resolution. Slicing at different resolutions with
 * float weights can lead to mismatches.
 */
static BufferParams slice_buffer_params(const BufferParams &buffer_params,
                                        const float resolution_divider,
                                        const int full_res_window_height,
                                        const vector<WorkBalanceInfo> &work_balance_infos,
                                        const int index,
                                        const int overscan)
{
  const int num_works = work_balance_infos.size();

  /* Find the Y coordinates at which the slice starts and ends, at the full resolution. */
  int full_res_window_y = 0;
  int full_res_window_end_y = 0;

  for (int i = 0; i <= index; i++) {
    full_res_window_y = full_res_window_end_y;

    const double weight = work_balance_infos[i].weight;
    const int slice_window_height = max(lround(full_res_window_height * weight), 1);

    const int remaining_window_height = max(0, full_res_window_height - full_res_window_y);
    full_res_window_end_y = full_res_window_y +
                            ((i < num_works - 1) ?
                                 min(slice_window_height, remaining_window_height) :
                                 remaining_window_height);
  }

  /* Scale the Y coordinates down to the resolution of the buffer. */
  const int window_y = scale_window_y(
      full_res_window_y, full_res_window_height, buffer_params.window_height, resolution_divider);
  const int window_end_y = scale_window_y(full_res_window_end_y,
                                          full_res_window_height,
                                          buffer_params.window_height,
                                          resolution_divider);

  const int slice_window_full_y = buffer_params.full_y + buffer_params.window_y + window_y;

  BufferParams slice_params = buffer_params;

  slice_params.full_y = max(slice_window_full_y - overscan, buffer_params.full_y);
  slice_params.window_y = slice_window_full_y - slice_params.full_y;
  slice_params.window_height = window_end_y - window_y;

  slice_params.height = slice_params.window_y + slice_params.window_height + overscan;
  slice_params.height = min(slice_params.height,
                            buffer_params.height + buffer_params.full_y - slice_params.full_y);

  slice_params.update_offset_stride();

  return slice_params;
}

void PathTrace::update_allocated_work_buffer_params()
{
  if (photon_replicas_) {
    for (auto &work : path_trace_works_) {
      work->get_render_buffers()->reset(big_tile_params_);
    }
    photon_reduce_work_->get_render_buffers()->reset(big_tile_params_);
    return;
  }
  const int overscan = tile_manager_.get_tile_overscan();

  const int num_works = path_trace_works_.size();
  for (int i = 0; i < num_works; ++i) {
    const BufferParams slice_params = slice_buffer_params(
        big_tile_params_, 1.0f, big_tile_params_.window_height, work_balance_infos_, i, overscan);

    RenderBuffers *buffers = path_trace_works_[i]->get_render_buffers();
    buffers->reset(slice_params);
  }
}

static BufferParams scale_buffer_params(const BufferParams &params, const float resolution_divider)
{
  BufferParams scaled_params = params;

  scaled_params.window_x = int(params.window_x / resolution_divider);
  scaled_params.window_y = int(params.window_y / resolution_divider);
  scaled_params.window_width = max(1, int(params.window_width / resolution_divider));
  scaled_params.window_height = max(1, int(params.window_height / resolution_divider));

  /* Scaled down buffer size, but not smaller than the window that we need to write into. */
  scaled_params.width = max(int(params.width / resolution_divider),
                            scaled_params.window_x + scaled_params.window_width);
  scaled_params.height = max(int(params.height / resolution_divider),
                             scaled_params.window_y + scaled_params.window_height);

  scaled_params.full_x = int(params.full_x / resolution_divider);
  scaled_params.full_y = int(params.full_y / resolution_divider);
  scaled_params.full_width = max(1, int(params.full_width / resolution_divider));
  scaled_params.full_height = max(1, int(params.full_height / resolution_divider));

  scaled_params.update_offset_stride();

  return scaled_params;
}

void PathTrace::update_effective_work_buffer_params(const RenderWork &render_work)
{
  const float denoised_resolution_divider = render_work.denoised_resolution_divider;
  const float resolution_divider = render_work.resolution_divider / denoised_resolution_divider;

  const BufferParams denoised_big_tile_params = scale_buffer_params(big_tile_params_,
                                                                    denoised_resolution_divider);
  const BufferParams scaled_big_tile_params = scale_buffer_params(denoised_big_tile_params,
                                                                  resolution_divider);

  const int overscan = tile_manager_.get_tile_overscan();
  const int scaled_overscan = int(overscan / resolution_divider);
  const int full_res_window_height = big_tile_params_.window_height;

  const int num_works = path_trace_works_.size();
  for (int i = 0; i < num_works; ++i) {
    const BufferParams denoised_slice_params = slice_buffer_params(denoised_big_tile_params,
                                                                   denoised_resolution_divider,
                                                                   full_res_window_height,
                                                                   work_balance_infos_,
                                                                   i,
                                                                   overscan);

    const BufferParams slice_params = slice_buffer_params(scaled_big_tile_params,
                                                          render_work.resolution_divider,
                                                          full_res_window_height,
                                                          work_balance_infos_,
                                                          i,
                                                          scaled_overscan);

    path_trace_works_[i]->set_effective_buffer_params(
        scaled_big_tile_params, slice_params, denoised_big_tile_params, denoised_slice_params);
  }

  if (photon_replicas_) {
    for (auto &work : path_trace_works_) {
      work->set_effective_buffer_params(scaled_big_tile_params, scaled_big_tile_params,
                                        denoised_big_tile_params, denoised_big_tile_params);
    }
    photon_reduce_work_->set_effective_buffer_params(scaled_big_tile_params, scaled_big_tile_params,
                                                     denoised_big_tile_params, denoised_big_tile_params);
  }

  render_state_.effective_big_tile_params = scaled_big_tile_params;
  render_state_.effective_denoised_big_tile_params = denoised_big_tile_params;
}

void PathTrace::update_work_buffer_params_if_needed(const RenderWork &render_work)
{
  if (render_state_.need_reset_params) {
    update_allocated_work_buffer_params();
  }

  if (render_state_.need_reset_params ||
      render_state_.resolution_divider != render_work.resolution_divider)
  {
    update_effective_work_buffer_params(render_work);
  }

  render_state_.resolution_divider = render_work.resolution_divider;
  render_state_.need_reset_params = false;
}

void PathTrace::init_render_buffers(const RenderWork &render_work)
{
  configure_photon_replicas();
  update_work_buffer_params_if_needed(render_work);

  /* Handle initialization scheduled by the render scheduler. */
  if (render_work.init_render_buffers) {
    completed_buffer_samples_ = 0;
    parallel_for_each(path_trace_works_, [&](unique_ptr<PathTraceWork> &path_trace_work) {
      path_trace_work->zero_render_buffers();
    });

    if (photon_replicas_) {
      photon_reduce_work_->zero_render_buffers();
    }
    tile_buffer_read();
  }
}

void PathTrace::path_trace(RenderWork &render_work)
{
  if (!render_work.path_trace.num_samples) {
    return;
  }

  LOG_DEBUG << "Will path trace " << render_work.path_trace.num_samples
            << " samples at the resolution divider " << render_work.resolution_divider;

  const double start_time = time_dt();

  const int num_works = path_trace_works_.size();

  thread_capture_fp_settings();

  const SamplingEpoch epoch{
      device_scene_->data.film.photon_compact          ? SamplingDomain::PHOTON_FULL_FRAME :
      (photon_cache_native_ || photon_estimator_ == 0) ? SamplingDomain::NATIVE_TILE :
      photon_estimator_ == 1                           ? SamplingDomain::PHOTON_TILE :
                                                         SamplingDomain::COUPLED_TILE,
      render_work.path_trace.start_sample,
      render_work.path_trace.num_samples,
      num_works,
      make_int4(big_tile_params_.full_x,
                big_tile_params_.full_y,
                big_tile_params_.width,
                big_tile_params_.height),
      device_scene_->photon_caustics_revision};
  if (sampling_epoch_begin_cb) {
    sampling_epoch_begin_cb(epoch);
  }

  if (photon_replicas_) {
    auto &buffer = photon_reduce_work_->get_render_buffers()->buffer;
    std::fill_n(buffer.data(), buffer.size(), 0.0f);
  }
  const auto trace_work = [&](int i) {
    const double work_start_time = time_dt();
    const int num_samples = render_work.path_trace.num_samples;

    PathTraceWork *path_trace_work = path_trace_works_[i].get();
    if (path_trace_work->get_device()->have_error()) {
      return;
    }

    if (photon_test_replicas_) {
      path_trace_work->init_execution();
    }
    PathTraceWork::RenderStatistics statistics;
    if (device_scene_->data.integrator.use_photon_caustics) {
      /* A photon map is immutable while camera paths query it. Split only photon rendering at
       * absolute sample boundaries, independently of scheduler batching, tiles and sample subsets.
       * Each new round replaces the map; the film averages its estimates through camera samples.
       * There must be no camera paths in flight while the photon prepass borrows their SoA slots.
       */
      const int interval = max(device_scene_->data.integrator.photon_caustics_update_interval, 1);
      float occupancy_sum = 0.0f;
      int rendered_samples = 0;
      while (rendered_samples < num_samples && !is_cancel_requested() &&
             !path_trace_work->get_device()->have_error())
      {
        /* start_sample already includes sample_offset. */
        const int sample = render_work.path_trace.start_sample + rendered_samples;
        const int chunk_samples = min(interval - sample % interval,
                                      num_samples - rendered_samples);
        PathTraceWork::RenderStatistics chunk_statistics;
        path_trace_work->render_estimator_samples(
            chunk_statistics, sample, chunk_samples, render_work.path_trace.sample_offset);
        occupancy_sum += chunk_statistics.occupancy * chunk_samples;
        rendered_samples += chunk_samples;
      }
      statistics.occupancy = rendered_samples ? occupancy_sum / rendered_samples : 0.0f;
    }
    else {
      path_trace_work->render_estimator_samples(statistics,
                                                render_work.path_trace.start_sample,
                                                num_samples,
                                                render_work.path_trace.sample_offset);
    }

    DCHECK(isfinite(statistics.occupancy));

    const double work_end_time = time_dt();
    const double work_time = work_end_time - work_start_time;
    if (photon_replicas_ && photon_test_hybrid(device_->info)) {
      const char *label = getenv("CYCLES_PHOTON_TEST_LABEL");
      fprintf(stderr,
              "PHOTON_HYBRID case=%s replica=%d backend=%s seed=%u sample=%d samples=%d "
              "begin=%.9f end=%.9f phase=%s\n",
              label ? label : "render",
              i,
              path_trace_work->get_device()->info.type == DEVICE_CPU ? "CPU" : "OPTIX",
              path_trace_work->photon_sampling_seed(),
              render_work.path_trace.start_sample,
              num_samples,
              work_start_time - start_time,
              work_end_time - start_time,
              device_scene_->data.film.photon_compact ? "photon_cache" :
              photon_cache_native_                    ? "native_tiles" :
              photon_estimator_ == 0                  ? "native_phase" :
              photon_estimator_ == 1                  ? "photon_phase" :
                                                        "coupled");
    }
    work_balance_infos_[i].time_spent += work_time;
    work_balance_infos_[i].occupancy = statistics.occupancy;

    LOG_INFO << "Rendered " << num_samples << " samples in " << work_time << " seconds ("
             << work_time / num_samples
             << " seconds per sample), occupancy: " << statistics.occupancy;
    if (photon_replicas_ && !is_cancel_requested() && !path_trace_work->get_device()->have_error())
    {
      merge_photon_replica(*path_trace_work, i);
    }
  };
  if (photon_test_replicas_) {
    for (int i = 0; i < num_works; ++i) {
      trace_work(i);
    }
  }
  else {
    parallel_for(0, num_works, trace_work);
  }
  if (photon_replicas_ && !is_cancel_requested() && !device_->have_error()) {
    photon_reduce_work_->init_execution();
    photon_reduce_work_->copy_render_buffers_to_device();
    photon_reduce_work_->set_estimator_progress(render_work.path_trace.start_sample +
                                                    render_work.path_trace.num_samples,
                                                render_work.path_trace.sample_offset);
  }

  if (!is_cancel_requested() && !device_->have_error() && sampling_epoch_complete_cb) {
    sampling_epoch_complete_cb(epoch);
  }

  float occupancy_accum = 0.0f;
  for (const WorkBalanceInfo &balance_info : work_balance_infos_) {
    occupancy_accum += balance_info.occupancy;
  }
  const float occupancy = occupancy_accum / num_works;
  render_scheduler_.report_path_trace_occupancy(render_work, occupancy);

  render_scheduler_.report_path_trace_time(
      render_work, time_dt() - start_time, is_cancel_requested());
}

void PathTrace::adaptive_sample(RenderWork &render_work)
{
  if (!render_work.adaptive_sampling.filter) {
    return;
  }

  bool did_reschedule_on_idle = false;

  while (true) {
    LOG_DEBUG << "Will filter adaptive stopping buffer, threshold "
              << render_work.adaptive_sampling.threshold;
    if (render_work.adaptive_sampling.reset) {
      LOG_DEBUG << "Will re-calculate convergency flag for currently converged pixels.";
    }

    const double start_time = time_dt();

    uint num_active_pixels = 0;
    if (photon_replicas_) {
      num_active_pixels = photon_reduce_work_->adaptive_sampling_converge_filter_count_active(
          render_work.adaptive_sampling.threshold, render_work.adaptive_sampling.reset);
    }
    else {
      parallel_for_each(path_trace_works_, [&](unique_ptr<PathTraceWork> &path_trace_work) {
        const uint num_active_pixels_in_work =
            path_trace_work->adaptive_sampling_converge_filter_count_active(
                render_work.adaptive_sampling.threshold, render_work.adaptive_sampling.reset);
        if (num_active_pixels_in_work) {
          atomic_add_and_fetch_u(&num_active_pixels, num_active_pixels_in_work);
        }
      });
    }

    render_scheduler_.report_adaptive_filter_time(
        render_work, time_dt() - start_time, is_cancel_requested());

    if (num_active_pixels == 0) {
      LOG_DEBUG << "All pixels converged.";
      if (!render_scheduler_.render_work_reschedule_on_converge(render_work)) {
        break;
      }
      LOG_DEBUG << "Continuing with lower threshold.";
    }
    else if (did_reschedule_on_idle) {
      break;
    }
    else if (num_active_pixels < 128 * 128) {
      /* NOTE: The hardcoded value of 128^2 is more of an empirical value to keep GPU busy so that
       * there is no performance loss from the progressive noise floor feature.
       *
       * A better heuristic is possible here: for example, use maximum of 128^2 and percentage of
       * the final resolution. */
      if (!render_scheduler_.render_work_reschedule_on_idle(render_work)) {
        LOG_DEBUG << "Rescheduling is not possible: final threshold is reached.";
        break;
      }
      LOG_DEBUG << "Rescheduling lower threshold.";
      did_reschedule_on_idle = true;
    }
    else {
      break;
    }
  }
  if (photon_replicas_) {
    broadcast_photon_convergence();
  }
}

void PathTrace::set_denoiser_params(const DenoiseParams &params)
{
  if (!lpe_denoise_params_.equals(params) || lpe_denoise_params_.use_gpu != params.use_gpu) {
    if (lpe_denoise_params_.type != params.type || lpe_denoise_params_.use_gpu != params.use_gpu)
      lpe_denoiser_.reset();
    lpe_denoised_plan_.reset();
    lpe_denoise_params_ = params;
  }
  if (!params.use) {
    denoiser_.reset();
    render_scheduler_.set_denoiser_params(params);
    return;
  }

  GraphicsInteropDevice interop_device;
  if (display_) {
    interop_device = display_->graphics_interop_get_device();
  }

  Device *effective_denoise_device;
  Device *cpu_fallback_device = cpu_device_.get();
  const DenoiseParams effective_denoise_params = get_effective_denoise_params(
      denoise_device_, cpu_fallback_device, params, interop_device, effective_denoise_device);

  bool need_to_recreate_denoiser = false;
  if (denoiser_) {
    const DenoiseParams old_denoiser_params = denoiser_->get_params();

    const bool is_cpu_denoising = old_denoiser_params.type == DENOISER_OPENIMAGEDENOISE &&
                                  old_denoiser_params.use_gpu == false;
    const bool always_gpu_denoising = effective_denoise_params.type == DENOISER_DLSS ||
                                      effective_denoise_params.type == DENOISER_OPTIX;
    const bool requested_gpu_denoising = always_gpu_denoising ||
                                         (effective_denoise_params.type ==
                                              DENOISER_OPENIMAGEDENOISE &&
                                          effective_denoise_params.use_gpu == true);
    if (requested_gpu_denoising && is_cpu_denoising &&
        effective_denoise_device->info.type == DEVICE_CPU)
    {
      /* It won't be possible to use GPU denoising when according to user settings we have
       * only CPU as available denoising device. So we just exiting early to avoid
       * unnecessary denoiser recreation or parameters update. */
      return;
    }

    const bool is_same_denoising_device_type = old_denoiser_params.use_gpu ==
                                               effective_denoise_params.use_gpu;
    /* Optix Denoiser is not supporting CPU devices, so use_gpu option is not
     * shown in the UI and changes in the option value should not be checked. */
    if (old_denoiser_params.type == effective_denoise_params.type &&
        (is_same_denoising_device_type || always_gpu_denoising))
    {
      denoiser_->set_params(effective_denoise_params);
    }
    else {
      need_to_recreate_denoiser = true;
    }
  }
  else {
    /* if there is no denoiser and param.use is true, then we need to create it. */
    need_to_recreate_denoiser = true;
  }

  if (need_to_recreate_denoiser) {
    denoiser_ = Denoiser::create(
        effective_denoise_device, cpu_fallback_device, effective_denoise_params, interop_device);

    if (denoiser_) {
      /* Only take into account the "immediate" cancel to have interactive rendering responding to
       * navigation as quickly as possible, but allow to run denoiser after user hit Escape key
       * while doing offline rendering. */
      denoiser_->is_cancelled_cb = [this]() { return render_cancel_.is_requested; };
    }
  }

  /* Use actual parameters, if available */
  if (denoise_device_ && denoiser_) {
    render_scheduler_.set_denoiser_params(denoiser_->get_params());
  }
  else {
    render_scheduler_.set_denoiser_params(effective_denoise_params);
  }
}

void PathTrace::set_adaptive_sampling(const AdaptiveSampling &adaptive_sampling)
{
  render_scheduler_.set_adaptive_sampling(adaptive_sampling);
}

void PathTrace::cryptomatte_postprocess(const RenderWork &render_work)
{
  if (!render_work.cryptomatte.postprocess) {
    return;
  }
  LOG_DEBUG << "Perform cryptomatte work.";

  if (photon_replicas_) {
    photon_reduce_work_->cryptomatte_postproces();
    return;
  }
  parallel_for_each(path_trace_works_, [&](unique_ptr<PathTraceWork> &path_trace_work) {
    path_trace_work->cryptomatte_postproces();
  });
}

void PathTrace::denoise(const RenderWork &render_work)
{
  const auto preview = film_->get_lpe_preview();
  if (display_ && (preview->combined || preview->evaluate || !preview->pass.empty() || !preview->terms.empty())) {
    /* This display needs its edited image denoised, not the unedited Combined. */
    return;
  }
  if (!render_work.tile.denoise) {
    return;
  }

  if (!denoiser_) {
    /* Denoiser was not configured, so nothing to do here. */
    return;
  }

  LOG_DEBUG << "Perform denoising work.";

  const double start_time = time_dt();

  RenderBuffers *buffer_to_denoise = nullptr;
  bool allow_inplace_modification = false;

  Device *denoiser_device = denoiser_->get_denoiser_device();
  if (path_trace_works_.size() > 1 && denoiser_device && !big_tile_denoise_work_) {
    big_tile_denoise_work_ = PathTraceWork::create(denoiser_device, film_, device_scene_, nullptr);
  }

  if (big_tile_denoise_work_) {
    big_tile_denoise_work_->set_effective_buffer_params(
        render_state_.effective_big_tile_params,
        render_state_.effective_big_tile_params,
        render_state_.effective_denoised_big_tile_params,
        render_state_.effective_denoised_big_tile_params);

    buffer_to_denoise = big_tile_denoise_work_->get_render_buffers();
    buffer_to_denoise->reset(render_state_.effective_denoised_big_tile_params);

    copy_to_render_buffers(buffer_to_denoise);

    allow_inplace_modification = true;
  }
  else {
    DCHECK_EQ(path_trace_works_.size(), 1);

    buffer_to_denoise = path_trace_works_.front()->get_render_buffers();
  }

  if (denoiser_->denoise_buffer(render_state_.effective_big_tile_params,
                                render_state_.effective_denoised_big_tile_params,
                                buffer_to_denoise,
                                get_num_samples_in_buffer(),
                                allow_inplace_modification,
                                device_scene_->data.integrator.pixel_jitter))
  {
    render_state_.has_denoised_result = true;
  }

  render_scheduler_.report_denoise_time(render_work, time_dt() - start_time);
}

void PathTrace::denoise_volume_guiding_buffers(const RenderWork &render_work,
                                               const bool has_volume)
{
  if (!has_volume || !render_work.volume_guiding_denoise) {
    return;
  }

  LOG_DEBUG << "Denoise volume guiding buffers.";

  const double start_time = time_dt();

  /* TODO: in the multi-GPU case, we can denoise on one device and copy to the rest, instead of
   * denoising on each device separately. */
  parallel_for_each(path_trace_works_, [&](unique_ptr<PathTraceWork> &path_trace_work) {
    path_trace_work->denoise_volume_guiding_buffers();
  });

  render_scheduler_.report_volume_guiding_denoise_time(render_work, time_dt() - start_time);
}

void PathTrace::set_output_driver(unique_ptr<OutputDriver> driver)
{
  output_driver_ = std::move(driver);
}

void PathTrace::set_display_driver(unique_ptr<DisplayDriver> driver)
{
  /* The display driver is the source of the drawing context which might be used by
   * path trace works. Make sure there is no graphics interop using resources from
   * the old display, as it might no longer be available after this call. */
  destroy_gpu_resources();

  if (driver) {
    display_ = make_unique<PathTraceDisplay>(std::move(driver));
  }
  else {
    display_ = nullptr;
  }
}

void PathTrace::set_deep_output_driver(unique_ptr<DeepOutputDriver> driver)
{
  deep_output_driver_ = std::move(driver);
}

void PathTrace::zero_display()
{
  if (display_) {
    display_->zero();
  }
}

void PathTrace::draw()
{
  if (!display_) {
    return;
  }

  did_draw_after_reset_ |= display_->draw();
}

void PathTrace::flush_display()
{
  if (!display_) {
    return;
  }

  display_->flush();
}

void PathTrace::update_display(const RenderWork &render_work)
{
  if (photon_cache_native_) {
    update_photon_cache_display(render_work, render_work.tile.write);
    return;
  }

  /* A user pass change should be shown after this completed batch, even when
   * the normal progressive display cadence would skip the update. */
  const uint64_t lpe_display_revision = lpe_display_revision_.load(std::memory_order_acquire);
  if (!render_work.display.update && !film_->display_pass_name_changed() &&
      lpe_display_revision == lpe_displayed_revision_)
  {
    return;
  }

  /* A failed GPU batch has no pixels to upload. Vulkan pixel buffers are
   * allocated by the writer; an empty upload can hide the original error. */
  if (device_->have_error()) {
    return;
  }

  if (!display_ && !output_driver_) {
    LOG_DEBUG << "Ignore display update.";
    return;
  }

  if (full_params_.width == 0 || full_params_.height == 0) {
    LOG_DEBUG << "Skipping PathTraceDisplay update due to 0 size of the render buffer.";
    return;
  }

  const double start_time = time_dt();

  if (output_driver_) {
    LOG_DEBUG << "Invoke buffer update callback.";

    const PathTraceTile tile(*this);
    output_driver_->update_render_tile(tile);
  }

  if (display_) {
    LOG_DEBUG << "Perform copy to GPUDisplay work.";

    const ustring previous_display_name = film_->get_display_pass_name();
    film_->update_display_pass_name();
    const bool lpe_denoised = update_lpe_denoised_preview(render_work);
    const ustring display_name = film_->get_display_pass_name();
    const auto program = film_->get_lpe_program();
    const LPEOutput *lpe_output = program ? program->find_display_output(display_name.string()) : nullptr;
    const string buffer_name = program ? program->display_buffer_name(display_name.string()) :
                                         display_name.string();
    const bool output_expression = lpe_output && buffer_name == display_name.string();
    const auto display_pass = [&](const PassMode mode) {
      return display_name.empty() ?
                 big_tile_params_.get_actual_display_pass(film_->get_display_pass(), mode) :
                 big_tile_params_.get_actual_display_pass(buffer_name, mode);
    };
    const BufferPass *noisy_pass = display_pass(PassMode::NOISY);
    const BufferPass *denoised_pass = display_pass(PassMode::DENOISED);
    if (display_name != previous_display_name && getenv("CYCLES_DISPLAY_PASS_TRACE")) {
      fprintf(stderr, "[Cycles display] name=%s group=%s offset=%d samples=%d\n",
              display_name.c_str(), noisy_pass ? noisy_pass->lightgroup.c_str() : "",
              noisy_pass ? noisy_pass->offset : PASS_UNUSED, get_num_samples_in_buffer());
    }
    const PassType pass_type = noisy_pass ? noisy_pass->type : film_->get_display_pass();
    const auto lpe_preview = film_->get_lpe_preview();
    const bool expression_preview = bool(lpe_preview->evaluate);
    const bool show_denoised =
        !lpe_preview->combined && !expression_preview && lpe_preview->terms.empty() &&
        lpe_preview->pass.empty() &&
        ((render_work.display.use_denoised_result && has_denoised_result() && denoised_pass &&
          denoised_pass->offset != PASS_UNUSED) || is_volume_guiding_pass(pass_type));

    const int texture_width = show_denoised ?
                                  render_state_.effective_denoised_big_tile_params.window_width :
                                  render_state_.effective_big_tile_params.window_width;
    const int texture_height = show_denoised ?
                                   render_state_.effective_denoised_big_tile_params.window_height :
                                   render_state_.effective_big_tile_params.window_height;
    std::vector<float> output_pixels;
    if (output_expression) {
      const PathTraceTile tile(*this, true);
      if (!program->display_evaluate ||
          !program->display_evaluate(*program, lpe_output->expression, tile, output_pixels) ||
          output_pixels.size() != size_t(texture_width) * texture_height * 3) {
        LOG_WARNING << "Unable to evaluate LPE display output " << display_name;
        return;
      }
    }
    /* Resolve CPU expressions/readbacks before update_begin takes Blender's graphics-context
     * lock. An expensive Light Edit/Viewer must not hold up the main thread's drawing while
     * waiting for Cycles device buffers or evaluating the image graph. */
    std::vector<half4> prepared_rgba;
    if (output_expression) {
      prepared_rgba.resize(size_t(texture_width) * texture_height);
      parallel_for(int64_t(0), int64_t(prepared_rgba.size()), [&](int64_t i) {
        const float *p = output_pixels.data() + i * 3;
        prepared_rgba[i] = {float_to_half_display(p[0]),
                            float_to_half_display(p[1]),
                            float_to_half_display(p[2]),
                            float_to_half(1.0f)};
      });
    }
    else if (lpe_denoised && lpe_denoised_plan_ == lpe_preview &&
             lpe_denoised_rgba_.size() == size_t(texture_width) * texture_height * 4)
    {
      prepared_rgba.resize(size_t(texture_width) * texture_height);
      parallel_for(int64_t(0), int64_t(prepared_rgba.size()), [&](int64_t i) {
        const float *p = lpe_denoised_rgba_.data() + i * 4;
        prepared_rgba[i] = {float_to_half_display(p[0]),
                            float_to_half_display(p[1]),
                            float_to_half_display(p[2]),
                            float_to_half(p[3])};
      });
    }
    else if (expression_preview) {
      const PathTraceTile tile(*this, true);
      std::vector<float> rgb;
      /* Use the native effective viewport dimensions, including HiDPI pixel
       * sizes and initial coarse previews. Never substitute Combined on failure. */
      if (tile.size != make_int2(texture_width, texture_height) ||
          !lpe_preview->evaluate(tile, rgb) ||
          rgb.size() != size_t(texture_width) * texture_height * 3)
      {
        return;
      }
      prepared_rgba.resize(size_t(texture_width) * texture_height);
      parallel_for(int64_t(0), int64_t(prepared_rgba.size()), [&](int64_t i) {
        prepared_rgba[i] = {float_to_half_display(rgb[3 * i]),
                            float_to_half_display(rgb[3 * i + 1]),
                            float_to_half_display(rgb[3 * i + 2]),
                            float_to_half(1.0f)};
      });
    }
    if (!display_->update_begin(texture_width, texture_height)) {
      LOG_ERROR << "Error beginning GPUDisplay update.";
      return;
    }

    const PassMode pass_mode = show_denoised ? PassMode::DENOISED : PassMode::NOISY;
    const int num_samples = get_num_samples_in_buffer();
    if (!prepared_rgba.empty()) {
      display_->copy_pixels_to_texture(prepared_rgba.data(), 0, 0, texture_width, texture_height);
    }
    else if (big_tile_denoise_work_ && show_denoised) {
      big_tile_denoise_work_->copy_to_display(display_.get(), pass_mode, num_samples);
    }
    else if (photon_replicas_) {
      photon_reduce_work_->copy_to_display(display_.get(), pass_mode, num_samples);
    }
    else {
      for (auto &&path_trace_work : path_trace_works_) {
        path_trace_work->copy_to_display(display_.get(), pass_mode, num_samples);
      }
    }

    display_->update_end();
  }

  lpe_displayed_revision_ = lpe_display_revision;
  render_scheduler_.report_display_update_time(render_work, time_dt() - start_time);
}

void PathTrace::rebalance(const RenderWork &render_work)
{
  if (photon_replicas_ || !render_work.rebalance) {
    return;
  }

  const int num_works = path_trace_works_.size();

  if (num_works == 1) {
    LOG_DEBUG << "Ignoring rebalance work due to single device render.";
    return;
  }

  const double start_time = time_dt();

  if (LOG_IS_ON(LOG_LEVEL_DEBUG)) {
    LOG_DEBUG << "Perform rebalance work.";
    LOG_DEBUG << "Per-device path tracing time (seconds):";
    for (int i = 0; i < num_works; ++i) {
      LOG_DEBUG << path_trace_works_[i]->get_device()->info.description << ": "
                << work_balance_infos_[i].time_spent;
    }
  }

  const bool did_rebalance = work_balance_do_rebalance(work_balance_infos_);

  if (LOG_IS_ON(LOG_LEVEL_DEBUG)) {
    LOG_DEBUG << "Calculated per-device weights for works:";
    for (int i = 0; i < num_works; ++i) {
      LOG_DEBUG << path_trace_works_[i]->get_device()->info.description << ": "
                << work_balance_infos_[i].weight;
    }
  }

  if (!did_rebalance) {
    LOG_DEBUG << "Balance in path trace works did not change.";
    render_scheduler_.report_rebalance_time(render_work, time_dt() - start_time, false);
    return;
  }

  RenderBuffers big_tile_cpu_buffers(cpu_device_.get());
  big_tile_cpu_buffers.reset(render_state_.effective_big_tile_params);

  copy_to_render_buffers(&big_tile_cpu_buffers);

  render_state_.need_reset_params = true;
  update_work_buffer_params_if_needed(render_work);

  copy_from_render_buffers(&big_tile_cpu_buffers);

  render_scheduler_.report_rebalance_time(render_work, time_dt() - start_time, true);
}

void PathTrace::sync_deep_output_buffers()
{
  if (!deep_output_driver_ || !deep_output_driver_->is_enabled()) {
    return;
  }

  vector<DeepOutputDriver::SliceParams> slices;
  slices.reserve(path_trace_works_.size());

  for (auto &&path_trace_work : path_trace_works_) {
    const BufferParams &params = path_trace_work->get_effective_buffer_params();
    DeepOutputDriver::SliceParams slice;
    slice.device = path_trace_work->get_device();
    slice.full_x = params.full_x;
    slice.full_y = params.full_y;
    slice.width = params.width;
    slice.height = params.height;
    slice.window_x = params.window_x;
    slice.window_y = params.window_y;
    slice.window_width = params.window_width;
    slice.window_height = params.window_height;
    slices.push_back(slice);
  }

  deep_output_driver_->sync_device_buffers(slices, device_scene_->data);
  deep_output_driver_->update_device_kernel_data(device_scene_->data, photon_estimator_ == 1);
}

void PathTrace::write_tile_buffer(const RenderWork &render_work)
{
  if (!render_work.tile.write) {
    return;
  }

  LOG_DEBUG << "Write tile result.";

  render_state_.tile_written = true;

  const bool has_multiple_tiles = tile_manager_.has_multiple_tiles();

  if (deep_output_driver_ && deep_output_driver_->is_enabled()) {
    const int2 tile_size = get_render_tile_size();
    const int2 tile_offset = get_render_tile_offset();
    vector<float> beauty_pixels(static_cast<size_t>(tile_size.x) * tile_size.y * 4);
    vector<float> sample_count_pixels(static_cast<size_t>(tile_size.x) * tile_size.y);
    bool has_beauty = false;
    bool has_sample_count = false;

    get_deep_tile_pixels(beauty_pixels, has_beauty, sample_count_pixels, has_sample_count);

    if (has_multiple_tiles) {
      deep_output_driver_->accumulate_tile(has_beauty ? beauty_pixels.data() : nullptr,
                                           has_sample_count ? sample_count_pixels.data() : nullptr,
                                           float(get_num_render_tile_samples()),
                                           tile_size.x,
                                           tile_size.y,
                                           tile_offset.x,
                                           tile_offset.y);
    }
    else if (has_sample_count) {
      /* Deep photon bins accumulate every camera sample, even when the user did
       * not request the public Debug Sample Count pass. Keep the actual count. */
      deep_output_driver_->set_sample_count_buffer(sample_count_pixels.data(),
                                                   tile_size.x,
                                                   tile_size.y,
                                                   float(get_num_render_tile_samples()));
    }
  }

  /* Write render tile result, but only if not using tiled rendering.
   *
   * Tiles are written to a file during rendering, and written to the software at the end
   * of rendering (wither when all tiles are finished, or when rendering was requested to be
   * canceled).
   *
   * Important thing is: tile should be written to the software via callback only once. */
  if (!has_multiple_tiles) {
    LOG_DEBUG << "Write tile result via buffer write callback.";
    tile_buffer_write();
  }
  /* Write tile to disk, so that the render work's render buffer can be re-used for the next tile.
   */
  else {
    LOG_DEBUG << "Write tile result to disk.";
    tile_buffer_write_to_disk();
  }
}

void PathTrace::finalize_full_buffer_on_disk(const RenderWork &render_work)
{
  if (!render_work.full.write) {
    return;
  }

  LOG_DEBUG << "Handle full-frame render buffer work.";

  if (!tile_manager_.has_written_tiles()) {
    LOG_DEBUG << "No tiles on disk.";
    return;
  }

  /* Make sure writing to the file is fully finished.
   * This will include writing all possible missing tiles, ensuring validness of the file. */
  tile_manager_.finish_write_tiles();

  /* NOTE: The rest of full-frame post-processing (such as full-frame denoising) will be done after
   * all scenes and layers are rendered by the Session (which happens after freeing Session memory,
   * so that we never hold scene and full-frame buffer in memory at the same time). */
}

void PathTrace::cancel()
{
  thread_scoped_lock lock(render_cancel_.mutex);

  /* With interactive denoisers like DLSS always render 1 complete sample,
   * otherwise interactivity becomes bad. Otherwise cancel immediately so
   * the main thread isn't blocked. */
  if (!render_scheduler_.is_denoiser_interactive() || get_num_samples_in_buffer() > 1) {
    render_cancel_.is_requested = true;
  }

  while (render_cancel_.is_rendering) {
    render_cancel_.condition.wait(lock);
  }

  render_cancel_.is_requested = false;

  if (display_) {
    display_->reset();
  }
}

int PathTrace::get_num_samples_in_buffer()
{
  return completed_buffer_samples_ * (photon_replicas_ ? int(path_trace_works_.size()) : 1);
}

bool PathTrace::is_cancel_requested()
{
  if (render_cancel_.is_requested) {
    return true;
  }

  if (progress_ != nullptr) {
    if (progress_->get_cancel()) {
      return true;
    }
  }

  return false;
}

void PathTrace::tile_buffer_write()
{
  if (!output_driver_) {
    return;
  }

  const PathTraceTile tile(*this);
  output_driver_->write_render_tile(tile);
}

void PathTrace::tile_buffer_read()
{
  if (!device_scene_->data.bake.use) {
    return;
  }

  if (!output_driver_) {
    return;
  }

  /* Read buffers back from device. */
  parallel_for_each(path_trace_works_, [&](unique_ptr<PathTraceWork> &path_trace_work) {
    path_trace_work->copy_render_buffers_from_device();
  });

  /* Read (subset of) passes from output driver. */
  const PathTraceTile tile(*this);
  if (output_driver_->read_render_tile(tile)) {
    /* Copy buffers to device again. */
    parallel_for_each(path_trace_works_, [](unique_ptr<PathTraceWork> &path_trace_work) {
      path_trace_work->copy_render_buffers_to_device();
    });
  }
}

void PathTrace::get_deep_tile_pixels(vector<float> &beauty_pixels,
                                     bool &has_beauty_pixels,
                                     vector<float> &sample_count_pixels,
                                     bool &has_sample_count_pixels)
{
  has_beauty_pixels = false;
  has_sample_count_pixels = false;

  if (!copy_render_tile_from_device()) {
    return;
  }

  const int2 tile_size = get_render_tile_size();
  const int2 tile_offset = get_render_tile_offset();
  const int fallback_sample_count = get_num_render_tile_samples();

  for (const unique_ptr<PathTraceWork> &path_trace_work : path_trace_works_) {
    RenderBuffers *buffers = path_trace_work->get_render_buffers();
    const BufferParams &params = buffers->params;

    const int combined_offset = params.get_pass_offset(PASS_COMBINED);
    const int sample_count_offset = params.get_pass_offset(PASS_SAMPLE_COUNT);

    if (params.width <= 0 || params.height <= 0 || params.stride <= 0 || params.pass_stride <= 0 ||
        buffers->buffer.size() < static_cast<size_t>(params.width) * params.height *
                                     params.pass_stride)
    {
      continue;
    }

    if (combined_offset == PASS_UNUSED && sample_count_offset == PASS_UNUSED) {
      continue;
    }

    const int64_t pass_stride = params.pass_stride;
    const int64_t row_stride = params.stride * pass_stride;
    const float *buffer = buffers->buffer.data();

    for (int y = 0; y < params.window_height; y++) {
      const int local_y = params.window_y + y;
      const int global_y = params.full_y + local_y;
      const int tile_y = global_y - tile_offset.y;
      if (local_y < 0 || local_y >= params.height || tile_y < 0 || tile_y >= tile_size.y) {
        continue;
      }

      const float *row = buffer + static_cast<int64_t>(local_y) * row_stride;
      for (int x = 0; x < params.window_width; x++) {
        const int local_x = params.window_x + x;
        const int global_x = params.full_x + local_x;
        const int tile_x = global_x - tile_offset.x;
        if (local_x < 0 || local_x >= params.width || tile_x < 0 || tile_x >= tile_size.x) {
          continue;
        }

        const float *pixel = row + static_cast<int64_t>(local_x) * pass_stride;
        const size_t dst_index = static_cast<size_t>(tile_y) * tile_size.x + tile_x;

        uint32_t sample_count = fallback_sample_count;
        if (sample_count_offset != PASS_UNUSED) {
          sample_count = pass_float_as_uint(pixel[sample_count_offset]);
          sample_count_pixels[dst_index] = fallback_sample_count > 0 ?
                                               float(sample_count) / float(fallback_sample_count) :
                                               0.0f;
          has_sample_count_pixels = true;
        }

        if (combined_offset != PASS_UNUSED) {
          const float scale = sample_count > 0 ? 1.0f / float(sample_count) : 0.0f;
          const size_t beauty_index = dst_index * 4;
          beauty_pixels[beauty_index + 0] = pixel[combined_offset + 0] * scale * params.exposure;
          beauty_pixels[beauty_index + 1] = pixel[combined_offset + 1] * scale * params.exposure;
          beauty_pixels[beauty_index + 2] = pixel[combined_offset + 2] * scale * params.exposure;
          beauty_pixels[beauty_index + 3] = combined_transparency_to_alpha(
              pixel[combined_offset + 3] * scale);
          has_beauty_pixels = true;
        }
      }
    }
  }
}

void PathTrace::tile_buffer_write_to_disk()
{
  /* Sample count pass is required to support per-tile partial results stored in the file. */
  DCHECK_NE(big_tile_params_.get_pass_offset(PASS_SAMPLE_COUNT), PASS_UNUSED);

  const int num_rendered_samples = render_scheduler_.get_num_rendered_samples();

  if (num_rendered_samples == 0) {
    /* The tile has zero samples, no need to write it. */
    return;
  }

  /* Get access to the CPU-side render buffers of the current big tile. */
  RenderBuffers *buffers;
  RenderBuffers big_tile_cpu_buffers(cpu_device_.get());

  if (path_trace_works_.size() == 1) {
    path_trace_works_[0]->copy_render_buffers_from_device();
    buffers = path_trace_works_[0]->get_render_buffers();
  }
  else {
    big_tile_cpu_buffers.reset(render_state_.effective_big_tile_params);
    copy_to_render_buffers(&big_tile_cpu_buffers);

    buffers = &big_tile_cpu_buffers;
  }

  if (!tile_manager_.write_tile(*buffers, photon_replicas_ ? int(path_trace_works_.size()) : 1)) {
    device_->set_error("Error writing tile to file");
  }
  else {
    store_photon_preview_tile(*buffers);
  }
}

void PathTrace::progress_update_if_needed(const RenderWork &render_work)
{
  if (progress_ != nullptr) {
    const int2 tile_size = get_render_tile_size();
    const uint64_t num_samples_added = uint64_t(tile_size.x) * tile_size.y *
                                       render_work.path_trace.num_samples;
    const int current_sample = render_scheduler_.get_total_rendered_samples();
    progress_->add_samples(num_samples_added, current_sample);
  }

  if (progress_update_cb) {
    progress_update_cb();
  }
}

void PathTrace::progress_set_status(const string &status, const string &substatus)
{
  if (progress_ != nullptr) {
    progress_->set_status(status, substatus);
  }
}

void PathTrace::copy_to_render_buffers(RenderBuffers *render_buffers)
{
  if (photon_replicas_) {
    photon_reduce_work_->copy_to_render_buffers(render_buffers);
    render_buffers->copy_to_device();
    return;
  }
  parallel_for_each(path_trace_works_,
                    [&render_buffers](unique_ptr<PathTraceWork> &path_trace_work) {
                      path_trace_work->copy_to_render_buffers(render_buffers);
                    });
  render_buffers->copy_to_device();
}

void PathTrace::copy_from_render_buffers(RenderBuffers *render_buffers)
{
  render_buffers->copy_from_device();
  parallel_for_each(path_trace_works_,
                    [&render_buffers](unique_ptr<PathTraceWork> &path_trace_work) {
                      path_trace_work->copy_from_render_buffers(render_buffers);
                    });
}

bool PathTrace::copy_render_tile_from_device()
{
  if (full_frame_state_.render_buffers) {
    /* Full-frame buffer is always allocated on CPU. */
    return true;
  }

  if (big_tile_denoise_work_ && render_state_.has_denoised_result) {
    if (!big_tile_denoise_work_->copy_render_buffers_from_device()) {
      return false;
    }
    if (!photon_replicas_) {
      return true;
    }
  }
  if (photon_replicas_) {
    return photon_reduce_work_->copy_render_buffers_from_device();
  }

  bool success = true;

  parallel_for_each(path_trace_works_, [&](unique_ptr<PathTraceWork> &path_trace_work) {
    if (!success) {
      return;
    }
    if (!path_trace_work->copy_render_buffers_from_device()) {
      success = false;
    }
  });

  return success;
}

static string get_layer_view_name(const RenderBuffers &buffers)
{
  string result;

  if (!buffers.params.layer.empty()) {
    result += string(buffers.params.layer);
  }

  if (!buffers.params.view.empty()) {
    if (!result.empty()) {
      result += ", ";
    }
    result += string(buffers.params.view);
  }

  return result;
}

void PathTrace::set_full_buffer_state(std::shared_ptr<FullBufferState> state)
{
  full_buffer_state_ = std::move(state);
}

void PathTrace::capture_full_buffer_context(const string_view filename)
{
  /* Scene film state is replaced by the next layer before deferred file replay. */
  full_buffer_state_->lpe_programs[string(filename)] = film_->get_lpe_program();
}

void PathTrace::process_full_buffer_from_disk(string_view filename)
{
  LOG_DEBUG << "Processing full frame buffer file " << filename;

  progress_set_status("Reading full buffer from disk");

  RenderBuffers full_frame_buffers(cpu_device_.get());

  DenoiseParams denoise_params;
  if (!tile_manager_.read_full_buffer_from_disk(filename, &full_frame_buffers, &denoise_params)) {
    const string error_message = "Error reading tiles from file";
    if (progress_) {
      progress_->set_error(error_message);
      progress_->set_cancel(error_message);
    }
    else {
      LOG_ERROR << error_message;
    }
    return;
  }

  const auto current_lpe_program = film_->get_lpe_program();
  const auto saved_lpe_program = full_buffer_state_->lpe_programs.find(string(filename));
  if (saved_lpe_program != full_buffer_state_->lpe_programs.end()) {
    film_->set_lpe_program(saved_lpe_program->second);
  }
  SCOPED_DEFER([&]() {
    film_->set_lpe_program(current_lpe_program);
    full_buffer_state_->lpe_programs.erase(string(filename));
  }());

  apply_photon_cache(filename, full_frame_buffers);

  const string layer_view_name = get_layer_view_name(full_frame_buffers);

  render_state_.has_denoised_result = false;

  if (denoise_params.use && denoiser_ && !progress_->get_cancel()) {
    progress_set_status(layer_view_name, "Denoising");

    /* If GPU should be used is not based on file metadata. */
    denoise_params.use_gpu = render_scheduler_.is_denoiser_gpu_used();

    /* Re-use the denoiser as much as possible, avoiding possible device re-initialization.
     *
     * It will not conflict with the regular rendering as:
     *  - Rendering is supposed to be finished here.
     *  - The next rendering will go via Session's `run_update_for_next_iteration` which will
     *    ensure proper denoiser is used. */
    set_denoiser_params(denoise_params);

    /* Number of samples doesn't matter too much, since the samples count pass will be used. */
    denoiser_->denoise_buffer(
        full_frame_buffers.params, full_frame_buffers.params, &full_frame_buffers, 0, false);

    render_state_.has_denoised_result = true;
  }

  /* Full-buffer replay writes the whole loaded physical buffer back to Blender as one tile.
   * Normalize the replay window to the loaded buffer extent so pass accessors do not apply the
   * display-window offset a second time. */
  full_frame_buffers.params.window_x = 0;
  full_frame_buffers.params.window_y = 0;
  full_frame_buffers.params.window_width = full_frame_buffers.params.width;
  full_frame_buffers.params.window_height = full_frame_buffers.params.height;

  full_frame_state_.render_buffers = &full_frame_buffers;

  progress_set_status(layer_view_name, "Finishing");

  /* Write the full result pretending that there is a single tile.
   * Requires some state change, but allows to use same communication API with the software. */
  tile_buffer_write();

  full_frame_state_.render_buffers = nullptr;
}

int PathTrace::get_num_render_tile_samples() const
{
  if (full_frame_state_.render_buffers) {
    return full_frame_state_.render_buffers->params.samples;
  }

  return completed_buffer_samples_ * (photon_replicas_ ? int(path_trace_works_.size()) : 1);
}

bool PathTrace::get_render_tile_pixels(const PassAccessor &pass_accessor,
                                       const PassAccessor::Destination &destination,
                                       const bool effective)
{
  if (full_frame_state_.render_buffers) {
    return pass_accessor.get_render_tile_pixels(full_frame_state_.render_buffers, destination);
  }

  if (big_tile_denoise_work_ && render_state_.has_denoised_result) {
    /* Only use the big tile denoised buffer to access the denoised passes.
     * The guiding passes are allowed to be modified in-place for the needs of the denoiser,
     * so copy those from the original devices buffers. */
    if (pass_accessor.get_pass_access_info().mode == PassMode::DENOISED) {
      return big_tile_denoise_work_->get_render_tile_pixels(pass_accessor, destination);
    }
  }

  if (photon_replicas_) {
    return photon_reduce_work_->get_render_tile_pixels(pass_accessor, destination, effective);
  }
  bool success = true;

  parallel_for_each(path_trace_works_, [&](unique_ptr<PathTraceWork> &path_trace_work) {
    if (!success) {
      return;
    }
    if (!path_trace_work->get_render_tile_pixels(pass_accessor, destination, effective)) {
      success = false;
    }
  });

  return success;
}

bool PathTrace::set_render_tile_pixels(PassAccessor &pass_accessor,
                                       const PassAccessor::Source &source)
{
  bool success = true;

  parallel_for_each(path_trace_works_, [&](unique_ptr<PathTraceWork> &path_trace_work) {
    if (!success) {
      return;
    }
    if (!path_trace_work->set_render_tile_pixels(pass_accessor, source)) {
      success = false;
    }
  });

  return success;
}

int2 PathTrace::get_render_tile_size() const
{
  if (full_frame_state_.render_buffers) {
    return make_int2(full_frame_state_.render_buffers->params.window_width,
                     full_frame_state_.render_buffers->params.window_height);
  }

  if (device_scene_->data.film.photon_compact) {
    return make_int2(big_tile_params_.window_width, big_tile_params_.window_height);
  }
  const Tile &tile = tile_manager_.get_current_tile();
  return make_int2(tile.window_width, tile.window_height);
}

int2 PathTrace::get_display_tile_size() const
{
  return make_int2(render_state_.effective_big_tile_params.window_width,
                   render_state_.effective_big_tile_params.window_height);
}

bool PathTrace::is_full_resolution() const
{
  return render_state_.resolution_divider == 1;
}

void PathTrace::refresh_display()
{
  if (get_num_render_tile_samples() <= 0)
    return;
  RenderWork work;
  work.display.update = true;
  work.tile.lpe_denoise = render_scheduler_.lpe_denoise_ready_for_edit();
  update_display(work);
}

int2 PathTrace::get_render_tile_offset() const
{
  if (full_frame_state_.render_buffers) {
    return make_int2(0, 0);
  }

  const Tile &tile = tile_manager_.get_current_tile();
  return make_int2(tile.x + tile.window_x, tile.y + tile.window_y);
}

int2 PathTrace::get_render_size() const
{
  return tile_manager_.get_size();
}

const BufferParams &PathTrace::get_render_tile_params() const
{
  if (full_frame_state_.render_buffers) {
    return full_frame_state_.render_buffers->params;
  }

  return big_tile_params_;
}

bool PathTrace::has_denoised_result() const
{
  return render_state_.has_denoised_result;
}

void PathTrace::destroy_gpu_resources()
{
  /* Destroy any GPU resource which was used for graphics interop.
   * Need to have access to the PathTraceDisplay as it is the only source of drawing context which
   * is used for interop. */
  if (display_) {
    for (auto &&path_trace_work : path_trace_works_) {
      path_trace_work->destroy_gpu_resources(display_.get());
    }

    if (big_tile_denoise_work_) {
      big_tile_denoise_work_->destroy_gpu_resources(display_.get());
    }
    if (photon_reduce_work_) {
      photon_reduce_work_->destroy_gpu_resources(display_.get());
    }
  }
}

/* --------------------------------------------------------------------
 * Report generation.
 */

static const char *device_type_for_description(const DeviceType type)
{
  switch (type) {
    case DEVICE_NONE:
      return "None";

    case DEVICE_CPU:
      return "CPU";
    case DEVICE_CUDA:
      return "CUDA";
    case DEVICE_OPTIX:
      return "OptiX";
    case DEVICE_HIP:
      return "HIP";
    case DEVICE_HIPRT:
      return "HIPRT";
    case DEVICE_ONEAPI:
      return "oneAPI";
    case DEVICE_DUMMY:
      return "Dummy";
    case DEVICE_MULTI:
      return "Multi";
    case DEVICE_METAL:
      return "Metal";
  }

  return "UNKNOWN";
}

/* Construct description of the device which will appear in the full report. */
/* TODO(sergey): Consider making it more reusable utility. */
static string full_device_info_description(const DeviceInfo &device_info)
{
  string full_description = device_info.description;

  full_description += " (" + string(device_type_for_description(device_info.type)) + ")";

  if (device_info.display_device) {
    full_description += " (display)";
  }

  if (device_info.type == DEVICE_CPU) {
    full_description += " (" + to_string(device_info.cpu_threads) + " threads)";
  }

  full_description += " [" + device_info.id + "]";

  return full_description;
}

/* Construct string which will contain information about devices, possibly multiple of the devices.
 *
 * In the simple case the result looks like:
 *
 *   Message: Full Device Description
 *
 * If there are multiple devices then the result looks like:
 *
 *   Message: Full First Device Description
 *            Full Second Device Description
 *
 * Note that the newlines are placed in a way so that the result can be easily concatenated to the
 * full report. */
static string device_info_list_report(const string &message,
                                      const DeviceInfo &device_info,
                                      const bool show_hardware_raytracing)
{
  string result = "\n" + message + ": ";
  const string pad(message.length() + 2, ' ');

  if (device_info.multi_devices.empty()) {
    result += full_device_info_description(device_info) + "\n";
    if (show_hardware_raytracing) {
      result += pad +
                "  Hardware Ray-Tracing: " + (device_info.use_hardware_raytracing ? "On" : "Off") +
                "\n";
    }
    return result;
  }

  bool is_first = true;
  for (const DeviceInfo &sub_device_info : device_info.multi_devices) {
    if (!is_first) {
      result += pad;
    }

    result += full_device_info_description(sub_device_info) + "\n";
    if (show_hardware_raytracing) {
      result += pad + "  Hardware Ray-Tracing: " +
                (sub_device_info.use_hardware_raytracing ? "On" : "Off") + "\n";
    }

    is_first = false;
  }

  return result;
}

static string path_trace_devices_report(const vector<unique_ptr<PathTraceWork>> &path_trace_works)
{
  DeviceInfo device_info;
  device_info.type = DEVICE_MULTI;

  for (auto &&path_trace_work : path_trace_works) {
    device_info.multi_devices.push_back(path_trace_work->get_device()->info);
  }

  return device_info_list_report("Path tracing on", device_info, true);
}

static string denoiser_device_report(const Denoiser *denoiser)
{
  if (!denoiser) {
    return "";
  }

  if (!denoiser->get_params().use) {
    return "";
  }

  const Device *denoiser_device = denoiser->get_denoiser_device();
  if (!denoiser_device) {
    return "";
  }

  return device_info_list_report("Denoising on", denoiser_device->info, false);
}

string PathTrace::full_report() const
{
  string result = "Full path tracing report:\n";

  result += path_trace_devices_report(path_trace_works_);
  result += denoiser_device_report(denoiser_.get());

  /* Report from the render scheduler, which includes:
   * - Render mode (interactive, offline, headless)
   * - Adaptive sampling and denoiser parameters
   * - Breakdown of timing. */
  result += render_scheduler_.full_report();

  return result;
}

void PathTrace::set_guiding_params(const GuidingParams &guiding_params, const bool reset)
{
#if defined(WITH_PATH_GUIDING)
  if (guiding_params_.modified(guiding_params)) {
    guiding_params_ = guiding_params;

#  if !(OPENPGL_VERSION_MAJOR == 0 && OPENPGL_VERSION_MINOR <= 5)
#    define OPENPGL_USE_FIELD_CONFIG
#  endif

    if (guiding_params_.use) {
#  ifdef OPENPGL_USE_FIELD_CONFIG
      openpgl::cpp::FieldConfig field_config;
#  else
      PGLFieldArguments field_args;
#  endif
      switch (guiding_params_.type) {
        default:
        /* Parallax-aware von Mises-Fisher mixture models. */
        case GUIDING_TYPE_PARALLAX_AWARE_VMM: {
#  ifdef OPENPGL_USE_FIELD_CONFIG
          field_config.Init(
              PGL_SPATIAL_STRUCTURE_TYPE::PGL_SPATIAL_STRUCTURE_KDTREE,
              PGL_DIRECTIONAL_DISTRIBUTION_TYPE::PGL_DIRECTIONAL_DISTRIBUTION_PARALLAX_AWARE_VMM,
              guiding_params.deterministic);
#  else
          pglFieldArgumentsSetDefaults(
              field_args,
              PGL_SPATIAL_STRUCTURE_TYPE::PGL_SPATIAL_STRUCTURE_KDTREE,
              PGL_DIRECTIONAL_DISTRIBUTION_TYPE::PGL_DIRECTIONAL_DISTRIBUTION_PARALLAX_AWARE_VMM);
#  endif
          break;
        }
        /* Directional quad-trees. */
        case GUIDING_TYPE_DIRECTIONAL_QUAD_TREE: {
#  ifdef OPENPGL_USE_FIELD_CONFIG
          field_config.Init(
              PGL_SPATIAL_STRUCTURE_TYPE::PGL_SPATIAL_STRUCTURE_KDTREE,
              PGL_DIRECTIONAL_DISTRIBUTION_TYPE::PGL_DIRECTIONAL_DISTRIBUTION_QUADTREE,
              guiding_params.deterministic);
#  else
          pglFieldArgumentsSetDefaults(
              field_args,
              PGL_SPATIAL_STRUCTURE_TYPE::PGL_SPATIAL_STRUCTURE_KDTREE,
              PGL_DIRECTIONAL_DISTRIBUTION_TYPE::PGL_DIRECTIONAL_DISTRIBUTION_QUADTREE);
#  endif
          break;
        }
        /* von Mises-Fisher mixture models. */
        case GUIDING_TYPE_VMM: {
#  ifdef OPENPGL_USE_FIELD_CONFIG
          field_config.Init(PGL_SPATIAL_STRUCTURE_TYPE::PGL_SPATIAL_STRUCTURE_KDTREE,
                            PGL_DIRECTIONAL_DISTRIBUTION_TYPE::PGL_DIRECTIONAL_DISTRIBUTION_VMM,
                            guiding_params.deterministic);
#  else
          pglFieldArgumentsSetDefaults(
              field_args,
              PGL_SPATIAL_STRUCTURE_TYPE::PGL_SPATIAL_STRUCTURE_KDTREE,
              PGL_DIRECTIONAL_DISTRIBUTION_TYPE::PGL_DIRECTIONAL_DISTRIBUTION_VMM);
#  endif
          break;
        }
      }
#  ifdef OPENPGL_USE_FIELD_CONFIG
      field_config.SetSpatialStructureArgMaxDepth(16);
#  else
      field_args.deterministic = guiding_params.deterministic;
      reinterpret_cast<PGLKDTreeArguments *>(field_args.spatialSturctureArguments)->maxDepth = 16;
#  endif
      openpgl::cpp::Device *guiding_device = static_cast<openpgl::cpp::Device *>(
          device_->get_guiding_device());
      if (guiding_device) {
        guiding_sample_data_storage_ = make_unique<openpgl::cpp::SampleStorage>();
#  ifdef OPENPGL_USE_FIELD_CONFIG
        guiding_field_ = make_unique<openpgl::cpp::Field>(guiding_device, field_config);
#  else
        guiding_field_ = make_unique<openpgl::cpp::Field>(guiding_device, field_args);
#  endif
      }
      else {
        guiding_sample_data_storage_ = nullptr;
        guiding_field_ = nullptr;
      }
    }
    else {
      guiding_sample_data_storage_ = nullptr;
      guiding_field_ = nullptr;
    }
  }
  else if (reset) {
    if (guiding_field_) {
      guiding_field_->Reset();
    }
  }
#else
  (void)guiding_params;
  (void)reset;
#endif
}

void PathTrace::guiding_prepare_structures()
{
#if defined(WITH_PATH_GUIDING)
  const bool train = (guiding_params_.training_samples == 0) ||
                     (guiding_field_->GetIteration() < guiding_params_.training_samples);

  for (auto &&path_trace_work : path_trace_works_) {
    path_trace_work->guiding_init_kernel_globals(
        guiding_field_.get(), guiding_sample_data_storage_.get(), train);
  }

  if (train) {
    /* For training the guiding distribution we need to force the number of samples
     * per update to be limited, for reproducible results and reasonable training size.
     *
     * Idea: we could stochastically discard samples with a probability of 1/num_samples_per_update
     * we can then update only after the num_samples_per_update iterations are rendered. */
    render_scheduler_.set_limit_samples_per_update(4);
  }
  else {
    render_scheduler_.set_limit_samples_per_update(0);
  }
#endif
}

void PathTrace::guiding_update_structures()
{
#if defined(WITH_PATH_GUIDING)
  LOG_DEBUG << "Update path guiding structures";

  LOG_TRACE << "Number of surface samples: " << guiding_sample_data_storage_->GetSizeSurface();
  LOG_TRACE << "Number of volume samples: " << guiding_sample_data_storage_->GetSizeVolume();

  const size_t num_valid_samples = guiding_sample_data_storage_->GetSizeSurface() +
                                   guiding_sample_data_storage_->GetSizeVolume();

  /* we wait until we have at least 1024 samples */
  if (num_valid_samples >= 1024) {
    guiding_field_->Update(*guiding_sample_data_storage_);
    guiding_update_count++;

    LOG_TRACE << "Path guiding field valid: " << guiding_field_->Validate();

    guiding_sample_data_storage_->Clear();
  }
#endif
}

CCL_NAMESPACE_END
