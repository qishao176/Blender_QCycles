/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#include <cmath>
#include <cstring>

#include "device/cpu/device.h"
#include "device/device.h"
#include "integrator/path_trace.h"
#include "scene/background.h"
#include "scene/camera.h"
#include "scene/image.h"
#include "scene/integrator.h"
#include "scene/light.h"
#include "scene/mesh.h"
#include "scene/object.h"
#include "scene/scene.h"
#include "scene/shader_graph.h"
#include "session/buffers.h"
#include "session/deep_buffers.h"
#include "session/deep_output_driver.h"
#include "session/display_driver.h"
#include "session/output_driver.h"
#include "session/session.h"

#include "util/log.h"
#include "util/lpe_query.h"
#include "util/math.h"
#include "util/string.h"
#include "util/task.h"
#include "util/time.h"

CCL_NAMESPACE_BEGIN

namespace {

bool buffer_params_has_expanded_data_window(const BufferParams &buffer_params)
{
  return buffer_params.window_x != 0 || buffer_params.window_y != 0 ||
         buffer_params.width > buffer_params.full_width ||
         buffer_params.height > buffer_params.full_height ||
         buffer_params.full_x < 0 || buffer_params.full_y < 0 ||
         (buffer_params.full_x + buffer_params.width) > buffer_params.full_width ||
         (buffer_params.full_y + buffer_params.height) > buffer_params.full_height;
}

bool buffer_params_use_data_window_camera_resolution(const BufferParams &buffer_params)
{
  if (!buffer_params_has_expanded_data_window(buffer_params)) {
    return false;
  }

  /* A negative window offset marks full-frame overscan whose camera space expands with the
   * physical buffer. Cropped and viewport buffers instead keep global pixel coordinates in the
   * original full-frame camera space, even when their padded data window extends outside it. */
  return buffer_params.window_x < 0 || buffer_params.window_y < 0;
}

}  // namespace

Session::Session(const SessionParams &params_, const SceneParams &scene_params)
    : params(params_),
      eviction_manager_(params_.background),
      render_scheduler_(tile_manager_, params)
{
  TaskScheduler::init(params.threads);

  delayed_reset_.do_reset = false;

  pause_ = false;
  new_work_added_ = false;

  device = Device::create(params.device, stats, profiler, params_.headless);

  if (device->have_error()) {
    progress.set_error(device->error_message());
  }

  scene = make_unique<Scene>(scene_params, device.get());

  if (params.device == params.denoise_device) {
    /* Reuse render device. */
  }
  else {
    denoise_device_ = Device::create(params.denoise_device, stats, profiler, params_.headless);

    if (denoise_device_->have_error()) {
      progress.set_error(denoise_device_->error_message());
    }
  }

  /* Configure path tracer. */
  path_trace_ = make_unique<PathTrace>(device.get(),
                                       denoise_device(),
                                       scene->film,
                                       &scene->dscene,
                                       render_scheduler_,
                                       tile_manager_);
  path_trace_->set_progress(&progress);
  path_trace_->progress_update_cb = [&]() { update_status_time(); };

  tile_manager_.full_buffer_written_cb = [&](string_view filename) {
    photon_native_filename_ = filename;
    path_trace_->capture_full_buffer_context(filename);
    if (!full_buffer_written_cb) {
      return;
    }
    full_buffer_written_cb(filename);
  };

  /* Create session thread. */
  session_thread_ = make_unique<thread>([this] { thread_run(); });
}

Session::~Session()
{
  /* Cancel any ongoing render operation. */
  cancel();

  /* Signal session thread to end. */
  {
    const thread_scoped_lock session_thread_lock(session_thread_mutex_);
    session_thread_state_ = SESSION_THREAD_END;
  }
  session_thread_cond_.notify_all();

  /* Destroy session thread. */
  session_thread_->join();
  session_thread_.reset();
  if (query_pending_) {
    query_pending_->result.error = "Viewport session closed";
    query_pending_->finished = true;
  }

  /* Destroy path tracer, before the device. This is needed because destruction might need to
   * access device for device memory free.
   * TODO(sergey): Convert device to be unique_ptr, and rely on C++ to destruct objects in the
   * pre-defined order. */
  path_trace_.reset();

  /* Destroy scene and device. */
  scene.reset();
  denoise_device_.reset();
  device.reset();

  /* Stop task scheduler. */
  TaskScheduler::exit();
}

void Session::start()
{
  {
    /* Signal session thread to start rendering. */
    const thread_scoped_lock session_thread_lock(session_thread_mutex_);
    if (session_thread_state_ == SESSION_THREAD_RENDER) {
      /* Already rendering, nothing to do. */
      return;
    }
    const thread_scoped_lock pause_lock(pause_mutex_);
    query_stopped_ = false;
    session_thread_state_ = SESSION_THREAD_RENDER;
  }

  session_thread_cond_.notify_all();
}

void Session::cancel(bool quick)
{
  /* Cancel any long running device operations (e.g. shader compilations). */
  device->cancel();

  /* Check if session thread is rendering. */
  const bool rendering = is_session_thread_rendering();

  if (rendering) {
    /* Cancel path trace operations. */
    if (quick && path_trace_) {
      path_trace_->cancel();
    }

    /* Cancel other operations. */
    progress.set_cancel("Exiting");

    /* Signal unpause in case the render was paused. */
    {
      const thread_scoped_lock pause_lock(pause_mutex_);
      pause_ = false;
    }
    pause_cond_.notify_all();

    /* Wait for render thread to be cancelled or finished. */
    wait();
  }
}

bool Session::ready_to_reset()
{
  return path_trace_->ready_to_reset();
}

void Session::release_query(const uint64_t stream)
{
  if (!stream) {
    return;
  }
  if (params.background) {
    const thread_scoped_lock buffers_lock(buffers_mutex_);
    path_trace_->release_query_workspace(stream);
    return;
  }
  const thread_scoped_lock lock(pause_mutex_);
  if (query_stopped_) {
    return;
  }
  query_releases_.push_back(stream);
  pause_cond_.notify_all();
}

void Session::enqueue_query(std::shared_ptr<QueryTicket> ticket)
{
  ticket->revision = query_revision.load();
  ticket->slice.revision = ticket->revision;
  if (params.background) {
    /* The offline test driver borrows a completed, persistent render session. */
    run_query_slice(ticket);
    return;
  }
  const thread_scoped_lock lock(pause_mutex_);
  if (query_stopped_) {
    ticket->result.error = "Viewport session closed";
    ticket->finished = true;
    return;
  }
  if (query_pending_) {
    query_pending_->result.error = "Viewport query superseded";
    query_pending_->cancelled = true;
    query_pending_->finished = true;
  }
  query_pending_ = std::move(ticket);
  pause_cond_.notify_all();
}

void Session::run_query_slice(const std::shared_ptr<QueryTicket> &ticket)
{
  const thread_scoped_lock scene_lock(scene->mutex);
  const thread_scoped_lock buffers_lock(buffers_mutex_);
  if (device->have_error()) {
    ticket->result.error = device->error_message();
    progress.set_error(ticket->result.error);
  }
  else if (!ticket->cancelled && ticket->revision == query_revision.load() && !progress.get_cancel()) {
    try {
      path_trace_->query_slice(ticket->slice, ticket->result);
      if (device->have_error()) {
        ticket->result.error = device->error_message();
        progress.set_error(ticket->result.error);
      }
      /* During navigation the resident camera may still use a coarse raster. Only retry
       * until full resolution if the requested grid actually matches this viewport. */
      if (!params.background && ticket->result.error == "Viewport projection does not match the selection") {
        const bool cropped = buffer_params_use_data_window_camera_resolution(buffer_params_);
        const int width = cropped ? buffer_params_.width : buffer_params_.full_width;
        const int height = cropped ? buffer_params_.height : buffer_params_.full_height;
        if (ticket->slice.frame_w == max(1, width / params.pixel_size) &&
            ticket->slice.frame_h == max(1, height / params.pixel_size))
        {
          ticket->result.error = "Viewport GPU scene is not resident";
        }
      }
      auto &data = ticket->result.data;
      if (ticket->result.error.empty()) {
        /* The device index is the position in the synchronized scene arrays. Only
         * resolve names reached by this crop while holding the scene lock. */
        for (const auto &entry : data.images) {
          for (const QueryVertex &vertex : entry.first.first) {
            const int object = vertex[0], shader = vertex[1];
            if (object >= 0 && size_t(object) < scene->objects.size()) {
              data.objects.try_emplace(object, scene->objects[object]->name.c_str());
            }
            if (shader >= 0 && size_t(shader) < scene->shaders.size()) {
              data.materials.try_emplace(shader, scene->shaders[shader]->name.c_str());
            }
          }
        }
        data.world = scene->dscene.data.background.object_index;
      }
    }
    catch (const std::exception &e) {
      ticket->result.error = e.what();
    }
  }
  else {
    ticket->result.error = "Viewport scene changed";
  }
  ticket->finished = true;
}

void Session::queue_display_plan(std::function<bool()> update)
{
  path_trace_->request_lpe_display_update();
  {
    const thread_scoped_lock pause_lock(pause_mutex_);
    lpe_pending_display_plan_ = std::move(update);
    lpe_display_update_requested_ = true;
  }
  pause_cond_.notify_all();
}

/* Called with scene->mutex held, between completed batches. Reuse the mesh BVHs
 * and update the instance motion, camera and light sampling data together. */
void Session::use_estimator_motion(const bool photon)
{
  const bool corrected = photon ? scene->integrator->get_use_photon_motion_blur_correction() :
                                  scene->integrator->get_use_relative_motion_blur();
  if (bool(scene->dscene.data.integrator.use_motion_blur_correction) != corrected) {
    scene->set_motion_blur_correction(corrected);
    scene->update(progress);
  }
}

void Session::configure_photon_tile_phase()
{
  const bool photon = photon_before_native_ != photon_tile_second_;
  path_trace_->set_photon_estimator(photon ? 1 : 0, photon_tile_second_);
  const int samples = photon ? scene->dscene.data.integrator.photon_samples : params.samples;
  /* Keep an empty subset phase alive long enough to publish the other estimator. */
  render_scheduler_.set_sample_params(
      params.use_sample_subset ? max(samples, params.sample_subset_offset + 1) : samples,
      params.use_sample_subset,
      params.sample_subset_offset,
      params.sample_subset_length);
  if (photon_tile_first_start_ == 0.0) {
    photon_tile_first_start_ = time_dt();
  }
  render_scheduler_.set_time_limit(photon_tile_second_ ? photon_tile_second_time_limit_ :
                                                         params.time_limit * 0.5);
}

void Session::run_photon_cache()
{
  if (split_estimator_motion_) {
    const thread_scoped_lock scene_lock(scene->mutex);
    use_estimator_motion(true);
  }
  const thread_scoped_lock buffers_lock(buffers_mutex_);
  BufferParams photon_params = buffer_params_;
  const int maximum = scene->dscene.data.integrator.photon_samples;
  photon_params.samples = params.use_sample_subset ?
                              min(maximum, params.sample_subset_offset + params.sample_subset_length) :
                              maximum;
  if (params.time_limit > 0.0) {
    double total_time, render_time;
    progress.get_time(total_time, render_time);
    const double total_budget = params.time_limit * tile_manager_.get_num_tiles();
    render_scheduler_.set_time_limit(photon_before_native_ ?
                                         total_budget * 0.5 :
                                         max(0.001, total_budget - render_time));
  }
  path_trace_->render_photon_cache(photon_params, photon_native_filename_);
  photon_cache_completed_ = true;
  if (photon_before_native_ && params.time_limit > 0.0) {
    double total_time, render_time;
    progress.get_time(total_time, render_time);
    photon_native_time_limit_ = max(0.001, params.time_limit -
                                              render_time / tile_manager_.get_num_tiles());
  }
  if (device->have_error()) {
    progress.set_error(device->error_message());
  }
}

void Session::run_main_render_loop()
{
  path_trace_->zero_display();

  while (true) {
    RenderWork render_work = run_update_for_next_iteration();

    const bool did_cancel = progress.get_cancel();

    if (!render_work) {
      if (LOG_IS_ON(LOG_LEVEL_INFO)) {
        if (did_cancel) {
          LOG_INFO << "Rendering was canceled.";
        }
        else {
          double total_time;
          double render_time;
          progress.get_time(total_time, render_time);
          LOG_INFO << "Rendering in main loop is done in " << render_time << " seconds.";
          LOG_INFO << path_trace_->full_report();
        }
      }

      if (params.background) {
        if (photon_cache_enabled_ && !photon_cache_completed_ && !did_cancel &&
            !progress.get_error() && !device->have_error()) {
          run_photon_cache();
        }
        /* All scene-dependent estimators finish before device_free and postprocessing. */
        progress.set_status("Finished");
        break;
      }
    }

    if (did_cancel) {
      render_scheduler_.render_work_reschedule_on_cancel(render_work);
      if (!render_work) {
        break;
      }
    }
    else if (run_wait_for_work(render_work)) {
      continue;
    }

    /* Stop rendering if error happened during scene update or other step of preparing scene
     * for render. */
    if (device->have_error()) {
      progress.set_error(device->error_message());
      break;
    }

    if (photon_cache_enabled_ && photon_before_native_ && !photon_cache_completed_) {
      run_photon_cache();
      /* The native work was scheduled, but not executed. */
      render_scheduler_.reset(buffer_params_);
      continue;
    }

    {
      /* buffers mutex is locked entirely while rendering each
       * sample, and released/reacquired on each iteration to allow
       * reset and draw in between */
      const thread_scoped_lock buffers_lock(buffers_mutex_);

      /* update status and timing */
      update_status_time();

      /* render */
      path_trace_->render(render_work);

      /* update status and timing */
      update_status_time();

      /* Stop rendering if error happened during path tracing. */
      if (device->have_error()) {
        progress.set_error(device->error_message());
        break;
      }
    }

    progress.set_update();

    if (did_cancel) {
      break;
    }
  }
}

void Session::thread_run()
{
  while (true) {
    {
      thread_scoped_lock session_thread_lock(session_thread_mutex_);

      if (session_thread_state_ == SESSION_THREAD_WAIT) {
        /* Continue waiting for any signal from the main thread. */
        session_thread_cond_.wait(session_thread_lock);
        continue;
      }
      if (session_thread_state_ == SESSION_THREAD_END) {
        /* End thread immediately. */
        break;
      }
    }

    /* Execute a render. */
    thread_render();

    /* The viewport pump also serves queries when its preview has reached the sample limit.
     * Leaving the pump (including a device error) means no thread can service a queued
     * query. Close admission and finish the remaining ticket before publishing WAIT. */
    if (!params.background) {
      {
        const thread_scoped_lock pause_lock(pause_mutex_);
        gpu_query_status = GPUQueryStatus::UNAVAILABLE;
        query_stopped_ = true;
        if (query_pending_) {
          query_pending_->result.error = progress.get_error() ? progress.get_error_message() :
                                                                "Viewport session closed";
          query_pending_->finished = true;
          query_pending_.reset();
        }
        query_releases_.clear();
      }
      const thread_scoped_lock buffers_lock(buffers_mutex_);
      path_trace_->release_query_workspace();
    }

    /* Go back from rendering to waiting. */
    {
      const thread_scoped_lock session_thread_lock(session_thread_mutex_);
      if (session_thread_state_ == SESSION_THREAD_RENDER) {
        session_thread_state_ = SESSION_THREAD_WAIT;
      }
    }
    session_thread_cond_.notify_all();
  }

  /* Flush any remaining operations and destroy display driver here. This ensure
   * graphics API resources are created and destroyed all in the session thread,
   * which can avoid problems contexts and multiple threads. */
  path_trace_->flush_display();
  path_trace_->set_display_driver(nullptr);
}

void Session::thread_render()
{
  if (params.use_profiling && (params.device.type == DEVICE_CPU)) {
    profiler.start();
  }

  /* session thread loop */
  progress.set_status("Waiting for render to start");

  /* run */
  if (!progress.get_cancel()) {
    /* reset number of rendered samples */
    progress.reset_sample();

    run_main_render_loop();
  }

  profiler.stop();

  /* progress update */
  if (progress.get_cancel()) {
    progress.set_status(progress.get_cancel_message());
  }
  else {
    progress.set_update();
  }
}

bool Session::is_session_thread_rendering()
{
  const thread_scoped_lock session_thread_lock(session_thread_mutex_);
  return (session_thread_state_ == SESSION_THREAD_RENDER);
}

RenderWork Session::run_update_for_next_iteration()
{
  RenderWork render_work;

  thread_scoped_lock scene_lock(scene->mutex);

  /* Perform delayed reset if requested. */
  const bool reset_buffers = delayed_reset_buffer_params();

  /* Update scene */
  if (reset_buffers) {
    scene->set_motion_blur_correction(scene->integrator->get_use_relative_motion_blur());
  }
  const bool reset_scene = update_scene(delayed_reset_.do_reset);

  /* Update buffers for new parameters. After scene update which influences the passes used. */
  bool have_tiles = true;
  bool switched_to_new_tile = false;

  if (reset_buffers) {
    update_buffers_for_params();

    /* After reset make sure the tile manager is at the first big tile. */
    have_tiles = tile_manager_.next();
    switched_to_new_tile = true;

    eviction_manager_.reset();
  }

  /* Update denoiser settings. */
  {
    const DenoiseParams denoise_params = scene->integrator->get_denoise_params();
    path_trace_->set_denoiser_params(denoise_params);
  }

  /* Update adaptive sampling. */
  {
    const AdaptiveSampling adaptive_sampling =
        scene->dscene.data.film.photon_segment_stride ?
            scene->integrator->get_adaptive_sampling() :
            scene->integrator->get_native_adaptive_sampling();
    path_trace_->set_adaptive_sampling(adaptive_sampling);
  }

  /* Update path guiding. */
  {
    const GuidingParams guiding_params = scene->integrator->get_guiding_params(device.get());
    const bool guiding_reset = (guiding_params.use) ? reset_scene : false;
    path_trace_->set_guiding_params(guiding_params, guiding_reset);
  }

  const int render_samples = !photon_cache_enabled_ &&
                                     scene->dscene.data.film.photon_segment_stride &&
                                     scene->dscene.data.integrator.use_photon_caustics ?
                                 max(params.samples,
                                     scene->dscene.data.integrator.photon_samples) :
                                 (photon_cache_enabled_ && params.use_sample_subset ?
                                      max(params.samples, params.sample_subset_offset + 1) : params.samples);
  render_scheduler_.set_sample_params(render_samples,
                                      params.use_sample_subset,
                                      params.sample_subset_offset,
                                      params.sample_subset_length);
  render_scheduler_.set_time_limit(photon_cache_enabled_ ? photon_native_time_limit_ :
                                                           params.time_limit);

  if (photon_tile_sequential_) {
    configure_photon_tile_phase();
  }

  while (have_tiles) {
    render_work = render_scheduler_.get_render_work();
    if (render_work) {
      break;
    }

    if (photon_tile_sequential_ && !photon_tile_second_) {
      photon_tile_second_ = true;
      photon_tile_second_time_limit_ = params.time_limit > 0.0 ?
                                           max(0.001,
                                               params.time_limit -
                                                   (time_dt() - photon_tile_first_start_)) :
                                           0.0;
      configure_photon_tile_phase();
      render_scheduler_.reset_for_next_tile();
      continue;
    }

    progress.add_finished_tile(false);

    have_tiles = tile_manager_.next();
    if (have_tiles) {
      if (photon_tile_sequential_) {
        photon_tile_second_ = false;
        photon_tile_has_samples_ = false;
        photon_tile_first_start_ = 0.0;
        configure_photon_tile_phase();
      }
      render_scheduler_.reset_for_next_tile();
      switched_to_new_tile = true;
    }
  }

  /* Evict unused image tiles periodically. */
  if (eviction_manager_.need_eviction(!render_work, switched_to_new_tile)) {
    scene->image_manager->evict_unused(device.get(), scene.get());
  }

  if (render_work) {
    if (split_estimator_motion_) {
      use_estimator_motion(photon_tile_sequential_ && (photon_before_native_ != photon_tile_second_));
    }
    if (photon_tile_sequential_) {
      if (photon_tile_second_ && photon_tile_has_samples_) {
        render_work.init_render_buffers = false;
      }
      if (render_work.path_trace.num_samples) {
        photon_tile_has_samples_ = true;
      }
    }
    const scoped_timer update_timer;

    if (switched_to_new_tile) {
      BufferParams tile_params = buffer_params_;

      const Tile &tile = tile_manager_.get_current_tile();

      tile_params.width = tile.width;
      tile_params.height = tile.height;

      tile_params.window_x = tile.window_x;
      tile_params.window_y = tile.window_y;
      tile_params.window_width = tile.window_width;
      tile_params.window_height = tile.window_height;

      tile_params.full_x = tile.x + buffer_params_.full_x;
      tile_params.full_y = tile.y + buffer_params_.full_y;
      tile_params.full_width = buffer_params_.full_width;
      tile_params.full_height = buffer_params_.full_height;

      tile_params.update_offset_stride();

      path_trace_->reset(buffer_params_, tile_params, reset_buffers);
    }

    /* Update camera if dimensions changed for progressive render. the camera
     * knows nothing about progressive or cropped rendering, it just gets the
     * image dimensions passed in. */
    const float resolution = render_work.resolution_divider;
    const bool use_data_window_resolution = buffer_params_use_data_window_camera_resolution(
        buffer_params_);
    const int width = max(
        1, int((use_data_window_resolution ? buffer_params_.width : buffer_params_.full_width) /
               resolution));
    const int height = max(
        1, int((use_data_window_resolution ? buffer_params_.height : buffer_params_.full_height) /
               resolution));

    scene->update_camera_resolution(progress, width, height);

    /* Unlock scene mutex before loading denoiser kernels, since that may attempt to activate
     * graphics interop, which can deadlock when the scene mutex is still being held. */
    const bool query_uses_photons = scene->dscene.data.integrator.use_photon_caustics;
    scene_lock.unlock();

    path_trace_->load_kernels();
    path_trace_->alloc_work_memory();

    /* Wait for device to be ready (e.g. finish any background compilations). */
    string device_status;
    while (!device->is_ready(device_status)) {
      progress.set_status(device_status);
      if (progress.get_cancel()) {
        break;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }
    if (!params.background) {
      const bool ready = !progress.get_cancel() && !device->have_error() && path_trace_->supports_gpu_query();
      gpu_query_status = !ready ? GPUQueryStatus::UNAVAILABLE :
                         query_uses_photons ? GPUQueryStatus::READY_WITH_PHOTONS : GPUQueryStatus::READY;
    }

    progress.add_skip_time(update_timer, params.background);
  }

  return render_work;
}

bool Session::run_wait_for_work(const RenderWork &render_work)
{
  /* In an offline rendering there is no pause, and no tiles will mean the job is fully done. */
  if (params.background) {
    return false;
  }

  thread_scoped_lock pause_lock(pause_mutex_);

  if (!pause_ && render_work && !lpe_display_update_requested_ && !query_pending_ &&
      query_releases_.empty()) {
    /* Rendering is not paused and there is work to be done. No need to wait for anything. */
    return false;
  }

  const bool no_work = !render_work;
  update_status_time(pause_, no_work);

  /* Only leave the loop when rendering is not paused. But even if the current render is
   * un-paused but there is nothing to render keep waiting until new work is added. */
  while (!progress.get_cancel()) {
    const scoped_timer pause_timer;

    if (lpe_display_update_requested_ ||
        path_trace_->lpe_denoise_wait_time() == std::chrono::milliseconds::zero()) {
      lpe_display_update_requested_ = false;
      auto update = std::move(lpe_pending_display_plan_);
      /* get_render_work() has already scheduled the next batch, possibly at a
       * new resolution. Let that batch publish its image instead of refreshing
       * the previous buffers in the middle of the transition. Paused edits can
       * still use the last completed batch. */
      const bool refresh = pause_ || !render_work;
      pause_lock.unlock();
      {
        const thread_scoped_lock scene_lock(scene->mutex);
        const thread_scoped_lock buffers_lock(buffers_mutex_);
        if (update) update();
        if (refresh) path_trace_->refresh_display();
      }
      progress.set_update();
      pause_lock.lock();
      if (!refresh) break;
      continue;
    }

    if (!query_releases_.empty()) {
      auto streams = std::move(query_releases_);
      query_releases_.clear();
      pause_lock.unlock();
      {
        const thread_scoped_lock buffers_lock(buffers_mutex_);
        for (const uint64_t stream : streams) {
          path_trace_->release_query_workspace(stream);
        }
      }
      pause_lock.lock();
      continue;
    }
    if (query_pending_ && !eviction_manager_.is_navigating() && !delayed_reset_.do_reset) {
      auto ticket = std::move(query_pending_);
      pause_lock.unlock();
      run_query_slice(ticket);
      pause_lock.lock();
      if (!pause_ && render_work) break;
      continue;
    }
    if (!pause_ && (render_work || new_work_added_ || delayed_reset_.do_reset)) {
      break;
    }

    const std::chrono::milliseconds eviction_wait = eviction_manager_.wait_time(!render_work);
    if (eviction_wait == std::chrono::milliseconds::zero()) {
      /* Break out of the loop for cache eviction. */
      break;
    }
    const std::chrono::milliseconds wait_time = std::min(
        eviction_wait, path_trace_->lpe_denoise_wait_time());
    if (wait_time == std::chrono::milliseconds::zero()) continue;

    /* Wait for either pause state changed, extra samples added to render, or idle
     * timer before performing eviction. */
    if (wait_time == std::chrono::milliseconds::max()) {
      pause_cond_.wait(pause_lock);
    }
    else {
      pause_cond_.wait_for(pause_lock, wait_time);
    }

    if (pause_) {
      progress.add_skip_time(pause_timer, params.background);
    }

    update_status_time(pause_, no_work);
    progress.set_update();
  }

  new_work_added_ = false;

  return no_work;
}

void Session::draw()
{
  path_trace_->draw();
}

int2 Session::get_effective_tile_size() const
{
  const int image_width = buffer_params_.width;
  const int image_height = buffer_params_.height;

  if (!params.use_auto_tile) {
    return make_int2(image_width, image_height);
  }

  const int64_t image_area = static_cast<int64_t>(image_width) * image_height;

  /* TODO(sergey): Take available memory into account for non-deep renders so we can
   * avoid tiling when there is enough memory. */
  /* Deep EXR uses a user-configured tile budget. */

  int tile_size = tile_manager_.compute_render_tile_size(params.tile_size);
  bool deep_tile_clamp_active = false;

  if (scene && scene->film->get_use_deep_output() && params.deep_tile_budget_mb > 0) {
    const int max_samples = deep_effective_max_samples(scene->film, scene->integrator);
    const int physical_channels = scene->film->get_deep_simple_mode() ? -1 :
                                      int(scene->film->deep_lpe_passes.size());
    const int photon_bins = physical_channels < 0 && scene->integrator->get_use_photon_caustics() ?
                                clamp(scene->film->get_deep_photon_samples(), 8, 256) :
                                0;

    size_t bytes_per_pixel = 0;
    if (deep_compute_buffer_bytes(1, 1, max_samples, bytes_per_pixel, photon_bins, physical_channels) &&
        bytes_per_pixel > 0)
    {
      const size_t deep_budget_bytes = static_cast<size_t>(params.deep_tile_budget_mb) *
                                       1024 * 1024;
      const size_t max_pixels = deep_budget_bytes / bytes_per_pixel;
      if (max_pixels > 0) {
        const double max_pixels_double = static_cast<double>(max_pixels);
        const int max_tile_unaligned = max(
            8, static_cast<int>(std::floor(std::sqrt(max_pixels_double))));
        int max_tile = max_tile_unaligned;
        if (max_tile >= TileManager::IMAGE_TILE_SIZE) {
          max_tile = (max_tile / TileManager::IMAGE_TILE_SIZE) * TileManager::IMAGE_TILE_SIZE;
        }

        if (max_tile < tile_size) {
          deep_tile_clamp_active = true;
          tile_size = max_tile;

          size_t estimated_bytes = 0;
          const string budget_str = string_human_readable_size(deep_budget_bytes);
          if (deep_compute_buffer_bytes(
                  tile_size, tile_size, max_samples, estimated_bytes, photon_bins, physical_channels))
          {
            const string estimated_str = string_human_readable_size(estimated_bytes);
            LOG_INFO << "Deep EXR: clamping tile size to " << tile_size << " (budget "
                     << budget_str << ", estimated " << estimated_str << ", max samples "
                     << max_samples << ")";
          }
          else {
            /* Overflow while estimating deep buffer bytes. */
            LOG_INFO << "Deep EXR: clamping tile size to " << tile_size << " (budget "
                     << budget_str << ", max samples " << max_samples << ")";
          }
        }
      }
    }
  }

  const int64_t actual_tile_area = static_cast<int64_t>(tile_size) * tile_size;

  if (!deep_tile_clamp_active && actual_tile_area >= image_area &&
      image_width <= TileManager::MAX_TILE_SIZE && image_height <= TileManager::MAX_TILE_SIZE)
  {
    return make_int2(image_width, image_height);
  }

  return make_int2(tile_size, tile_size);
}

bool Session::delayed_reset_buffer_params()
{
  /* Reset buffer parameters, delayed from when we got the reset call so we can complete
   * rendering the sample. Otherwise e.g. viewport navigation might reset without ever
   * finishing anything. */
  const thread_scoped_lock reset_lock(delayed_reset_.mutex);
  if (!delayed_reset_.do_reset) {
    return false;
  }

  const thread_scoped_lock buffers_lock(buffers_mutex_);
  delayed_reset_.do_reset = false;
  path_trace_->release_query_workspace();

  params = delayed_reset_.session_params;
  path_trace_->reset_vcm(params.vcm_paths, params.samples);
  buffer_params_ = delayed_reset_.buffer_params;

  /* Store parameters used for buffers access outside of scene graph. */
  buffer_params_.samples = min(params.samples, Integrator::MAX_SAMPLES);
  buffer_params_.exposure = scene->film->get_exposure();
  buffer_params_.use_approximate_shadow_catcher =
      scene->film->get_use_approximate_shadow_catcher();
  buffer_params_.use_transparent_background = scene->background->get_transparent();

  /* Tile and work scheduling. */
  tile_manager_.reset_scheduling(buffer_params_, get_effective_tile_size());

  return true;
}

void Session::update_buffers_for_params()
{
  split_estimator_motion_ = scene->dscene.data.integrator.use_photon_caustics &&
                            scene->needs_separate_motion();
  const bool deep = scene->film->get_use_deep_output();
  const bool sequential = params.background &&
                          (split_estimator_motion_ ||
                           params.photon_order != SessionParams::PhotonOrder::SIMULTANEOUS);
  photon_before_native_ = !deep && params.photon_order != SessionParams::PhotonOrder::AFTER;
  photon_cache_enabled_ = sequential && !deep && params.use_photon_cache &&
                          scene->dscene.data.film.photon_segment_stride &&
                          scene->dscene.data.integrator.use_photon_caustics;
  photon_tile_sequential_ = sequential && !photon_cache_enabled_ &&
                            scene->dscene.data.film.photon_segment_stride &&
                            scene->dscene.data.integrator.use_photon_caustics;
  photon_tile_second_ = false;
  photon_tile_has_samples_ = false;
  photon_tile_first_start_ = 0.0;
  path_trace_->set_photon_estimator(-1, true);
  photon_native_filename_.clear();
  photon_cache_completed_ = false;
  photon_native_time_limit_ = params.time_limit * 0.5;
  path_trace_->set_photon_cache_native(photon_cache_enabled_);
  tile_manager_.set_force_file(photon_cache_enabled_);
  const int samples = !photon_cache_enabled_ && scene->dscene.data.film.photon_segment_stride &&
                              scene->dscene.data.integrator.use_photon_caustics ?
                          max(params.samples, scene->dscene.data.integrator.photon_samples) :
                          params.samples;
  buffer_params_.samples = min(photon_cache_enabled_ ?
                                   max(params.samples, scene->dscene.data.integrator.photon_samples) : samples,
                               Integrator::MAX_SAMPLES);
  render_scheduler_.set_sample_params(
      samples, params.use_sample_subset, params.sample_subset_offset, params.sample_subset_length);
  render_scheduler_.reset(buffer_params_);

  /* Update for new state of scene and passes. */
  buffer_params_.update_passes(scene->passes);
  tile_manager_.update(buffer_params_, scene.get());

  /* Update temp directory on reset.
   * This potentially allows to finish the existing rendering with a previously configure
   * temporary
   * directory in the host software and switch to a new temp directory when new render starts. */
  tile_manager_.set_temp_dir(params.temp_dir);
  if (photon_cache_enabled_) {
    photon_native_filename_ = tile_manager_.get_full_buffer_filename();
  }
  path_trace_->set_photon_cache_context(photon_native_filename_, buffer_params_);

  /* Progress. */
  progress.reset_sample();
  uint64_t progress_samples = buffer_params_.samples;
  if (photon_cache_enabled_ || photon_tile_sequential_) {
    const auto subset_samples = [&](const int maximum) {
      return params.use_sample_subset ?
                 max(0, min(maximum - params.sample_subset_offset, params.sample_subset_length)) :
                 maximum;
    };
    progress_samples = uint64_t(subset_samples(params.samples)) +
                       subset_samples(scene->dscene.data.integrator.photon_samples);
  }
  progress.set_total_pixel_samples(static_cast<uint64_t>(buffer_params_.width) *
                                   buffer_params_.height * progress_samples);

  if (!params.background) {
    progress.set_start_time();
  }
  const double time_limit = params.time_limit * ((double)tile_manager_.get_num_tiles());
  progress.set_render_start_time();
  progress.set_time_limit(time_limit);
}

void Session::reset(const SessionParams &session_params, const BufferParams &buffer_params)
{
  gpu_query_status = GPUQueryStatus::UNAVAILABLE;
  ++query_revision;
  {
    const thread_scoped_lock reset_lock(delayed_reset_.mutex);
    const thread_scoped_lock pause_lock(pause_mutex_);

    delayed_reset_.do_reset = true;
    delayed_reset_.session_params = session_params;
    delayed_reset_.buffer_params = buffer_params;

    scene->scene_updated_while_loading_kernels = true;

    path_trace_->cancel();
  }

  pause_cond_.notify_all();
}

void Session::set_samples(const int samples)
{
  if (samples == params.samples) {
    return;
  }

  params.samples = samples;

  {
    const thread_scoped_lock pause_lock(pause_mutex_);
    new_work_added_ = true;
  }

  pause_cond_.notify_all();
}

void Session::set_time_limit(const double time_limit)
{
  if (time_limit == params.time_limit) {
    return;
  }

  params.time_limit = time_limit;

  {
    const thread_scoped_lock pause_lock(pause_mutex_);
    new_work_added_ = true;
  }

  pause_cond_.notify_all();
}

void Session::set_pause(bool pause)
{
  bool notify = false;

  {
    const thread_scoped_lock pause_lock(pause_mutex_);

    if (pause != pause_) {
      pause_ = pause;
      notify = true;
    }
  }

  if (is_session_thread_rendering()) {
    if (notify) {
      pause_cond_.notify_all();
    }
  }
  else if (pause_) {
    update_status_time(pause_);
  }
}

void Session::set_navigating(bool navigating)
{
  path_trace_->set_lpe_navigating(navigating);
  if (navigating != eviction_manager_.is_navigating()) {
    /* A queued query can be the only work on an idle viewport without an LPE graph.
     * Change its wait condition under the same mutex used by run_wait_for_work. */
    const thread_scoped_lock lock(pause_mutex_);
    eviction_manager_.set_navigating(navigating);
    if (!navigating) {
      pause_cond_.notify_all();
    }
  }
}

void Session::set_output_driver(unique_ptr<OutputDriver> driver)
{
  path_trace_->set_output_driver(std::move(driver));
}

void Session::set_display_driver(unique_ptr<DisplayDriver> driver)
{
  path_trace_->set_display_driver(std::move(driver));
}

void Session::set_deep_output_driver(unique_ptr<DeepOutputDriver> driver)
{
  path_trace_->set_deep_output_driver(std::move(driver));
}

DeepOutputDriver *Session::get_deep_output_driver()
{
  return path_trace_->get_deep_output_driver();
}

PhotonViewportProbe Session::get_photon_viewport_probe() const
{
  return path_trace_->get_photon_viewport_probe();
}

double Session::get_estimated_remaining_time() const
{
  const double completed = progress.get_progress();
  if (completed == 0.0) {
    return 0.0;
  }

  double total_time;
  double render_time;
  progress.get_time(total_time, render_time);
  double remaining = (1.0 - (double)completed) * (render_time / (double)completed);

  const double time_limit = params.time_limit * ((double)tile_manager_.get_num_tiles());
  if (time_limit != 0.0) {
    remaining = min(remaining, max(time_limit - render_time, 0.0));
  }

  return remaining;
}

void Session::wait()
{
  /* Wait until session thread either is waiting or ending. */
  while (true) {
    thread_scoped_lock session_thread_lock(session_thread_mutex_);
    if (session_thread_state_ != SESSION_THREAD_RENDER) {
      break;
    }
    session_thread_cond_.wait(session_thread_lock);
  }
}

bool Session::update_scene(const bool reset_samples)
{
  /* Session samples include viewport and per-view-layer overrides. Keep this native
   * budget before extending the sequence to accommodate independent photon work. */
  scene->integrator->set_photon_caustics_native_samples(params.samples);

  /* Update number of samples in the integrator.
   * Ideally this would need to happen once in `Session::set_samples()`, but the issue there is
   * the initial configuration when Session is created where the `set_samples()` is not used.
   *
   * NOTE: Unless reset was requested only allow increasing number of samples. */
  int sequence_samples = params.samples;
  if (scene->integrator->get_use_photon_caustics() &&
      scene->integrator->get_photon_caustics_separate_sampling() &&
      scene->integrator->get_photon_caustics_independent_samples() &&
      !scene->film->get_use_deep_output())
  {
    sequence_samples = max(sequence_samples, scene->integrator->get_photon_caustics_samples());
  }
  if (reset_samples || scene->integrator->get_aa_samples() < sequence_samples) {
    scene->integrator->set_aa_samples(sequence_samples);
  }

  scene->integrator->set_use_sample_subset(params.use_sample_subset);
  scene->integrator->set_sample_subset_offset(params.sample_subset_offset);
  scene->integrator->set_sample_subset_length(params.sample_subset_length);

  /* When multiple tiles are used SAMPLE_COUNT pass is used to keep track of possible partial
   * tile results. Deep output also needs per-pixel sample counts to reconstruct hard-surface
   * edge coverage after sorting deep hits by depth. */
  scene->film->set_use_sample_count(tile_manager_.has_multiple_tiles() ||
                                    scene->film->get_use_deep_output());

  const bool reset = scene->need_reset(false);

  if (scene->update(progress)) {
    profiler.reset(scene->shaders.size(), scene->objects.size());
  }

  return reset;
}

static string status_append(const string &status, const string &suffix)
{
  string prefix = status;
  if (!prefix.empty()) {
    prefix += ", ";
  }
  return prefix + suffix;
}

void Session::update_status_time(bool show_pause, bool show_done)
{
  string status;
  string substatus;

  const int current_tile = progress.get_rendered_tiles();
  const int num_tiles = tile_manager_.get_num_tiles();

  const int current_sample = progress.get_current_sample();
  const int num_samples = render_scheduler_.get_num_samples();

  /* TIle. */
  if (tile_manager_.has_multiple_tiles()) {
    substatus = status_append(substatus,
                              string_printf("Rendered %d/%d Tiles", current_tile, num_tiles));
  }

  /* Sample. */
  if (!params.background && num_samples == Integrator::MAX_SAMPLES) {
    substatus = status_append(substatus, string_printf("Sample %d", current_sample));
  }
  else {
    substatus = status_append(substatus,
                              string_printf("Sample %d/%d", current_sample, num_samples));
  }

  /* Append any device-specific status (such as background kernel optimization) */
  string device_status;
  if (device->is_ready(device_status) && !device_status.empty()) {
    substatus += string_printf(" (%s)", device_status.c_str());
  }

  /* TODO(sergey): Denoising status from the path trace. */

  if (show_pause) {
    status = "Rendering Paused";
  }
  else if (show_done) {
    status = "Rendering Done";
    progress.set_end_time(); /* Save end time so that further calls to get_time are accurate. */
  }
  else {
    status = substatus;
    substatus.clear();
  }

  progress.set_status(status, substatus);
}

void Session::device_free()
{
  gpu_query_status = GPUQueryStatus::UNAVAILABLE;
  scene->device_free();
  path_trace_->device_free();
}

void Session::collect_statistics(RenderStats *render_stats)
{
  scene->collect_statistics(render_stats);
  if (params.use_profiling && (params.device.type == DEVICE_CPU)) {
    render_stats->collect_profiling(scene.get(), profiler);
  }
}

/* --------------------------------------------------------------------
 * Full-frame on-disk storage.
 */

void Session::set_full_buffer_state(std::shared_ptr<FullBufferState> state)
{
  path_trace_->set_full_buffer_state(std::move(state));
}

void Session::process_full_buffer_from_disk(string_view filename)
{
  path_trace_->process_full_buffer_from_disk(filename);
}

CCL_NAMESPACE_END
