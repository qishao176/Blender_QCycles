/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#pragma once

#include <functional>

#include "device/device.h"
#include "integrator/render_scheduler.h"
#include "scene/shader.h"
#include "scene/stats.h"
#include "session/buffers.h"
#include "session/cache_eviction.h"
#include "session/tile.h"

#include "util/progress.h"
#include "util/stats.h"
#include "util/thread.h"
#include "util/unique_ptr.h"

CCL_NAMESPACE_BEGIN

class BufferParams;
class DeepOutputDriver;
class Device;
class DeviceScene;
struct QueryTicket;
class DisplayDriver;
class OutputDriver;
class PathTrace;
struct PhotonViewportProbe;
struct FullBufferState;
class Progress;
class RenderBuffers;
class Scene;
class SceneParams;

/* Session Parameters */

class SessionParams {
 public:
  /* Device, which is chosen based on Blender Cycles preferences, as well as Scene settings and
   * command line arguments. */
  DeviceInfo device;
  /* Device from Cycles preferences for denoising. */
  DeviceInfo denoise_device;

  bool headless;
  bool background;

  int samples;
  bool use_sample_subset;
  int sample_subset_offset;
  int sample_subset_length;
  int pixel_size;
  int threads;

  /* Limit in seconds for how long path tracing is allowed to happen.
   * Zero means no limit is applied. */
  double time_limit;

  bool use_profiling;

  /* Zero paths selects the ordinary Cycles integrator. */
  int vcm_paths = 0;
  bool use_auto_tile;
  bool use_photon_cache;
  enum class PhotonOrder { BEFORE, AFTER, SIMULTANEOUS };
  PhotonOrder photon_order;
  int tile_size;
  /* Deep EXR buffer budget per device (MB); 0 disables tile-size clamping. */
  int deep_tile_budget_mb;

  bool use_resolution_divider;

  ShadingSystem shadingsystem;

  /* Session-specific temporary directory to store in-progress EXR files in. */
  string temp_dir;

  SessionParams()
  {
    headless = false;
    background = false;

    samples = 1024;
    use_sample_subset = false;
    sample_subset_offset = 0;
    sample_subset_length = 1024;
    pixel_size = 1;
    threads = 0;
    time_limit = 0.0;

    use_profiling = false;

    use_auto_tile = false;
    use_photon_cache = true;
    photon_order = PhotonOrder::BEFORE;
    tile_size = 2048;
    deep_tile_budget_mb = 1024;

    use_resolution_divider = true;

    shadingsystem = SHADINGSYSTEM_SVM;
  }

  bool modified(const SessionParams &params) const
  {
    /* Modified means we have to recreate the session, any parameter changes
     * that can be handled by an existing Session are omitted. */
    return !(device == params.device && headless == params.headless &&
             background == params.background && pixel_size == params.pixel_size &&
             vcm_paths == params.vcm_paths &&
             threads == params.threads && use_profiling == params.use_profiling &&
             use_auto_tile == params.use_auto_tile && tile_size == params.tile_size &&
             use_photon_cache == params.use_photon_cache && photon_order == params.photon_order &&
             use_resolution_divider == params.use_resolution_divider &&
             deep_tile_budget_mb == params.deep_tile_budget_mb &&
             shadingsystem == params.shadingsystem);
  }
};

/* Session
 *
 * This is the class that contains the session thread, running the render
 * control loop and dispatching tasks. */

class Session {
 public:
  unique_ptr<Device> device;
  /* Denoiser device. Could be the same as the path trace device. */
  unique_ptr<Device> denoise_device_;
  unique_ptr<Scene> scene;
  Progress progress;
  SessionParams params;
  Stats stats;
  Profiler profiler;

  /* Callback is invoked by tile manager whenever on-dist tiles storage file is closed after
   * writing. Allows an engine integration to keep track of those files without worry about
   * transferring the information when it needs to re-create session during rendering. */
  std::function<void(string_view)> full_buffer_written_cb;

  explicit Session(const SessionParams &params, const SceneParams &scene_params);
  ~Session();

  void start();

  /* When quick cancel is requested path tracing is cancels as soon as possible, without waiting
   * for the buffer to be uniformly sampled. */
  void cancel(bool quick = false);

  void draw();
  void wait();

  bool ready_to_reset();
  void reset(const SessionParams &session_params, const BufferParams &buffer_params);

  void set_pause(bool pause);
  /* Read a stable full-resolution snapshot after the current sample batch ends. */
  void release_query(uint64_t stream);
  void enqueue_query(std::shared_ptr<QueryTicket> ticket);
  std::atomic_uint64_t query_revision{0};
  enum class GPUQueryStatus { UNAVAILABLE, READY, READY_WITH_PHOTONS };
  std::atomic<GPUQueryStatus> gpu_query_status{GPUQueryStatus::UNAVAILABLE};
  void queue_display_plan(std::function<bool()> update);
  bool lpe_display_update_requested_ = false;
  std::function<bool()> lpe_pending_display_plan_;
  void set_navigating(bool navigating);

  void set_samples(const int samples);
  void set_time_limit(const double time_limit);

  void set_output_driver(unique_ptr<OutputDriver> driver);
  void set_display_driver(unique_ptr<DisplayDriver> driver);
  void set_deep_output_driver(unique_ptr<DeepOutputDriver> driver);

  /* Get deep output driver (for finalization). */
  DeepOutputDriver *get_deep_output_driver();

  double get_estimated_remaining_time() const;
  PhotonViewportProbe get_photon_viewport_probe() const;

  void device_free();

  /* Returns the rendering progress or 0 if no progress can be determined
   * (for example, when rendering with unlimited samples). */
  float get_progress();

  void collect_statistics(RenderStats *stats);

  /* --------------------------------------------------------------------
   * Full-frame on-disk storage.
   */

  /* Read given full-frame file from disk, perform needed processing and write it to the software
   * via the write callback. */
  void set_full_buffer_state(std::shared_ptr<FullBufferState> state);
  void process_full_buffer_from_disk(string_view filename);

 protected:
  struct DelayedReset {
    thread_mutex mutex;
    bool do_reset;
    SessionParams session_params;
    BufferParams buffer_params;
  } delayed_reset_;

  void thread_run();
  void thread_render();

  /* Check whether the session thread is in `SESSION_THREAD_RENDER` state.
   * Returns true if it is so. */
  bool is_session_thread_rendering();

  /* Update for the new iteration of the main loop in run implementation (run_cpu and run_gpu).
   *
   * Will take care of the following things:
   *  - Delayed reset
   *  - Scene update
   *  - Tile manager advance
   *  - Render scheduler work request
   *
   * The updates are done in a proper order with proper locking around them, which guarantees
   * that the device side of scene and render buffers are always in a consistent state.
   *
   * Returns render work which is to be rendered next. */
  RenderWork run_update_for_next_iteration();

  /* Wait for rendering to be unpaused, or for new tiles for render to arrive.
   * Returns true if new main render loop iteration is required after this function call.
   *
   * The `render_work` is the work which was scheduled by the render scheduler right before
   * checking the pause. */
  bool run_wait_for_work(const RenderWork &render_work);
  void run_query_slice(const std::shared_ptr<QueryTicket> &ticket);
  std::shared_ptr<QueryTicket> query_pending_;
  bool query_stopped_ = false;  /* Protected by pause_mutex_. */
  std::vector<uint64_t> query_releases_;

  void run_main_render_loop();

  bool update_scene(const bool reset_samples);

  void update_status_time(bool show_pause = false, bool show_done = false);

  bool delayed_reset_buffer_params();
  void update_buffers_for_params();

  int2 get_effective_tile_size() const;

  /* Get device used for denoising, may be the same as render device. */
  Device *denoise_device()
  {
    return (denoise_device_) ? denoise_device_.get() : device.get();
  }

  /* Session thread that performs rendering tasks decoupled from the thread
   * controlling the sessions. The thread is created and destroyed along with
   * the session. */
  unique_ptr<thread> session_thread_ = nullptr;
  thread_condition_variable session_thread_cond_;
  thread_mutex session_thread_mutex_;
  enum {
    SESSION_THREAD_WAIT,
    SESSION_THREAD_RENDER,
    SESSION_THREAD_END,
  } session_thread_state_ = SESSION_THREAD_WAIT;

  void run_photon_cache();
  void use_estimator_motion(bool photon);
  bool split_estimator_motion_ = false;
  bool photon_before_native_ = false;
  void configure_photon_tile_phase();
  bool photon_tile_sequential_ = false;
  bool photon_tile_second_ = false;
  bool photon_tile_has_samples_ = false;
  double photon_tile_first_start_ = 0.0;
  double photon_tile_second_time_limit_ = 0.0;
  bool photon_cache_enabled_ = false;
  bool photon_cache_completed_ = false;
  double photon_native_time_limit_ = 0.0;
  string photon_native_filename_;

  bool pause_ = false;
  bool new_work_added_ = false;

  thread_condition_variable pause_cond_;
  thread_mutex pause_mutex_;
  thread_mutex tile_mutex_;
  thread_mutex buffers_mutex_;

  TileManager tile_manager_;
  BufferParams buffer_params_;

  /* Manages when image cache eviction happens. */
  CacheEvictionManager eviction_manager_;

  /* Render scheduler is used to get work to be rendered with the current big tile. */
  RenderScheduler render_scheduler_;

  /* Path tracer object.
   *
   * Is a single full-frame path tracer for interactive viewport rendering.
   * A path tracer for the current big-tile for an offline rendering. */
  unique_ptr<PathTrace> path_trace_;
};

CCL_NAMESPACE_END
