/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#pragma once

#include <atomic>
#include <chrono>
#include <cstdio>
#include <functional>
#include <memory>

#include "integrator/denoiser.h"
#include "integrator/guiding.h"
#include "integrator/pass_accessor.h"
#include "integrator/path_trace_work.h"
#include "integrator/work_balancer.h"

#include "session/buffers.h"
#include "session/deep_output_driver.h"

#include "util/guiding.h"  // IWYU pragma: keep
#include "util/half.h"
#include "util/map.h"
#include "util/thread.h"
#include "util/unique_ptr.h"
#include "util/vector.h"

CCL_NAMESPACE_BEGIN

class AdaptiveSampling;
class Device;
class DeviceScene;
class DedicatedTaskPool;
class DisplayDriver;
class Film;
struct FullBufferState;
class RenderBuffers;
class RenderScheduler;
class RenderWork;
class PathTraceDisplay;
class OutputDriver;
class Progress;
class TileManager;

/* PathTrace class takes care of kernel graph and scheduling on a (multi)device. It takes care of
 * all the common steps of path tracing which are not device-specific. The list of tasks includes
 * but is not limited to:
 *  - Kernel graph.
 *  - Scheduling logic.
 *  - Queues management.
 *  - Adaptive stopping. */
struct QuerySlice;
struct QuerySliceResult;

class PathTrace {
 public:
  /* Render scheduler is used to report timing information and access things like start/finish
   * sample. */
  PathTrace(Device *device,
            Device *denoise_device,
            Film *film,
            DeviceScene *device_scene,
            RenderScheduler &render_scheduler,
            TileManager &tile_manager);
  ~PathTrace();

  /* Create devices and load kernels which are created on-demand (for example, denoising devices).
   * The progress is reported to the currently configure progress object (via `set_progress`). */
  void load_kernels();
  bool supports_gpu_query() const;
  void release_query_workspace(uint64_t stream = 0);
  void query_slice(const QuerySlice &slice, QuerySliceResult &result);

  /* Allocate working memory. This runs after scene memory allocation, so that the number of
   * path states can take the updated scene into account. Scene memory can be moved back to
   * the host as part of this allocation, as work memory is not host mapped and allocating
   * it may request more space to be freed on the GPU. */
  void alloc_work_memory();

  /* Check whether now it is a good time to reset rendering.
   * Used to avoid very often resets in the viewport, giving it a chance to draw intermediate
   * render result. */
  bool ready_to_reset();

  void reset(const BufferParams &full_params,
             const BufferParams &big_tile_params,
             bool reset_rendering);

  void device_free();

  /* Set progress tracker.
   * Used to communicate details about the progress to the outer world, check whether rendering is
   * to be canceled.
   *
   * The path tracer writes to this object, and then at a convenient moment runs
   * progress_update_cb() callback. */
  void set_progress(Progress *progress);
  void reset_vcm(int paths, int rounds);

  /* NOTE: This is a blocking call. Meaning, it will not return until given number of samples are
   * rendered (or until rendering is requested to be canceled). */
  void render(const RenderWork &render_work);

  void set_photon_cache_native(bool enabled);
  void set_photon_estimator(int estimator, bool final_phase);
  void select_photon_lpe_program(bool photon);
  void set_photon_cache_context(const string &filename, const BufferParams &params);
  void render_photon_cache(const BufferParams &params, const string &native_filename);

  /* Future guiding proposals are frozen for this absolute sample range on every
   * replica. Completion is published only after all devices have reached its end.
   * Trees/distributions remain algorithm-owned; this does not average tree nodes. */
  enum class SamplingDomain { COUPLED_TILE, NATIVE_TILE, PHOTON_TILE, PHOTON_FULL_FRAME };
  struct SamplingEpoch {
    SamplingDomain domain;
    int start_sample;
    int num_samples;
    int devices;
    int4 region;
    uint64_t scene_revision;
  };
  std::function<void(const SamplingEpoch &)> sampling_epoch_begin_cb;
  std::function<void(const SamplingEpoch &)> sampling_epoch_complete_cb;

  /* TODO(sergey): Decide whether denoiser is really a part of path tracer. Currently it is
   * convenient to have it here because then its easy to access render buffer. But the downside is
   * that this adds too much of entities which can live separately with some clear API. */

  /* Set denoiser parameters.
   * Use this to configure the denoiser before rendering any samples. */
  void set_denoiser_params(const DenoiseParams &params);

  /* Set parameters used for adaptive sampling.
   * Use this to configure the adaptive sampler before rendering any samples. */
  void set_adaptive_sampling(const AdaptiveSampling &adaptive_sampling);

  /* Set the parameters for guiding.
   * Use to setup the guiding structures before each rendering iteration. */
  void set_guiding_params(const GuidingParams &params, const bool reset);

  /* Sets output driver for render buffer output. */
  void set_output_driver(unique_ptr<OutputDriver> driver);

  /* Set display driver for interactive render buffer display. */
  void set_display_driver(unique_ptr<DisplayDriver> driver);

  /* Set deep output driver for deep EXR output. */
  void set_deep_output_driver(unique_ptr<DeepOutputDriver> driver);

  /* Get deep output driver (for finalization after render). */
  DeepOutputDriver *get_deep_output_driver() const
  {
    return deep_output_driver_.get();
  }

  /* Clear the display buffer by filling it in with all zeroes. */
  void zero_display();

  /* Perform drawing of the current state of the DisplayDriver. */
  void draw();

  /* Flush outstanding display commands before ending the render loop. */
  void flush_display();

  /* Cancel rendering process as soon as possible, without waiting for full tile to be sampled.
   * Used in cases like reset of render session.
   *
   * This is a blocking call, which returns as soon as there is no running `render_samples()` call.
   */
  void cancel();

  /* Copy an entire render buffer to/from the path trace. */

  /* Copy happens via CPU side buffer: data will be copied from every device of the path trace, and
   * the data will be copied to the device of the given render buffers. */
  void copy_to_render_buffers(RenderBuffers *render_buffers);

  /* Copy happens via CPU side buffer: data will be copied from the device of the given render
   * buffers and will be copied to all devices of the path trace. */
  void copy_from_render_buffers(RenderBuffers *render_buffers);

  /* Copy render buffers of the big tile from the device to host.
   * Return true if all copies are successful. */
  bool copy_render_tile_from_device();

  /* Read given full-frame file from disk, perform needed processing and write it to the software
   * via the write callback. */
  void set_full_buffer_state(std::shared_ptr<FullBufferState> state);
  void capture_full_buffer_context(string_view filename);
  void process_full_buffer_from_disk(string_view filename);

  /* Get number of samples in the current big tile render buffers. */
  int get_num_render_tile_samples() const;
  uint64_t get_buffer_revision() const { return lpe_buffer_revision_; }
  bool is_full_resolution() const;
  void refresh_display();
  void tag_lpe_interaction();
  void request_lpe_display_update();
  bool set_lpe_navigating(bool navigating);
  std::chrono::milliseconds lpe_denoise_wait_time() const;

  /* Get pass data of the entire big tile.
   * This call puts pass render result from all devices into the final pixels storage.
   *
   * NOTE: Expects buffers to be copied to the host using `copy_render_tile_from_device()`.
   *
   * Returns false if any of the accessor's `get_render_tile_pixels()` returned false. */
  bool get_render_tile_pixels(const PassAccessor &pass_accessor,
                              const PassAccessor::Destination &destination,
                              bool effective = false);
  bool read_photon_preview_pass(const BufferPass &pass, int channels, float *pixels);
  int2 get_display_tile_size() const;

  /* Set pass data for baking. */
  bool set_render_tile_pixels(PassAccessor &pass_accessor, const PassAccessor::Source &source);

  /* Check whether denoiser was run and denoised passes are available. */
  bool has_denoised_result() const;

  /* Get size and offset (relative to the buffer's full x/y) of the currently rendering tile.
   * In the case of tiled rendering this will return full-frame after all tiles has been rendered.
   *
   * NOTE: If the full-frame buffer processing is in progress, returns parameters of the full-frame
   * instead. */
  int2 get_render_tile_size() const;
  int2 get_render_tile_offset() const;
  int2 get_render_size() const;

  /* Get buffer parameters of the current tile.
   *
   * NOTE: If the full-frame buffer processing is in progress, returns parameters of the full-frame
   * instead. */
  const BufferParams &get_render_tile_params() const;

  /* Generate full multi-line report of the rendering process, including rendering parameters,
   * times, and so on. */
  string full_report() const;

  /* Callback which is called to report current rendering progress.
   *
   * It is supposed to be cheaper than buffer update/write, hence can be called more often.
   * Additionally, it might be called form the middle of wavefront (meaning, it is not guaranteed
   * that the buffer is "uniformly" sampled at the moment of this callback). */
  std::function<void(void)> progress_update_cb;

 protected:
  /* Actual implementation of the rendering pipeline.
   * Calls steps in order, checking for the cancel to be requested in between.
   *
   * Is separate from `render()` to simplify dealing with the early outputs and keeping
   * `render_cancel_` in the consistent state. */
  void render_pipeline(RenderWork render_work);

  bool render_vcm(uint first_round = 0, uint round_count = 0);
  bool write_vcm_tile(uint film_samples);
  uint vcm_paths_ = 0;
  uint vcm_rounds_ = 1;
  vector<float3> vcm_film_;
  vector<float3> vcm_adaptive_film_;
  /* A completed round's per-emitter Gaussian and effective-sample mixing probability. */
  vector<float4> vcm_point_targets_;
  vector<float> vcm_point_fractions_;
  string vcm_capture_directory_;
  bool vcm_capture_complete_ = false;

  /* Initialize kernel execution on all integrator queues. */
  void render_init_kernel_execution();

  /* Release kernel execution resources on all integrator queues. */
  void render_deinit_kernel_execution();

  /* Make sure both allocated and effective buffer parameters of path tracer works are up to date
   * with the current big tile parameters, performance-dependent slicing, and resolution divider.
   */
  void update_work_buffer_params_if_needed(const RenderWork &render_work);
  void update_allocated_work_buffer_params();
  void update_effective_work_buffer_params(const RenderWork &render_work);

  /* Perform various steps of the render work.
   *
   * Note that some steps might modify the work, forcing some steps to happen within this iteration
   * of rendering. */
  void init_render_buffers(const RenderWork &render_work);
  void path_trace(RenderWork &render_work);
  void configure_photon_replicas();
  void merge_photon_replica(PathTraceWork &work, int replica);
  void broadcast_photon_convergence();
  void apply_photon_cache(string_view filename, RenderBuffers &native);
  void adaptive_sample(RenderWork &render_work);
  void denoise(const RenderWork &render_work);
  void denoise_volume_guiding_buffers(const RenderWork &render_work, const bool has_volume);
  void cryptomatte_postprocess(const RenderWork &render_work);
  void update_display(const RenderWork &render_work);
  void rebalance(const RenderWork &render_work);
  void write_tile_buffer(const RenderWork &render_work);
  void finalize_full_buffer_on_disk(const RenderWork &render_work);
  void sync_deep_output_buffers();
  void get_deep_tile_pixels(vector<float> &beauty_pixels,
                            bool &has_beauty_pixels,
                            vector<float> &sample_count_pixels,
                            bool &has_sample_count_pixels);

  /* Updates/initializes the guiding structures after a rendering iteration.
   * The structures are updated using the training data/samples generated during the previous
   * rendering iteration */
  void guiding_update_structures();

  /* Prepares the per-kernel thread related guiding structures (e.g., PathSegmentStorage,
   * pointers to the global Field and SegmentStorage)*/
  void guiding_prepare_structures();

  /* Get number of samples in the current state of the render buffers. */
  int get_num_samples_in_buffer();

  /* Check whether user requested to cancel rendering, so that path tracing is to be finished as
   * soon as possible. */
  bool is_cancel_requested();

  /* Write the big tile render buffer via the write callback. */
  void tile_buffer_write();

  /* Read the big tile render buffer via the read callback. */
  void tile_buffer_read();

  /* Write current tile into the file on disk. */
  void tile_buffer_write_to_disk();

  /* Run the progress_update_cb callback if it is needed. */
  void progress_update_if_needed(const RenderWork &render_work);

  void progress_set_status(const string &status, const string &substatus = "");

  /* Destroy GPU resources (such as graphics interop) used by work. */
  void destroy_gpu_resources();

  /* Pointer to a device which is configured to be used for path tracing. If multiple devices
   * are configured this is a `MultiDevice`. */
  Device *device_ = nullptr;

  /* Pointer to a device which is configured to be used for denoising. Can be identical
   * to the device */
  Device *denoise_device_ = nullptr;

  /* CPU device for creating temporary render buffers on the CPU side. */
  unique_ptr<Device> cpu_device_;

  Film *film_;
  DeviceScene *device_scene_;

  RenderScheduler &render_scheduler_;
  TileManager &tile_manager_;

  /* Display driver for interactive render buffer display. */
  unique_ptr<PathTraceDisplay> display_;

  /* Output driver to write render buffer to. */
  unique_ptr<OutputDriver> output_driver_;

  /* Deep output driver for deep EXR output. */
  unique_ptr<DeepOutputDriver> deep_output_driver_;

  /* Per-compute device descriptors of work which is responsible for path tracing on its configured
   * device. */
  vector<unique_ptr<PathTraceWork>> path_trace_works_;

  /* Full-tile replica films are read back once per scheduler batch. This work owns
   * only the aggregate film, never integrator SoA or a shared photon map. */
  unique_ptr<PathTraceWork> photon_reduce_work_;
  bool photon_replicas_ = false;
  bool photon_test_replicas_ = false;
  thread_mutex photon_reduce_mutex_;
  bool photon_cache_native_ = false;
  int photon_estimator_ = -1;
  bool photon_final_phase_ = true;
  void update_photon_cache_display(const RenderWork &work, bool force = false);
  bool get_photon_preview_pixels(const RenderBuffers *current,
                                 const BufferParams &region,
                                 bool photon_phase,
                                 const string &name,
                                 vector<float> &rgba);
  void write_photon_preview_pixels(const BufferParams &region,
                                   const vector<float> &rgba,
                                   half4 *mapped);
  void clear_photon_preview_tiles();
  void store_photon_preview_tile(const RenderBuffers &buffer);
  struct PhotonPreviewNativeTile {
    BufferParams params;
    fpos_t position;
  };
  vector<PhotonPreviewNativeTile> photon_preview_tiles_;
  FILE *photon_preview_tiles_file_ = nullptr;
  string photon_preview_tiles_filename_;
  bool photon_preview_tiles_failed_ = false;
  bool load_photon_preview_pass(string_view name, int channels, vector<float> &pixels);
  string photon_preview_filename_;
  string photon_preview_pass_;
  BufferParams photon_preview_params_;
  map<string, vector<float>> photon_preview_native_passes_;
  vector<float> photon_preview_test_pixels_;
  bool photon_preview_display_initialized_ = false;
  /* Shared with the host so earlier layers survive Session recreation. */
  std::shared_ptr<FullBufferState> full_buffer_state_;

  /* Per-path trace work information needed for multi-device balancing. */
  vector<WorkBalanceInfo> work_balance_infos_;

  /* Render buffer parameters of the full frame and current big tile. */
  BufferParams full_params_;
  BufferParams big_tile_params_;

  /* Denoiser which takes care of denoising the big tile. */
  unique_ptr<Denoiser> denoiser_;
  /* Post-LPE display denoising owns no path-tracing accumulation buffers. */
  DenoiseParams lpe_denoise_params_;
  unique_ptr<Denoiser> lpe_denoiser_;
  std::shared_ptr<const void> lpe_denoised_plan_;
  uint64_t lpe_denoised_revision_ = 0;
  vector<float> lpe_denoised_rgba_;
  std::atomic<double> lpe_last_interaction_time_{0.0};
  std::atomic<bool> lpe_navigating_{false};
  bool lpe_denoise_pending_ = false;  // Session thread only.
  bool update_lpe_denoised_preview(const RenderWork &render_work);

  /* Denoiser device descriptor which holds the denoised big tile for multi-device workloads. */
  unique_ptr<PathTraceWork> big_tile_denoise_work_;

#if defined(WITH_PATH_GUIDING)
  /* Guiding related attributes */
  GuidingParams guiding_params_;

  /* The guiding field which holds the representation of the incident radiance field for the
   * complete scene. */
  unique_ptr<openpgl::cpp::Field> guiding_field_;

  /* The storage container which holds the training data/samples generated during the last
   * rendering iteration. */
  unique_ptr<openpgl::cpp::SampleStorage> guiding_sample_data_storage_;

  /* The number of already performed training iterations for the guiding field. */
  int guiding_update_count = 0;
#endif

  /* State which is common for all the steps of the render work.
   * Is brought up to date in the `render()` call and is accessed from all the steps involved into
   * rendering the work. */
  struct {
    /* Denotes whether render buffers parameters of path trace works are to be reset for the new
     * value of the big tile parameters. */
    bool need_reset_params = false;

    /* Divider of the resolution for faster previews.
     *
     * Allows to re-use same render buffer, but have less pixels rendered into in it. The way to
     * think of render buffer in this case is as an over-allocated array: the resolution divider
     * affects both resolution and stride as visible by the integrator kernels. */
    float resolution_divider = 0;

    /* Parameters of the big tile with the current resolution divider applied. */
    BufferParams effective_big_tile_params;
    BufferParams effective_denoised_big_tile_params;

    /* Denoiser was run and there are denoised versions of the passes in the render buffers. */
    bool has_denoised_result = false;

    /* Current tile has been written (to either disk or callback.
     * Indicates that no more work will be done on this tile. */
    bool tile_written = false;
  } render_state_;

  /* Progress object which is used to communicate sample progress. */
  Progress *progress_;

  /* Fields required for canceling render on demand, as quickly as possible. */
  struct {
    /* Indicates whether there is an on-going `render_samples()` call. */
    bool is_rendering = false;

    /* Indicates whether rendering is requested to be canceled by `cancel()`. */
    bool is_requested = false;

    /* Synchronization between thread which does `render_samples()` and thread which does
     * `cancel()`. */
    thread_mutex mutex;
    thread_condition_variable condition;
  } render_cancel_;

  /* Indicates whether a render result was drawn after latest session reset.
   * Used by `ready_to_reset()` to implement logic which feels the most interactive. */
  bool did_draw_after_reset_ = true;
  /* Changes on reset and completed batches, not on display-only refreshes. */
  uint64_t lpe_buffer_revision_ = 1;
  /* UI publication and render-thread consumption of a display-only edit. */
  std::atomic<uint64_t> lpe_display_revision_{0};
  uint64_t lpe_displayed_revision_ = 0;
  struct LPEBatchPreview;
  std::shared_ptr<LPEBatchPreview> lpe_batch_preview_;
  unique_ptr<DedicatedTaskPool> lpe_sampling_worker_;
  thread_mutex lpe_batch_mutex_;
  thread_condition_variable lpe_batch_condition_;
  void capture_lpe_batch_preview();
  void trace_with_lpe_preview(RenderWork &work);
  bool display_lpe_batch_preview(uint64_t revision);
  /* Actual completed samples, unlike the scheduler count which includes the
   * next batch before render(). Display-only refreshes must not use that count. */
  int completed_buffer_samples_ = 0;

  /* State of the full frame processing and writing to the software. */
  struct {
    RenderBuffers *render_buffers = nullptr;
  } full_frame_state_;
};

CCL_NAMESPACE_END
