/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#pragma once

#include "kernel/integrator/state.h"

#include "device/graphics_interop.h"
#include "device/memory.h"
#include "device/queue.h"

#include "integrator/path_trace_work.h"
#include "integrator/work_tile_scheduler.h"

#include "util/vector.h"

#include "util/hash.h"

CCL_NAMESPACE_BEGIN

/* Choose one round from each pair. Alternate assignment independently of the
 * device round-robin and wavelength strata. Both estimators share this split. */
inline bool vcm_adaptive_half(uint round, uint seed)
{
  return ((round ^ hash_uint2(round >> 1, seed)) & 1u) == 0;
}

struct KernelWorkTile;
struct QuerySlice;
struct QuerySliceResult;

/* Implementation of PathTraceWork which schedules work to the device in tiles which are sized
 * to match device queue's number of path states.
 * This implementation suits best devices which have a lot of integrator states, such as GPU. */
class PathTraceWorkGPU : public PathTraceWork {
 public:
  PathTraceWorkGPU(Device *device,
                   Film *film,
                   DeviceScene *device_scene,
                   const bool *cancel_requested_flag);

  ~PathTraceWorkGPU() override;
  void alloc_work_memory() override;
  uint vcm_batch_size(uint requested) const override;
  bool vcm_emit_batch(const VCMBatch &batch,
                      uint count,
                      VCMLightPath *paths,
                      VCMVertex *vertices) override;
  void reset_vcm_guides()
  {
    vcm_guide_targets_.clear();
    vcm_guide_fractions_.clear();
  }
  /* One physical GPU owns complete global rounds, including its VM photon table. */
  bool vcm_render_rounds(uint first, uint end, uint stride, uint count, uint depth,
                         bool area, bool native_bsdf, int planes, const vector<int> &routes, vector<float> &film);

  void release_query_workspace(uint64_t stream);
  bool supports_gpu_query() const;
  void query_slice(const QuerySlice &slice, QuerySliceResult &result);
  void init_execution() override;

  void render_samples(RenderStatistics &statistics,
                      const int start_sample,
                      const int samples_num,
                      const int sample_offset) override;

  void copy_to_display(PathTraceDisplay *display,
                       PassMode pass_mode,
                       const int num_samples) override;
  void destroy_gpu_resources(PathTraceDisplay *display) override;

  bool read_preview_pass(const PassAccessor::PassAccessInfo &info,
                         float exposure, int samples, int channels, float *pixels) override;
  bool copy_render_buffers_from_device() override;
  bool copy_render_buffers_to_device() override;
  bool zero_render_buffers() override;
  bool copy_convergence_masks(const RenderBuffers &source) override;

  void merge_photon_estimates() override;
  int adaptive_sampling_converge_filter_count_active_impl(const float threshold,
                                                          bool reset) override;
  void cryptomatte_postproces() override;
  void denoise_volume_guiding_buffers() override;

 protected:
  vector<float4> vcm_guide_targets_;
  vector<float> vcm_guide_fractions_;

  struct QueryWorkspace;
  unique_ptr<QueryWorkspace> query_workspace_;
  int diagnostic_slots_ = 0;
  const IntegratorStateGPU *query_viewport_guide_ = nullptr;
  void alloc_integrator_soa();
  void alloc_integrator_queue();
  void alloc_integrator_sorting();
  void alloc_integrator_path_split();
  void alloc_photon_caustics();
  void alloc_gpu_guiding();
  void render_samples_batch(RenderStatistics &statistics, int start_sample,
                            int samples_num, int sample_offset);
  void publish_gpu_guiding(int completed_sample);

  /* Returns DEVICE_KERNEL_NUM if there are no scheduled kernels. */
  DeviceKernel get_most_queued_kernel() const;

  void enqueue_reset();
  bool enqueue_photon_caustics(int start_sample);

  bool enqueue_work_tiles(bool &finished);
  void enqueue_work_tiles(DeviceKernel kernel,
                          const KernelWorkTile work_tiles[],
                          const int num_work_tiles,
                          const int num_active_paths,
                          const int num_predicted_splits);

  bool enqueue_path_iteration();
  void enqueue_path_iteration(DeviceKernel kernel, const int num_paths_limit = INT_MAX);

  bool update_queue_counter_and_cache();

  void compute_queued_paths(DeviceKernel kernel, DeviceKernel queued_kernel);
  void compute_sorted_queued_paths(DeviceKernel queued_kernel, const int num_paths_limit);

  void compact_main_paths(const int num_active_paths);
  void compact_shadow_paths(const bool force = false);
  void compact_paths(const int num_active_paths,
                     const int max_active_path_index,
                     DeviceKernel terminated_paths_kernel,
                     DeviceKernel compact_paths_kernel,
                     DeviceKernel compact_kernel);

  int num_active_main_paths_paths();

  /* Check whether graphics interop can be used for the PathTraceDisplay update. */
  bool should_use_graphics_interop(PathTraceDisplay *display);

  /* Naive implementation of the `copy_to_display()` which performs film conversion on the
   * device, then copies pixels to the host and pushes them to the `display`. */
  void copy_to_display_naive(PathTraceDisplay *display, PassMode pass_mode, const int num_samples);

  /* Implementation of `copy_to_display()` which uses driver's OpenGL/GPU interoperability
   * functionality, avoiding copy of pixels to the host. */
  bool copy_to_display_interop(PathTraceDisplay *display,
                               PassMode pass_mode,
                               const int num_samples);

  /* Synchronously run film conversion kernel and store display result in the given destination. */
  void get_render_tile_film_pixels(const PassAccessor::Destination &destination,
                                   PassMode pass_mode,
                                   int num_samples);

  int adaptive_sampling_convergence_check_count_active(const float threshold, bool reset);
  void enqueue_adaptive_sampling_filter_x();
  void enqueue_adaptive_sampling_filter_y();

  bool has_shadow_catcher() const;

  /* Count how many currently scheduled paths can still split. */
  int shadow_catcher_count_possible_splits();

  /* Kernel properties. */
  bool kernel_uses_sorting(DeviceKernel kernel);
  bool kernel_creates_shadow_paths(DeviceKernel kernel);
  bool kernel_creates_ao_paths(DeviceKernel kernel);
  bool kernel_is_shadow_path(DeviceKernel kernel);
  int kernel_max_active_main_path_index(DeviceKernel kernel);

  /* Integrator queue. */
  unique_ptr<DeviceQueue> queue_;
  device_vector<float> preview_pass_;
  unique_ptr<RenderBuffers> convergence_buffer_;

  /* Scheduler which gives work to path tracing threads. */
  WorkTileScheduler work_tile_scheduler_;

  /* Integrate state for paths. */
  IntegratorStateGPU integrator_state_gpu_;
  /* SoA arrays for integrator state. */
  vector<unique_ptr<device_memory>> integrator_state_soa_;
  uint64_t integrator_state_soa_kernel_features_;
  bool integrator_state_soa_use_shadow_color_ = false;
  int integrator_state_soa_volume_stack_size_ = 0;
  /* Keep track of number of queued kernels. */
  device_vector<IntegratorQueueCounter> integrator_queue_counter_;
  /* Shader sorting. */
  device_vector<int> integrator_shader_sort_counter_;
  device_vector<int> integrator_shader_raytrace_sort_counter_;
  device_vector<int> integrator_shader_sort_prefix_sum_;
  device_vector<int> integrator_shader_sort_partition_key_offsets_;
  /* Path split. */
  device_vector<int> integrator_next_main_path_index_;
  device_vector<int> integrator_next_shadow_path_index_;

  /* Camera-path guiding, independent of photon emission guiding. */
  device_only_memory<float> gpu_guiding_cdf_;
  device_only_memory<float> gpu_guiding_statistics_;
  device_vector<uint> gpu_guiding_ready_cells_;
  bool gpu_guiding_reset_ = true;
  bool gpu_guiding_frozen_ = false;
  int gpu_guiding_next_publish_ = 32;
  double gpu_guiding_update_seconds_ = 0.0;

  /* Caustics prepass buffers, shared with camera shading for read-only gathers. */
  device_only_memory<KernelPhotonCaustic> photon_caustics_;
  device_only_memory<uint> photon_caustics_hash_;
  device_only_memory<uint> photon_caustics_query_hash_;
  device_only_memory<uint> photon_caustics_query_next_;
  device_vector<uint> photon_caustics_count_;
  device_only_memory<uint> photon_caustics_stored_indices_;
  device_only_memory<float4> photon_caustics_pilot_footprints_;
  device_only_memory<uint2> photon_cell_statistics_;
  device_only_memory<KernelPhotonReconstructionMetadata> photon_reconstruction_metadata_;
  device_only_memory<KernelPhotonBeamMetadata> photon_beam_metadata_;
  device_only_memory<KernelPhotonCausticsEvent> photon_caustics_events_;
  device_only_memory<uint> photon_caustics_path_indices_;
  device_vector<KernelPhotonGuide> photon_guide_;
  device_vector<KernelPhotonGuideStatistics> photon_guide_statistics_;
  device_only_memory<float4> photon_guide_hits_;
  device_vector<float> photon_area_guide_cdf_;
  device_vector<float4> photon_area_guide_statistics_;
  device_only_memory<float2> photon_area_guide_round_statistics_;
  uint photon_area_guide_epoch_ = 0;
  bool photon_guide_reset_ = true;
  uint photon_guide_epoch_ = 0;
  bool photon_caustics_valid_ = false;
  uint64_t photon_caustics_revision_ = 0;
  int photon_caustics_seed_ = 0;
  int photon_caustics_requested_count_ = 0;
  uint photon_caustics_round_ = 0;
  int photon_caustics_update_interval_ = 0;
  uint photon_caustics_capacity_ = 0;
  float photon_caustics_radius_ = 0.0f;
  uint photon_caustics_event_stride_ = 0;
  uint64_t photon_caustics_build_count_ = 0;
  uint64_t photon_caustics_reuse_count_ = 0;
  double photon_caustics_build_seconds_ = 0.0;

  /* Temporary buffer to get an array of queued path for a particular kernel. */
  device_vector<int> queued_paths_;
  device_vector<int> num_queued_paths_;

  /* Temporary buffer for passing work tiles to kernel. */
  device_vector<KernelWorkTile> work_tiles_;

  /* Temporary buffer used by the copy_to_display() whenever graphics interoperability is not
   * available. Is allocated on-demand. */
  device_vector<half4> display_rgba_half_;

  unique_ptr<DeviceGraphicsInterop> device_graphics_interop_;

  /* Cached result of device->should_use_graphics_interop(). */
  bool interop_use_checked_ = false;
  bool interop_use_ = false;

  /* Number of partitions to sort state indices into prior to material sort. */
  int num_sort_partitions_;

  /* Maximum number of concurrent integrator states. */
  int max_num_paths_;

  /* Minimum number of paths which keeps the device bust. If the actual number of paths falls below
   * this value more work will be scheduled. */
  int min_num_active_main_paths_;

  /* Maximum path index, effective number of paths used may be smaller than
   * the size of the integrator_state_ buffer so can avoid iterating over the
   * full buffer. */
  int max_active_main_path_index_;
};

CCL_NAMESPACE_END
