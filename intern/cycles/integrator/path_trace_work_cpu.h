/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#pragma once

#include "kernel/device/cpu/globals.h"
#include "kernel/integrator/state.h"
#include "kernel/sample/photon_probe.h"

#include "device/queue.h"

#include "integrator/path_trace_work.h"

#include "util/vector.h"

CCL_NAMESPACE_BEGIN

struct KernelWorkTile;
struct ThreadKernelGlobalsCPU;
struct IntegratorStateCPU;

class CPUKernels;

/* Implementation of PathTraceWork which schedules work on to queues pixel-by-pixel,
 * for CPU devices.
 *
 * NOTE: For the CPU rendering there are assumptions about TBB arena size and number of concurrent
 * queues on the render device which makes this work be only usable on CPU. */
class PathTraceWorkCPU : public PathTraceWork {
 public:
  PathTraceWorkCPU(Device *device,
                   Film *film,
                   DeviceScene *device_scene,
                   const bool *cancel_requested_flag);

  bool vcm_emit_batch(const VCMBatch &batch,
                      uint count,
                      VCMLightPath *paths,
                      VCMVertex *vertices) override;
  bool vcm_connect_batch(const VCMBatch &batch, uint count, const VCMLightPath *paths,
                         const VCMVertex *vertices, const struct VCMMergeRound *merging,
                         struct VCMConnectionResult *results);
  void init_execution() override;
  void deinit_execution() override;

  void render_samples(RenderStatistics &statistics,
                      const int start_sample,
                      const int samples_num,
                      const int sample_offset) override;

  void copy_to_display(PathTraceDisplay *display,
                       PassMode pass_mode,
                       const int num_samples) override;
  void destroy_gpu_resources(PathTraceDisplay *display) override;

  bool copy_render_buffers_from_device() override;
  bool copy_render_buffers_to_device() override;
  bool zero_render_buffers() override;

  void merge_photon_estimates() override;
  int adaptive_sampling_converge_filter_count_active_impl(const float threshold,
                                                          bool reset) override;
  void cryptomatte_postproces() override;
  void denoise_volume_guiding_buffers() override;

#if defined(WITH_PATH_GUIDING)
  /* Initializes the per-thread guiding kernel data. The function sets the pointers to the
   * global guiding field and the sample data storage as well es initializes the per-thread
   * guided sampling distributions (e.g., SurfaceSamplingDistribution and
   * VolumeSamplingDistribution). */
  void guiding_init_kernel_globals(void *guiding_field,
                                   void *sample_data_storage,
                                   const bool train) override;

  /* Pushes the collected training data/samples of a path to the global sample storage.
   * This function is called at the end of a random walk/path generation. */
  void guiding_push_sample_data_to_global_storage(ThreadKernelGlobalsCPU *kg,
                                                  IntegratorStateCPU *state,
                                                  ccl_global float *ccl_restrict render_buffer);
#endif

 protected:
  /* Reuse the current emission round, or build it before camera paths in the caller's TBB arena.
   */
  bool build_photon_caustics(int start_sample);

  /* Core path tracing routine. Renders given work time on the given queue. */
  void render_samples_full_pipeline(ThreadKernelGlobalsCPU *kernel_globals,
                                    const KernelWorkTile &work_tile,
                                    const int samples_num);

  /* CPU kernels. */
  const CPUKernels &kernels_;

  /* Pointer to device-owned kernel globals which is suitable for concurrent access from multiple
   * threads. This allows dynamic updates to image_info when textures are loaded on demand. */
  vector<ThreadKernelGlobalsCPU> *kernel_thread_globals_ = nullptr;

  vector<KernelPhotonCaustic> photon_caustics_;
  vector<uint> photon_caustics_hash_;
  vector<uint> photon_caustics_query_hash_;
  vector<uint> photon_caustics_query_next_;
  vector<uint> photon_caustics_stored_indices_;
  vector<float4> photon_caustics_pilot_footprints_;
  vector<uint2> photon_cell_statistics_;
  vector<KernelPhotonReconstructionMetadata> photon_reconstruction_metadata_;
  vector<KernelPhotonBeamMetadata> photon_beam_metadata_;
  vector<KernelPhotonCausticsEvent> photon_caustics_events_;
  vector<uint> photon_caustics_path_indices_;
  vector<KernelPhotonGuide> photon_guide_;
  vector<KernelPhotonGuideStatistics> photon_guide_statistics_;
  vector<float4> photon_guide_hits_;
  vector<float> photon_area_guide_cdf_;
  vector<float4> photon_area_guide_statistics_;
  vector<float2> photon_area_guide_round_statistics_;
  uint photon_area_guide_epoch_ = 0;
  bool photon_guide_reset_ = true;
  uint photon_guide_epoch_ = 0;
  float photon_guide_fraction_ = 0.0f;
  uint photon_caustics_count_[PHOTON_MAP_COUNTERS] = {};
  uint photon_caustics_cache_miss_ = 0;
  KernelPhotonCausticsCache photon_caustics_cache_ = {};
  bool photon_caustics_valid_ = false;
  uint64_t photon_caustics_revision_ = 0;
  int photon_caustics_seed_ = 0;
  int photon_caustics_requested_count_ = 0;
  uint photon_caustics_round_ = 0;
  int photon_caustics_update_interval_ = 0;
  uint64_t photon_caustics_build_count_ = 0;
  uint64_t photon_caustics_reuse_count_ = 0;
  double photon_caustics_build_seconds_ = 0.0;
};

CCL_NAMESPACE_END
