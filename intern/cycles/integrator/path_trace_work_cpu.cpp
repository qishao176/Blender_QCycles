/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#include "integrator/path_trace_work_cpu.h"
#include "kernel/film/deep_physical.h"
#include "integrator/photon_guiding.h"
#include "integrator/photon_spectrum.h"
#include "integrator/vcm_merge_round.h"
#include "kernel/integrator/vcm_types.h"

#include <limits>
#include <cstdlib>
#include <new>

#include "device/cpu/kernel.h"
#include "kernel/device/cpu/vcm_connect.h"
#include "device/device.h"

#ifdef WITH_CYCLES_DEBUG
#  include "kernel/film/write.h"
#endif

#include "kernel/integrator/path_state.h"
#include "kernel/sample/photon_spectrum.h"

#include "integrator/pass_accessor_cpu.h"
#include "integrator/path_trace_display.h"

#include "scene/scene.h"
#include "session/buffers.h"

#include "util/algorithm.h"
#include "util/hash.h"
#include "util/log.h"
#include "util/lpe_query.h"
#include "util/tbb.h"
#include "util/time.h"

CCL_NAMESPACE_BEGIN

/* Create TBB arena for execution of path tracing and rendering tasks. */
static inline tbb::task_arena local_tbb_arena_create(const Device *device)
{
  /* TODO: limit this to number of threads of CPU device, it may be smaller than
   * the system number of threads when we reduce the number of CPU threads in
   * CPU + GPU rendering to dedicate some cores to handling the GPU device. */
  return tbb::task_arena(device->info.cpu_threads);
}

/* Get ThreadKernelGlobalsCPU for the current thread. */
static inline ThreadKernelGlobalsCPU *kernel_thread_globals_get(
    vector<ThreadKernelGlobalsCPU> &kernel_thread_globals)
{
  const int thread_index = tbb::this_task_arena::current_thread_index();
  DCHECK_GE(thread_index, 0);
  DCHECK_LE(thread_index, kernel_thread_globals.size());

  return &kernel_thread_globals[thread_index];
}

PathTraceWorkCPU::PathTraceWorkCPU(Device *device,
                                   Film *film,
                                   DeviceScene *device_scene,
                                   const bool *cancel_requested_flag)
    : PathTraceWork(device, film, device_scene, cancel_requested_flag),
      kernels_(Device::get_cpu_kernels())
{
  DCHECK_EQ(device->info.type, DEVICE_CPU);
}

void PathTraceWorkCPU::init_execution()
{
  /* Acquire thread globals, updating all data pointers. */
  kernel_thread_globals_ = device_->acquire_cpu_kernel_thread_globals();
  if (photon_replica_ && device_scene_->data.integrator.use_photon_caustics) {
    /* Preserve the CPU-specific BVH and texture pointers. Each replica only changes
     * its sampling stream, consistently for camera paths and photon emission. */
    for (ThreadKernelGlobalsCPU &kg : *kernel_thread_globals_) {
      kg.data.integrator.seed = photon_sampling_seed();
    }
  }
}

void PathTraceWorkCPU::deinit_execution()
{
  device_->release_cpu_kernel_thread_globals();
  kernel_thread_globals_ = nullptr;
}

bool PathTraceWorkCPU::vcm_emit_batch(const VCMBatch &batch,
                                      const uint count,
                                      VCMLightPath *paths,
                                      VCMVertex *vertices)
{
  VCMLaunch launch = {batch, paths, vertices};
  tbb::task_arena arena = local_tbb_arena_create(device_);
  arena.execute([&] {
    parallel_for(0u, count, [&](uint i) {
      ThreadKernelGlobalsCPU *kg = kernel_thread_globals_get(*kernel_thread_globals_);
      IntegratorStateCPU state = {};
      kernels_.integrator_vcm_emit(kg, &state, &launch, i);
    });
  });
  return !is_cancel_requested() && !device_->have_error();
}

bool PathTraceWorkCPU::vcm_connect_batch(const VCMBatch &batch,
                                         const uint count,
                                         const VCMLightPath *paths,
                                         const VCMVertex *vertices,
                                         const VCMMergeRound *merging,
                                         VCMConnectionResult *results)
{
  tbb::task_arena arena = local_tbb_arena_create(device_);
  arena.execute([&] {
    parallel_for(0u, count, [&](uint i) {
      if (!is_cancel_requested()) {
        const VCMVertex *prefix = merging ? merging->prefix(paths[i].index) :
                                           vertices + size_t(i) * batch.max_depth;
        vcm_connect_cpu(kernel_thread_globals_get(*kernel_thread_globals_),
                        batch, paths[i], prefix, merging, results[i]);
      }
    });
  });
  return !is_cancel_requested() && !device_->have_error();
}

bool PathTraceWorkCPU::build_photon_caustics(const int start_sample)
{
  if (is_cancel_requested() || device_->have_error()) {
    return false;
  }

  const KernelIntegrator &integrator = device_scene_->data.integrator;
  if (!integrator.use_photon_caustics || integrator.photon_caustics_count <= 0) {
    for (ThreadKernelGlobalsCPU &kg : *kernel_thread_globals_) {
      kg.photon_caustics = {};
    }
    photon_caustics_valid_ = false;
    photon_caustics_cache_ = {};
    photon_caustics_.free_memory();
    photon_caustics_hash_.free_memory();
    photon_caustics_query_hash_.free_memory();
    photon_caustics_query_next_.free_memory();
    photon_caustics_stored_indices_.free_memory();
    photon_caustics_pilot_footprints_.free_memory();
    photon_reconstruction_metadata_.free_memory();
    photon_cell_statistics_.free_memory();
    photon_beam_metadata_.free_memory();
    photon_caustics_events_.free_memory();
    photon_caustics_path_indices_.free_memory();
    photon_guide_.free_memory();
    photon_guide_statistics_.free_memory();
    photon_guide_hits_.free_memory();
    photon_fields_.free_memory();
    photon_field_observations_.free_memory();
    photon_field_vertices_.free_memory();
    photon_area_guide_cdf_.free_memory();
    photon_area_guide_statistics_.free_memory();
    photon_area_guide_round_statistics_.free_memory();
    photon_guide_reset_ = true;
    return true;
  }

  const uint capacity = uint(clamp(integrator.photon_caustics_count, 1024, 4 * 1024 * 1024));
  const bool query_index = integrator.use_photon_query_index;
  const float radius = max(integrator.photon_caustics_radius, 1.0e-6f);
  const int update_interval = max(integrator.photon_caustics_update_interval, 1);
  const uint round = uint(start_sample / update_interval);
  const bool path_filter = integrator.photon_reconstruction_filter &
                           PHOTON_RECONSTRUCTION_FILTER_PATH;
  const uint event_stride = (lpe_query_active() || device_scene_->data.film.lpe_state_count ||
                             device_scene_->data.film.lpe_photon_count || path_filter) ?
                                uint(max(integrator.max_bounce, 1) +
                                     integrator.transparent_max_bounce) :
                                0;
  if (photon_caustics_valid_ && !photon_guide_reset_ &&
      photon_caustics_revision_ == device_scene_->photon_caustics_revision &&
      photon_caustics_seed_ == photon_sampling_seed() &&
      photon_caustics_requested_count_ == integrator.photon_caustics_count &&
      photon_caustics_round_ == round && photon_caustics_update_interval_ == update_interval &&
      photon_caustics_cache_.capacity == capacity && photon_caustics_cache_.radius == radius &&
      bool(photon_caustics_cache_.query_hash) == query_index &&
      photon_caustics_cache_.event_stride == event_stride)
  {
    /* Thread globals are recreated for every batch in PathTrace::render. A batch may also split
     * an emission interval, so rebind this round without emitting it a second time. */
    for (ThreadKernelGlobalsCPU &kg : *kernel_thread_globals_) {
      kg.photon_caustics = photon_caustics_cache_;
    }
    ++photon_caustics_reuse_count_;
    if ((photon_caustics_reuse_count_ & (photon_caustics_reuse_count_ - 1)) == 0) {
      LOG_DEBUG << "CPU photon caustics cache: builds " << photon_caustics_build_count_
                << ", reused batches " << photon_caustics_reuse_count_ << ", build seconds "
                << photon_caustics_build_seconds_;
    }
    return true;
  }

  const bool guide_restart = photon_guide_reset_ ||
                             photon_caustics_revision_ !=
                                 device_scene_->photon_caustics_revision ||
                             photon_caustics_seed_ != photon_sampling_seed() ||
                             round < photon_caustics_round_;
  photon_caustics_valid_ = false;
  const scoped_timer build_timer;
  /* There are no camera tasks in flight while replacing the shared backing storage. */
  for (ThreadKernelGlobalsCPU &kg : *kernel_thread_globals_) {
    kg.photon_caustics = {};
  }
  uint hash_size = 1;
  while (hash_size < 2u * capacity) {
    hash_size <<= 1;
  }
  /* A complete row is reserved per emission attempt, including paths which do not deposit.
   * Never truncate the sequence: doing so would silently misclassify LPE contributions. */
  const size_t path_index_bytes = event_stride ? size_t(capacity) * sizeof(uint) : 0;
  const size_t max_events = (std::numeric_limits<size_t>::max() - path_index_bytes) /
                            sizeof(KernelPhotonCausticsEvent);
  if (event_stride && size_t(capacity) > max_events / event_stride) {
    device_->set_error("Photon caustics LPE event allocation size overflow.");
    return false;
  }
  const size_t event_count = size_t(capacity) * event_stride;
  if (event_count > photon_caustics_events_.max_size()) {
    device_->set_error("Photon caustics LPE event allocation exceeds the CPU buffer limit.");
    return false;
  }
  try {
    const size_t model_size = size_t(integrator.num_lights) * PHOTON_GUIDE_COMPONENTS;
    if (integrator.use_photon_field_guiding) {
      const size_t n = size_t(integrator.num_lights) * PHOTON_FIELD_COMPONENTS;
      const bool reset = guide_restart || photon_fields_.size() != n;
      photon_fields_.resize(n);
      photon_field_observations_.resize(size_t(integrator.num_lights) * PHOTON_FIELD_OBSERVATIONS);
      photon_field_vertices_.resize((capacity + 15) / 16);
      if (reset) {
        std::fill(photon_fields_.begin(), photon_fields_.end(), KernelPhotonField{});
      }
      else if (round / 32 != photon_guide_epoch_) {
        const int fitted = photon_field_guiding_fit(photon_fields_.data(),
                                                    photon_field_observations_.data(),
                                                    integrator.num_lights,
                                                    radius,
                                                    integrator.photon_caustics_target.w);
        LOG_DEBUG << "CPU photon field guiding: fitted " << fitted << " local proposals";
      }
      if (reset || round / 32 != photon_guide_epoch_) {
        std::fill(photon_field_observations_.begin(),
                  photon_field_observations_.end(),
                  KernelPhotonFieldObservation{});
      }
    }
    else {
      photon_fields_.free_memory();
      photon_field_observations_.free_memory();
      photon_field_vertices_.free_memory();
    }
    if (integrator.use_photon_guiding) {
      const bool reset = guide_restart || photon_guide_.size() != model_size;
      photon_guide_.resize(model_size);
      photon_guide_statistics_.resize(model_size);

      if (reset) {
        photon_guiding_initialize(photon_guide_.data(), integrator.num_lights);
        std::fill(photon_guide_statistics_.begin(),
                  photon_guide_statistics_.end(),
                  KernelPhotonGuideStatistics{});
        photon_guide_fraction_ = 0.0f;
        photon_guide_epoch_ = round / 32;
      }
      else if (round / 32 != photon_guide_epoch_) {
        if (photon_guiding_update(
                photon_guide_.data(), photon_guide_statistics_.data(), integrator.num_lights))
        {
          photon_guide_fraction_ = .5f;
          LOG_DEBUG << "CPU photon guiding: fitted epoch " << round / 32;
        }
        photon_guide_epoch_ = round / 32;
      }
    }
    else {
      photon_guide_.free_memory();
      photon_guide_statistics_.free_memory();
      photon_guide_fraction_ = 0.0f;
    }
    if (integrator.use_photon_area_guiding) {
      const size_t size = size_t(integrator.num_lights) * PHOTON_AREA_GUIDE_CELLS;
      const bool reset = guide_restart || photon_area_guide_cdf_.size() != size;
      photon_area_guide_cdf_.resize(size);
      photon_area_guide_statistics_.resize(size);
      photon_area_guide_round_statistics_.resize(size);
      if (reset) {
        photon_area_guiding_initialize(photon_area_guide_cdf_.data(), integrator.num_lights);
        std::fill(photon_area_guide_statistics_.begin(),
                  photon_area_guide_statistics_.end(),
                  zero_float4());
        photon_area_guide_epoch_ = round / 32;
      }
      else if (round / 32 != photon_area_guide_epoch_) {
        if (photon_area_guiding_update(photon_area_guide_cdf_.data(),
                                       photon_area_guide_statistics_.data(),
                                       integrator.num_lights))
        {
          LOG_DEBUG << "CPU photon area guiding: fitted epoch " << round / 32;
        }
        photon_area_guide_epoch_ = round / 32;
      }
    }
    else {
      photon_area_guide_cdf_.free_memory();
      photon_area_guide_statistics_.free_memory();
      photon_area_guide_round_statistics_.free_memory();
    }
    if (integrator.use_photon_guiding || integrator.use_photon_area_guiding) {
      photon_guide_hits_.resize(capacity);
    }
    else {
      photon_guide_hits_.free_memory();
    }
    photon_guide_reset_ = false;
    photon_caustics_.resize(capacity);
    if (integrator.use_photon_cell_statistics) {
      photon_cell_statistics_.resize(hash_size + 1);
    }
    else {
      photon_cell_statistics_.free_memory();
    }
    photon_caustics_hash_.resize(hash_size);
    if (query_index) {
      photon_caustics_query_hash_.resize(hash_size);
      photon_caustics_query_next_.resize(capacity);
    }
    else {
      photon_caustics_query_hash_.free_memory();
      photon_caustics_query_next_.free_memory();
    }
    if (integrator.use_photon_adaptive_radius &&
        (integrator.photon_reconstruction_method == PHOTON_RECONSTRUCTION_ELLIPSE_COVERAGE ||
         (integrator.photon_reconstruction_method != PHOTON_RECONSTRUCTION_REGIONAL &&
          integrator.use_photon_gradient_guard && integrator.photon_gradient_strength > 0.0f &&
          integrator.photon_caustics_differentials)))
    {
      photon_caustics_pilot_footprints_.resize(capacity);
    }
    else {
      photon_caustics_pilot_footprints_.free_memory();
    }
    if (integrator.use_photon_adaptive_radius) {
      photon_caustics_stored_indices_.resize(capacity);
    }
    else {
      photon_caustics_stored_indices_.free_memory();
    }
    if (integrator.photon_reconstruction_filter != PHOTON_RECONSTRUCTION_FILTER_LEGACY) {
      photon_reconstruction_metadata_.resize(capacity);
    }
    else {
      photon_reconstruction_metadata_.free_memory();
    }
    if (integrator.use_photon_beam_continuity) {
      photon_beam_metadata_.resize(capacity);
    }
    else {
      photon_beam_metadata_.free_memory();
    }
    if (event_stride) {
      photon_caustics_events_.resize(event_count);
      photon_caustics_path_indices_.resize(capacity);
    }
    else {
      photon_caustics_events_.free_memory();
      photon_caustics_path_indices_.free_memory();
    }
  }
  catch (const std::bad_alloc &) {
    device_->set_error(
        "Unable to allocate the photon caustics cache and complete LPE event "
        "history. Reduce Photon Count or Max Bounces.");
    return false;
  }

  KernelPhotonCausticsCache cache = {};
  cache.photons = photon_caustics_.data();
  cache.cell_statistics = photon_cell_statistics_.empty() ? nullptr :
                                                            photon_cell_statistics_.data();
  cache.hash = photon_caustics_hash_.data();
  cache.query_hash = photon_caustics_query_hash_.empty() ? nullptr :
                                                        photon_caustics_query_hash_.data();
  cache.query_next = photon_caustics_query_next_.empty() ? nullptr :
                                                        photon_caustics_query_next_.data();
  cache.count = photon_caustics_count_;
  cache.stored_indices = photon_caustics_stored_indices_.empty() ?
                             nullptr :
                             photon_caustics_stored_indices_.data();
  cache.cache_miss = &photon_caustics_cache_miss_;
  cache.pilot_footprints = photon_caustics_pilot_footprints_.empty() ?
                              nullptr :
                              photon_caustics_pilot_footprints_.data();
  cache.beam_metadata = photon_beam_metadata_.empty() ? nullptr : photon_beam_metadata_.data();
  cache.reconstruction_metadata = photon_reconstruction_metadata_.empty() ?
                                      nullptr :
                                      photon_reconstruction_metadata_.data();
  cache.events = event_stride ? photon_caustics_events_.data() : nullptr;
  cache.path_indices = event_stride ? photon_caustics_path_indices_.data() : nullptr;
  cache.event_stride = event_stride;
  cache.capacity = capacity;
  cache.hash_size = hash_size;
  /* The absolute sample range determines the round, independent of scheduler batch boundaries.
   * Power remains normalized per map; the camera film averages the independent map estimates. */
  cache.iteration = hash_uint2(round, photon_sampling_seed());
  cache.wavelength_random = getenv("CYCLES_PHOTON_WAVELENGTH_LEGACY") ?
                                hash_uint2_to_float(cache.iteration, 0xa54ff53au) :
                                photon_caustics_wavelength_random(round, photon_sampling_seed());
  cache.wavelength_groups = photon_wavelength_groups_from_environment();
  cache.radius = radius;
  cache.guide = photon_guide_.empty() ? nullptr : photon_guide_.data();
  cache.guide_statistics = photon_guide_statistics_.empty() ? nullptr :
                                                              photon_guide_statistics_.data();
  cache.guide_hits = photon_guide_hits_.empty() ? nullptr : photon_guide_hits_.data();
  cache.guide_fraction = photon_guide_fraction_;
  cache.fields = photon_fields_.empty() ? nullptr : photon_fields_.data();
  cache.field_observations = photon_field_observations_.empty() ?
                                 nullptr :
                                 photon_field_observations_.data();
  cache.field_vertices = photon_field_vertices_.empty() ? nullptr : photon_field_vertices_.data();
  cache.area_guide_cdf = photon_area_guide_cdf_.empty() ? nullptr : photon_area_guide_cdf_.data();
  cache.area_guide_statistics = photon_area_guide_statistics_.empty() ?
                                    nullptr :
                                    photon_area_guide_statistics_.data();
  cache.area_guide_round_statistics = photon_area_guide_round_statistics_.empty() ?
                                          nullptr :
                                          photon_area_guide_round_statistics_.data();
  for (ThreadKernelGlobalsCPU &kg : *kernel_thread_globals_) {
    kg.photon_caustics = cache;
  }

  /* CPU image lookups load missing tiles synchronously. Keep the shared cache-miss contract
   * nevertheless: a partially emitted map must never replace the corresponding camera paths. */
  const bool probe = begin_photon_probe(round, guide_restart);
  constexpr int max_attempts = 16;
  for (int attempt = 0; attempt < max_attempts; attempt++) {
    if (is_cancel_requested()) {
      return false;
    }
    std::fill(photon_caustics_hash_.begin(), photon_caustics_hash_.end(), 0u);
    std::fill(photon_caustics_query_hash_.begin(), photon_caustics_query_hash_.end(), 0u);
    std::fill(photon_area_guide_round_statistics_.begin(),
              photon_area_guide_round_statistics_.end(),
              zero_float2());
    std::fill_n(photon_caustics_count_, PHOTON_MAP_COUNTERS, 0u);
    photon_caustics_count_[PHOTON_PROBE_ENABLED] = probe;
    photon_caustics_cache_miss_ = 0;
    parallel_for(uint(0), capacity, [&](uint photon_index) {
      if (is_cancel_requested()) {
        return;
      }
      ThreadKernelGlobalsCPU *kg = kernel_thread_globals_get(*kernel_thread_globals_);
      IntegratorStateCPU state;
      path_state_init_queues(&state);
      kernels_.integrator_photon_caustics_emit(kg, &state, photon_index, cache.iteration);
    });

    /* parallel_for joins all writers. Camera kernels only query the completed map afterwards. */
    if (is_cancel_requested() || device_->have_error()) {
      return false;
    }
    if (photon_caustics_cache_miss_ == 0) {
      for (size_t i = 0; i < photon_area_guide_round_statistics_.size(); ++i) {
        photon_area_guide_statistics_[i].x += photon_area_guide_round_statistics_[i].x;
        photon_area_guide_statistics_[i].y += photon_area_guide_round_statistics_[i].y;
      }
      if (integrator.use_photon_adaptive_radius) {
        if (cache.cell_statistics) {
          std::fill(
              photon_cell_statistics_.begin(), photon_cell_statistics_.end(), make_uint2(0, 0));
          parallel_for(uint(0), photon_caustics_count_[PHOTON_MAP_STORED], [&](uint stored_index) {
            if (!is_cancel_requested()) {
              ThreadKernelGlobalsCPU *kg = kernel_thread_globals_get(*kernel_thread_globals_);
              kernels_.integrator_photon_caustics_cells(kg, stored_index);
            }
          });
        }
        parallel_for(uint(0), photon_caustics_count_[PHOTON_MAP_STORED], [&](uint stored_index) {
          if (!is_cancel_requested()) {
            ThreadKernelGlobalsCPU *kg = kernel_thread_globals_get(*kernel_thread_globals_);
            kernels_.integrator_photon_caustics_reconstruct(kg, stored_index);
          }
        });
        if (is_cancel_requested()) {
          return false;
        }
      }
      record_photon_probe(photon_caustics_count_, capacity, 4u * 1024 * 1024);
      photon_caustics_cache_ = cache;
      photon_caustics_revision_ = device_scene_->photon_caustics_revision;
      photon_caustics_seed_ = photon_sampling_seed();
      photon_caustics_requested_count_ = integrator.photon_caustics_count;
      photon_caustics_round_ = round;
      photon_caustics_update_interval_ = update_interval;
      photon_caustics_valid_ = true;
      ++photon_caustics_build_count_;
      const double build_seconds = build_timer.get_time();
      photon_caustics_build_seconds_ += build_seconds;
      LOG_DEBUG << "CPU photon caustics: emitted " << capacity << ", stored "
                << photon_caustics_count_[PHOTON_MAP_STORED] << ", radius " << cache.radius << ", build "
                << photon_caustics_build_count_ << ", scene revision " << photon_caustics_revision_
                << ", round " << round << ", sample interval " << update_interval << ", seed "
                << cache.iteration << ", build seconds " << build_seconds << ", attempts "
                << attempt + 1 << ", LPE event stride " << event_stride << ", LPE cache bytes "
                << event_count * sizeof(KernelPhotonCausticsEvent) +
                       photon_caustics_path_indices_.size() * sizeof(uint)
                << ", query index bytes " << (photon_caustics_query_hash_.size() +
                                               photon_caustics_query_next_.size()) * sizeof(uint);
      return true;
    }
  }

  device_->set_error(
      "Photon caustics image cache did not converge after 16 attempts. The photon map was not "
      "used.");
  return false;
}

void PathTraceWorkCPU::render_samples(RenderStatistics &statistics,
                                      const int start_sample,
                                      const int samples_num,
                                      const int sample_offset)
{
  const int64_t image_width = effective_buffer_params_.width;
  const int64_t image_height = effective_buffer_params_.height;
  const int64_t total_pixels_num = image_width * image_height;

  if (device_->profiler.active()) {
    for (ThreadKernelGlobalsCPU &kernel_globals : *kernel_thread_globals_) {
      kernel_globals.start_profiling();
    }
  }

  tbb::task_arena local_arena = local_tbb_arena_create(device_);
  bool photon_map_ready = true;
  local_arena.execute([&]() {
    photon_map_ready = (device_scene_->data.film.photon_segment_stride && !photon_only_) ||
                       build_photon_caustics(start_sample);
    if (!photon_map_ready) {
      return;
    }
    parallel_for(int64_t(0), total_pixels_num, [&](int64_t work_index) {
      if (is_cancel_requested()) {
        return;
      }

      const int y = work_index / image_width;
      const int x = work_index - y * image_width;

      KernelWorkTile work_tile;
      work_tile.x = effective_buffer_params_.full_x + x;
      work_tile.y = effective_buffer_params_.full_y + y;
      work_tile.w = 1;
      work_tile.h = 1;
      work_tile.start_sample = start_sample;
      work_tile.sample_offset = photon_only_ ? ~uint(sample_offset) : uint(sample_offset);
      work_tile.num_samples = 1;
      work_tile.offset = effective_buffer_params_.offset;
      work_tile.stride = effective_buffer_params_.stride;

      ThreadKernelGlobalsCPU *kernel_globals = kernel_thread_globals_get(*kernel_thread_globals_);

      const bool train_guiding = kernel_globals->data.integrator.train_guiding;
      if (photon_only_) {
        /* Photon-only camera prefixes may query the guiding field, but do not train
         * it with deliberately omitted native emission and direct-light samples. */
        kernel_globals->data.integrator.train_guiding = false;
      }
      render_samples_full_pipeline(kernel_globals, work_tile, samples_num);
      kernel_globals->data.integrator.train_guiding = train_guiding;
    });
  });
  if (device_->profiler.active()) {
    for (ThreadKernelGlobalsCPU &kernel_globals : *kernel_thread_globals_) {
      kernel_globals.stop_profiling();
    }
  }

  statistics.occupancy = photon_map_ready ? 1.0f : 0.0f;
}

void PathTraceWorkCPU::render_samples_full_pipeline(ThreadKernelGlobalsCPU *kernel_globals,
                                                    const KernelWorkTile &work_tile,
                                                    const int samples_num)
{
  const bool has_bake = device_scene_->data.bake.use;

  IntegratorStateCPU integrator_states[2];

  IntegratorStateCPU *state = &integrator_states[0];
  IntegratorStateCPU *shadow_catcher_state = nullptr;

  if (device_scene_->data.integrator.has_shadow_catcher) {
    shadow_catcher_state = &integrator_states[1];
    path_state_init_queues(shadow_catcher_state);
  }

  KernelWorkTile sample_work_tile = work_tile;
  float *render_buffer = trace_buffer();

  float *query_combined = nullptr;
  QueryRGB before{};
  unique_ptr<QueryPixel> query;
  if (lpe_query_active()) {
    query = make_unique<QueryPixel>(work_tile.x, work_tile.y);
    kernel_globals->query = query.get();
    const auto &film = device_scene_->data.film;
    const int pixel = work_tile.offset + work_tile.y * work_tile.stride + work_tile.x;
    if (film.pass_combined != PASS_UNUSED) {
      query_combined = render_buffer + uint64_t(pixel) * film.pass_stride + film.pass_combined;
      before = {query_combined[0], query_combined[1], query_combined[2]};
    }
  }
  fast_timer render_timer;

  for (int sample = 0; sample < samples_num; ++sample) {
    if (query) {
      lpe_query_wait_if_paused();
    }
    if (is_cancel_requested()) {
      break;
    }

    if (has_bake) {
      if (!kernels_.integrator_init_from_bake(
              kernel_globals, state, &sample_work_tile, render_buffer))
      {
        break;
      }
    }
    else {
      if (!kernels_.integrator_init_from_camera(
              kernel_globals, state, &sample_work_tile, render_buffer))
      {
        break;
      }
    }

#if defined(WITH_PATH_GUIDING)
    if (kernel_globals->data.integrator.train_guiding) {
      assert(kernel_globals->opgl_path_segment_storage);
      assert(kernel_globals->opgl_path_segment_storage->GetNumSegments() == 0);

      kernels_.integrator_megakernel(kernel_globals, state, render_buffer);

      /* Push the generated sample data to the global sample data storage. */
      guiding_push_sample_data_to_global_storage(kernel_globals, state, render_buffer);

      /* No training for shadow catcher paths. */
      if (shadow_catcher_state) {
        kernel_globals->data.integrator.train_guiding = false;
        kernels_.integrator_megakernel(kernel_globals, shadow_catcher_state, render_buffer);
        kernel_globals->data.integrator.train_guiding = true;
      }
    }
    else
#endif
    {
      kernels_.integrator_megakernel(kernel_globals, state, render_buffer);
      if (shadow_catcher_state) {
        kernels_.integrator_megakernel(kernel_globals, shadow_catcher_state, render_buffer);
      }
    }

    if (kernel_globals->data.film.deep_physical_ptr) {
      deep_physical_finish(kernel_globals->data.film.deep_physical_ptr,
                           state->path.render_pixel_index,
                           kernel_globals->data.film.deep_lpe_channels);
    }
    if (kernel_globals->data.film.pass_render_time != PASS_UNUSED) {
      uint64_t time;
      if (render_timer.lap(time)) {
        ccl_global float *buffer = render_buffer + (uint64_t)state->path.render_pixel_index *
                                                       kernel_globals->data.film.pass_stride;
        *(buffer + kernel_globals->data.film.pass_render_time) += float(time);
      }
    }
    ++sample_work_tile.start_sample;
  }
  if (query) {
    kernel_globals->query = nullptr;
    const int count = sample_work_tile.start_sample - work_tile.start_sample;
    if (query_combined && count) {
      query->record({}, photon_only_, query_combined[0] - before[0],
                    query_combined[1] - before[1], query_combined[2] - before[2]);
      query->record({{-1, -1, -1}}, photon_only_, float(count), 0, 0);
      if (kernel_globals->data.integrator.use_photon_caustics &&
          !kernel_globals->data.film.photon_segment_stride && !kernel_globals->data.film.photon_compact)
      {
        query->record({{-1, -1, -1}}, true, float(count), 0, 0);
      }
    }
    query->flush();
  }
}

void PathTraceWorkCPU::copy_to_display(PathTraceDisplay *display,
                                       PassMode pass_mode,
                                       const int num_samples)
{
  half4 *rgba_half = display->map_texture_buffer();
  if (!rgba_half) {
    /* TODO(sergey): Look into using copy_to_display() if mapping failed. Might be needed for
     * some implementations of PathTraceDisplay which can not map memory? */
    return;
  }

  const KernelFilm &kfilm = device_scene_->data.film;

  const PassAccessor::PassAccessInfo pass_access_info = get_display_pass_access_info(pass_mode);
  if (pass_access_info.type == PASS_NONE) {
    display->unmap_texture_buffer();
    return;
  }

  const BufferParams &effective_buffer_params = (pass_mode == PassMode::DENOISED) ?
                                                    effective_denoised_buffer_params_ :
                                                    effective_buffer_params_;

  const PassAccessorCPU pass_accessor(pass_access_info, kfilm.exposure, num_samples);

  PassAccessor::Destination destination = get_display_destination_template(display, pass_mode);
  destination.pixels_half_rgba = rgba_half;

  tbb::task_arena local_arena = local_tbb_arena_create(device_);
  local_arena.execute([&]() {
    pass_accessor.get_render_tile_pixels(buffers_.get(), effective_buffer_params, destination);
  });

  display->unmap_texture_buffer();
}

void PathTraceWorkCPU::destroy_gpu_resources(PathTraceDisplay * /*display*/) {}

bool PathTraceWorkCPU::copy_render_buffers_from_device()
{
  return buffers_->copy_from_device();
}

bool PathTraceWorkCPU::copy_render_buffers_to_device()
{
  buffers_->buffer.copy_to_device();
  return true;
}

bool PathTraceWorkCPU::zero_render_buffers()
{
  photon_guide_reset_ = true;
  reset_photon_sampling();
  buffers_->zero();
  return true;
}

int PathTraceWorkCPU::adaptive_sampling_converge_filter_count_active_impl(const float threshold,
                                                                          bool reset)
{
  const int full_x = effective_buffer_params_.full_x;
  const int full_y = effective_buffer_params_.full_y;
  const int width = effective_buffer_params_.width;
  const int height = effective_buffer_params_.height;
  const int offset = effective_buffer_params_.offset;
  const int stride = effective_buffer_params_.stride;

  float *render_buffer = trace_buffer();

  uint num_active_pixels = 0;

  tbb::task_arena local_arena = local_tbb_arena_create(device_);

  /* Check convergency and do x-filter in a single `parallel_for`, to reduce threading overhead. */
  local_arena.execute([&]() {
    parallel_for(full_y, full_y + height, [&](int y) {
      ThreadKernelGlobalsCPU *kernel_globals = kernel_thread_globals_->data();

      bool row_converged = true;
      uint num_row_pixels_active = 0;
      for (int x = 0; x < width; ++x) {
        if (!kernels_.adaptive_sampling_convergence_check(
                kernel_globals, render_buffer, full_x + x, y, threshold, reset, offset, stride))
        {
          ++num_row_pixels_active;
          row_converged = false;
        }
      }

      atomic_fetch_and_add_uint32(&num_active_pixels, num_row_pixels_active);

      if (!row_converged) {
        kernels_.adaptive_sampling_filter_x(
            kernel_globals, render_buffer, y, full_x, width, offset, stride);
      }
    });
  });

  if (num_active_pixels) {
    local_arena.execute([&]() {
      parallel_for(full_x, full_x + width, [&](int x) {
        ThreadKernelGlobalsCPU *kernel_globals = kernel_thread_globals_->data();
        kernels_.adaptive_sampling_filter_y(
            kernel_globals, render_buffer, x, full_y, height, offset, stride);
      });
    });
  }

  return num_active_pixels;
}

void PathTraceWorkCPU::merge_photon_estimates()
{
  if (!device_scene_->data.film.photon_segment_stride) {
    return;
  }
  const int count = effective_buffer_params_.width * effective_buffer_params_.height;
  tbb::task_arena arena = local_tbb_arena_create(device_);
  arena.execute([&]() {
    parallel_for(0, count, [&](int pixel) {
      kernels_.photon_merge(kernel_thread_globals_->data(), buffers_->buffer.data(), pixel);
    });
  });
}

void PathTraceWorkCPU::cryptomatte_postproces()
{
  const int width = effective_buffer_params_.width;
  const int height = effective_buffer_params_.height;

  const int segment = device_scene_->data.film.photon_segment_stride;
  const bool photon_metadata = estimator_sample_offset_ >=
                               device_scene_->data.integrator.photon_native_samples;
  float *render_buffer = buffers_->buffer.data() + segment * (photon_metadata ? 2 : 1);

  tbb::task_arena local_arena = local_tbb_arena_create(device_);

  /* Check convergency and do x-filter in a single `parallel_for`, to reduce threading overhead. */
  local_arena.execute([&]() {
    parallel_for(0, height, [&](int y) {
      ThreadKernelGlobalsCPU *kernel_globals = kernel_thread_globals_->data();
      int pixel_index = y * width;

      for (int x = 0; x < width; ++x, ++pixel_index) {
        kernels_.cryptomatte_postprocess(kernel_globals, render_buffer, pixel_index);
      }
    });
  });
}

void PathTraceWorkCPU::denoise_volume_guiding_buffers()
{
  const int min_x = effective_buffer_params_.full_x;
  const int min_y = effective_buffer_params_.full_y;
  const int max_x = effective_buffer_params_.width + min_x;
  const int max_y = effective_buffer_params_.height + min_y;
  const int offset = effective_buffer_params_.offset;
  const int stride = effective_buffer_params_.stride;

  float *render_buffer = buffers_->buffer.data();

  tbb::task_arena local_arena = local_tbb_arena_create(device_);

  const blocked_range2d<int> range(min_x, max_x, min_y, max_y);

  /* Filter in x direction. */
  local_arena.execute([&]() {
    parallel_for(range, [&](const blocked_range2d<int> r) {
      ThreadKernelGlobalsCPU *kernel_globals = kernel_thread_globals_->data();
      for (int y = r.cols().begin(); y < r.cols().end(); ++y) {
        for (int x = r.rows().begin(); x < r.rows().end(); ++x) {
          kernels_.volume_guiding_filter_x(
              kernel_globals, render_buffer, y, x, min_x, max_x, offset, stride);
        }
      }
    });
  });

  /* Filter in y direction. Unlike `filter_x`, the inner loop of `filter_y` is serially run inside
   * the kernel, to avoid the need of intermediate buffers. */
  local_arena.execute([&]() {
    parallel_for(min_x, max_x, [&](int x) {
      ThreadKernelGlobalsCPU *kernel_globals = kernel_thread_globals_->data();
      kernels_.volume_guiding_filter_y(
          kernel_globals, render_buffer, x, min_y, max_y, offset, stride);
    });
  });
}

#if defined(WITH_PATH_GUIDING)
/* NOTE: It seems that this is called before every rendering iteration/progression and not once per
 * rendering. May be we find a way to call it only once per rendering. */
void PathTraceWorkCPU::guiding_init_kernel_globals(void *guiding_field,
                                                   void *sample_data_storage,
                                                   const bool train)
{
  /* Linking the global guiding structures (e.g., Field and SampleStorage) to the per-thread
   * kernel globals. */
  for (int thread_index = 0; thread_index < kernel_thread_globals_->size(); thread_index++) {
    ThreadKernelGlobalsCPU &kg = (*kernel_thread_globals_)[thread_index];
    openpgl::cpp::Field *field = (openpgl::cpp::Field *)guiding_field;

    /* Allocate sampling distributions. */
    kg.opgl_guiding_field = field;

#  if PATH_GUIDING_LEVEL >= 4
    if (kg.opgl_surface_sampling_distribution) {
      kg.opgl_surface_sampling_distribution.reset();
    }
    if (kg.opgl_volume_sampling_distribution) {
      kg.opgl_volume_sampling_distribution.reset();
    }

    if (field) {
      kg.opgl_surface_sampling_distribution =
          make_unique<openpgl::cpp::SurfaceSamplingDistribution>(field);
      kg.opgl_volume_sampling_distribution = make_unique<openpgl::cpp::VolumeSamplingDistribution>(
          field);
    }
#  endif

    /* Reserve storage for training. */
    kg.data.integrator.train_guiding = train;
    kg.opgl_sample_data_storage = (openpgl::cpp::SampleStorage *)sample_data_storage;

    if (train) {
      kg.opgl_path_segment_storage_size = kg.data.integrator.transparent_max_bounce +
                                          kg.data.integrator.max_bounce + 3;
      kg.opgl_path_segment_storage->Reserve(kg.opgl_path_segment_storage_size);
      kg.opgl_path_segment_storage->Clear();
    }
  }
}

void PathTraceWorkCPU::guiding_push_sample_data_to_global_storage(ThreadKernelGlobalsCPU *kg,
                                                                  IntegratorStateCPU *state,
                                                                  ccl_global float *ccl_restrict
                                                                      render_buffer)
{
#  ifdef WITH_CYCLES_DEBUG
  if (LOG_IS_ON(LOG_LEVEL_DEBUG)) {
    /* Check if the generated path segments contain valid values. */
    const bool validSegments = kg->opgl_path_segment_storage->ValidateSegments();
    if (!validSegments) {
      LOG_DEBUG << "Guiding: invalid path segments!";
    }
  }

  /* Write debug render pass to validate it matches combined pass. */
  pgl_vec3f pgl_final_color = kg->opgl_path_segment_storage->CalculatePixelEstimate(false);
  ccl_global float *buffer = film_pass_pixel_render_buffer(kg, state, render_buffer);
  float3 final_color = make_float3(pgl_final_color.x, pgl_final_color.y, pgl_final_color.z);
  if (kernel_data.film.pass_guiding_color != PASS_UNUSED) {
    film_write_pass_float3(buffer + kernel_data.film.pass_guiding_color, final_color);
  }
#  else
  (void)state;
  (void)render_buffer;
#  endif

  /* Convert the path segment representation of the random walk into radiance samples. */
#  if PATH_GUIDING_LEVEL >= 2
  const bool use_direct_light = kernel_data.integrator.use_guiding_direct_light;
  const bool use_mis_weights = kernel_data.integrator.use_guiding_mis_weights;
  kg->opgl_path_segment_storage->PrepareSamples(use_mis_weights, use_direct_light, false);
#  endif

#  ifdef WITH_CYCLES_DEBUG
  /* Check if the training/radiance samples generated by the path segment storage are valid. */
  if (LOG_IS_ON(LOG_LEVEL_DEBUG)) {
    const bool validSamples = kg->opgl_path_segment_storage->ValidateSamples();
    if (!validSamples) {
      LOG_DEBUG
          << "Guiding: path segment storage generated/contains invalid radiance/training samples!";
    }
  }
#  endif

#  if PATH_GUIDING_LEVEL >= 3
  /* Push radiance samples from current random walk/path to the global sample storage. */
  size_t num_samples = 0;
  const openpgl::cpp::SampleData *samples = kg->opgl_path_segment_storage->GetSamples(num_samples);
  kg->opgl_sample_data_storage->AddSamples(samples, num_samples);
#  endif

  /* Clear storage for the current path, to be ready for the next path. */
  kg->opgl_path_segment_storage->Clear();
}
#endif

CCL_NAMESPACE_END
