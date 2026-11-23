/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#include "integrator/path_trace_work_gpu.h"
#include "integrator/photon_guiding.h"
#include "integrator/photon_spectrum.h"
#include "kernel/integrator/vcm_types.h"

#include <limits>
#include <cstdlib>
#include <new>

#include "device/device.h"

#include "integrator/pass_accessor_gpu.h"
#include "integrator/path_trace_display.h"

#include "scene/scene.h"
#include "session/buffers.h"

#include "util/hash.h"
#include "util/log.h"
#include "util/lpe_query.h"
#include "util/path.h"
#include "util/string.h"
#include "util/time.h"

#include "kernel/device/gpu/block_sizes.h"
#include "kernel/integrator/shadow_state.h"
#include "kernel/types.h"
#include "kernel/sample/photon_spectrum.h"

CCL_NAMESPACE_BEGIN

static size_t estimate_single_state_size(const uint64_t kernel_features,
                                         const int volume_stack_size, const bool use_shadow_color)
{
  size_t state_size = 0;

#define KERNEL_STRUCT_VOLUME_STACK_SIZE (volume_stack_size)
#define KERNEL_STRUCT_BEGIN(name) for (int array_index = 0;; array_index++) {

#ifdef __INTEGRATOR_GPU_PACKED_STATE__
#  define KERNEL_STRUCT_MEMBER(parent_struct, type, name, feature) \
    state_size += (KernelFeatureRequest(feature).test(kernel_features)) ? \
                      sizeof(gpu_state_storage<type>::gpu_type) : \
                      0;
#  define KERNEL_STRUCT_MEMBER_PACKED(parent_struct, type, name, feature)
#  define KERNEL_STRUCT_BEGIN_PACKED(parent_struct, feature) \
    KERNEL_STRUCT_BEGIN(parent_struct) \
    KERNEL_STRUCT_MEMBER(parent_struct, packed_##parent_struct, packed, feature)
#else
#  define KERNEL_STRUCT_MEMBER(parent_struct, type, name, feature) \
    state_size += (KernelFeatureRequest(feature).test(kernel_features)) ? \
                      sizeof(gpu_state_storage<type>::gpu_type) : \
                      0;
#  define KERNEL_STRUCT_MEMBER_PACKED KERNEL_STRUCT_MEMBER
#  define KERNEL_STRUCT_BEGIN_PACKED(parent_struct, feature) KERNEL_STRUCT_BEGIN(parent_struct)
#endif

#define KERNEL_STRUCT_ARRAY_MEMBER(parent_struct, type, name, feature) \
  state_size += (KernelFeatureRequest(feature).test(kernel_features)) ? \
                    sizeof(gpu_state_storage<type>::gpu_type) : \
                    0;
#define KERNEL_STRUCT_MEMBER_RUNTIME_SHADOW_COLOR(parent_struct, type, name, feature) \
  state_size += shadow_state_unshadowed_throughput_is_needed(kernel_features, use_shadow_color) ? \
                    sizeof(gpu_state_storage<type>::gpu_type) : \
                    0;
#define KERNEL_STRUCT_END(name) \
  (void)array_index; \
  break; \
  }
#define KERNEL_STRUCT_END_ARRAY(name, cpu_array_size, gpu_array_size) \
  if (array_index >= gpu_array_size - 1) { \
    break; \
  } \
  }

#include "kernel/integrator/state_template.h"

#include "kernel/integrator/shadow_state_template.h"

#undef KERNEL_STRUCT_VOLUME_STACK_SIZE
#undef KERNEL_STRUCT_BEGIN
#undef KERNEL_STRUCT_BEGIN_PACKED
#undef KERNEL_STRUCT_MEMBER
#undef KERNEL_STRUCT_MEMBER_PACKED
#undef KERNEL_STRUCT_ARRAY_MEMBER
#undef KERNEL_STRUCT_MEMBER_RUNTIME_SHADOW_COLOR
#undef KERNEL_STRUCT_END
#undef KERNEL_STRUCT_END_ARRAY

  return state_size;
}

PathTraceWorkGPU::PathTraceWorkGPU(Device *device,
                                   Film *film,
                                   DeviceScene *device_scene,
                                   const bool *cancel_requested_flag)
    : PathTraceWork(device, film, device_scene, cancel_requested_flag),
      queue_(device->gpu_queue_create()),
      preview_pass_(device, "LPE viewport input", MEM_READ_WRITE),
      integrator_state_soa_kernel_features_(0),
      /* Use MEM_FLAG_NO_HOST_FALLBACK since having this on the GPU is more important
       * than scene memory for performance. */
      integrator_queue_counter_(
          device, "integrator_queue_counter", MEM_READ_WRITE, MEM_FLAG_NO_HOST_FALLBACK),
      integrator_shader_sort_counter_(
          device, "integrator_shader_sort_counter", MEM_READ_WRITE, MEM_FLAG_NO_HOST_FALLBACK),
      integrator_shader_raytrace_sort_counter_(device,
                                               "integrator_shader_raytrace_sort_counter",
                                               MEM_READ_WRITE,
                                               MEM_FLAG_NO_HOST_FALLBACK),
      integrator_shader_sort_prefix_sum_(
          device, "integrator_shader_sort_prefix_sum", MEM_READ_WRITE, MEM_FLAG_NO_HOST_FALLBACK),
      integrator_shader_sort_partition_key_offsets_(device,
                                                    "integrator_shader_sort_partition_key_offsets",
                                                    MEM_READ_WRITE,
                                                    MEM_FLAG_NO_HOST_FALLBACK),
      integrator_next_main_path_index_(
          device, "integrator_next_main_path_index", MEM_READ_WRITE, MEM_FLAG_NO_HOST_FALLBACK),
      integrator_next_shadow_path_index_(
          device, "integrator_next_shadow_path_index", MEM_READ_WRITE, MEM_FLAG_NO_HOST_FALLBACK),
      gpu_guiding_cdf_(device, "gpu_guiding_cdf"),
      gpu_guiding_statistics_(device, "gpu_guiding_statistics"),
      gpu_guiding_ready_cells_(device, "gpu_guiding_ready_cells", MEM_READ_WRITE),
      photon_caustics_(device, "photon_caustics"),
      photon_caustics_hash_(device, "photon_caustics_hash"),
      photon_caustics_query_hash_(device, "photon_caustics_query_hash"),
      photon_caustics_query_next_(device, "photon_caustics_query_next"),
      photon_caustics_count_(device, "photon_caustics_count", MEM_READ_WRITE),
      photon_caustics_stored_indices_(device, "photon_caustics_stored_indices"),
      photon_caustics_pilot_footprints_(device, "photon_caustics_pilot_footprints"),
      photon_cell_statistics_(device, "photon_cell_statistics"),
      photon_reconstruction_metadata_(device, "photon_reconstruction_metadata"),
      photon_beam_metadata_(device, "photon_beam_metadata"),
      photon_caustics_events_(device, "photon_caustics_events"),
      photon_caustics_path_indices_(device, "photon_caustics_path_indices"),
      photon_guide_(device, "photon_guide", MEM_READ_ONLY),
      photon_guide_statistics_(device, "photon_guide_statistics", MEM_READ_WRITE),
      photon_guide_hits_(device, "photon_guide_hits"),
      photon_area_guide_cdf_(device, "photon_area_guide_cdf", MEM_READ_ONLY),
      photon_area_guide_statistics_(device, "photon_area_guide_statistics", MEM_READ_WRITE),
      photon_area_guide_round_statistics_(device, "photon_area_guide_round_statistics"),
      queued_paths_(device, "queued_paths", MEM_READ_WRITE, MEM_FLAG_NO_HOST_FALLBACK),
      num_queued_paths_(device, "num_queued_paths", MEM_READ_WRITE, MEM_FLAG_NO_HOST_FALLBACK),
      work_tiles_(device, "work_tiles", MEM_READ_WRITE),
      display_rgba_half_(device, "display buffer half", MEM_READ_WRITE),
      max_num_paths_(0),
      min_num_active_main_paths_(0),
      max_active_main_path_index_(0)
{
  memset(&integrator_state_gpu_, 0, sizeof(integrator_state_gpu_));
}

void PathTraceWorkGPU::alloc_integrator_soa()
{
  /* IntegrateState allocated as structure of arrays. */

  /* Check if we already allocated memory for the required features.
   * Note that both disabling and enabling features may require memory
   * allocations, so we check for equality. */
  const int volume_stack_size = device_scene_->data.volume_stack_size;
  const uint64_t kernel_features = device_scene_->data.kernel_features;
  const bool use_shadow_color = device_scene_->data.integrator.light_flags &
                                KERNEL_INTEGRATOR_SHADOW_COLOR;
  if (integrator_state_soa_kernel_features_ == kernel_features &&
      integrator_state_soa_use_shadow_color_ == use_shadow_color &&
      integrator_state_soa_volume_stack_size_ == volume_stack_size)
  {
    return;
  }
  integrator_state_soa_kernel_features_ = kernel_features;
  integrator_state_soa_use_shadow_color_ = use_shadow_color;
  integrator_state_soa_volume_stack_size_ = volume_stack_size;

  /* Free existing state, before state size estimation which takes into account free memory. */
  integrator_state_soa_.clear();

  /* Determine the number of path states. Deferring this for as long as possible allows the
   * back-end to make better decisions about memory availability. */
  const size_t single_state_size = estimate_single_state_size(
      kernel_features, integrator_state_soa_volume_stack_size_, use_shadow_color);

  max_num_paths_ = diagnostic_slots_ ? diagnostic_slots_ : queue_->num_concurrent_states(single_state_size);

  /* A 1:4 busy:total ratio was found to give good performance, independent of the
   * total state count. This is the number of states which keeps the device occupied
   * with work without losing performance. The renderer will add more work (when
   * available) when the number of active paths falls below this value. */
  min_num_active_main_paths_ = max_num_paths_ / 4;

  /* Allocate a device only memory buffer before for each struct member, and then
   * write the pointers into a struct that resides in constant memory.
   *
   * TODO: store float3 in separate XYZ arrays. */
#define KERNEL_STRUCT_BEGIN(name) for (int array_index = 0;; array_index++) {
#define KERNEL_STRUCT_MEMBER_IMPL(parent_struct, type, name, condition) \
  integrator_state_gpu_.parent_struct.name = nullptr; \
  if (condition) { \
    string name_str = string_printf("%sintegrator_state_" #parent_struct "_" #name, \
                                    shadow ? "shadow_" : ""); \
    auto array = make_unique<device_only_memory<gpu_state_storage<type>::gpu_type>>( \
        device_, name_str.c_str()); \
    array->alloc_to_device(max_num_paths_); \
    memcpy(&integrator_state_gpu_.parent_struct.name, \
           &array->device_pointer, \
           sizeof(array->device_pointer)); \
    integrator_state_soa_.emplace_back(std::move(array)); \
  }
#define KERNEL_STRUCT_MEMBER(parent_struct, type, name, feature) \
  KERNEL_STRUCT_MEMBER_IMPL( \
      parent_struct, type, name, KernelFeatureRequest(feature).test(kernel_features))
#define KERNEL_STRUCT_MEMBER_RUNTIME_SHADOW_COLOR(parent_struct, type, name, feature) \
  KERNEL_STRUCT_MEMBER_IMPL( \
      parent_struct, \
      type, \
      name, \
      shadow_state_unshadowed_throughput_is_needed(kernel_features, use_shadow_color))
#ifdef __INTEGRATOR_GPU_PACKED_STATE__
#  define KERNEL_STRUCT_MEMBER_PACKED(parent_struct, type, name, feature) \
    if ((KernelFeatureRequest(feature).test(kernel_features))) { \
      string name_str = string_printf("%sintegrator_state_" #parent_struct "_" #name, \
                                      shadow ? "shadow_" : ""); \
      LOG_TRACE << "Skipping " << name_str \
                << " -- data is packed inside integrator_state_" #parent_struct "_packed"; \
    }
#  define KERNEL_STRUCT_BEGIN_PACKED(parent_struct, feature) \
    KERNEL_STRUCT_BEGIN(parent_struct) \
    KERNEL_STRUCT_MEMBER(parent_struct, packed_##parent_struct, packed, feature)
#else
#  define KERNEL_STRUCT_MEMBER_PACKED KERNEL_STRUCT_MEMBER
#  define KERNEL_STRUCT_BEGIN_PACKED(parent_struct, feature) KERNEL_STRUCT_BEGIN(parent_struct)
#endif

#define KERNEL_STRUCT_ARRAY_MEMBER(parent_struct, type, name, feature) \
  integrator_state_gpu_.parent_struct[array_index].name = nullptr; \
  if (KernelFeatureRequest(feature).test(kernel_features)) { \
    string name_str = string_printf( \
        "%sintegrator_state_" #name "_%d", shadow ? "shadow_" : "", array_index); \
    auto array = make_unique<device_only_memory<gpu_state_storage<type>::gpu_type>>( \
        device_, name_str.c_str()); \
    array->alloc_to_device(max_num_paths_); \
    memcpy(&integrator_state_gpu_.parent_struct[array_index].name, \
           &array->device_pointer, \
           sizeof(array->device_pointer)); \
    integrator_state_soa_.emplace_back(std::move(array)); \
  }
#define KERNEL_STRUCT_END(name) \
  (void)array_index; \
  break; \
  }
#define KERNEL_STRUCT_END_ARRAY(name, cpu_array_size, gpu_array_size) \
  if (array_index >= gpu_array_size - 1) { \
    break; \
  } \
  }
#define KERNEL_STRUCT_VOLUME_STACK_SIZE (integrator_state_soa_volume_stack_size_)

  bool shadow = false;
#include "kernel/integrator/state_template.h"

  shadow = true;
#include "kernel/integrator/shadow_state_template.h"

#undef KERNEL_STRUCT_BEGIN
#undef KERNEL_STRUCT_BEGIN_PACKED
#undef KERNEL_STRUCT_MEMBER
#undef KERNEL_STRUCT_MEMBER_IMPL
#undef KERNEL_STRUCT_MEMBER_RUNTIME_SHADOW_COLOR
#undef KERNEL_STRUCT_MEMBER_PACKED
#undef KERNEL_STRUCT_ARRAY_MEMBER
#undef KERNEL_STRUCT_END
#undef KERNEL_STRUCT_END_ARRAY
#undef KERNEL_STRUCT_VOLUME_STACK_SIZE

  if (LOG_IS_ON(LOG_LEVEL_TRACE)) {
    size_t total_soa_size = 0;
    for (auto &&soa_memory : integrator_state_soa_) {
      total_soa_size += soa_memory->memory_size();
    }

    LOG_TRACE << "GPU SoA state size: " << string_human_readable_size(total_soa_size);
  }
}

void PathTraceWorkGPU::alloc_integrator_queue()
{
  if (integrator_queue_counter_.size() == 0) {
    integrator_queue_counter_.alloc(1);
    integrator_queue_counter_.zero_to_device();
    integrator_queue_counter_.copy_from_device();
    integrator_state_gpu_.queue_counter = (IntegratorQueueCounter *)
                                              integrator_queue_counter_.device_pointer;
  }

  /* Allocate data for active path index arrays. */
  if (num_queued_paths_.size() == 0) {
    num_queued_paths_.alloc(1);
    num_queued_paths_.zero_to_device();
  }

  if (queued_paths_.size() != max_num_paths_) {
    queued_paths_.alloc(max_num_paths_);
    /* TODO: this could be skip if we had a function to just allocate on device. */
    queued_paths_.zero_to_device();
  }
}

void PathTraceWorkGPU::alloc_integrator_sorting()
{
  num_sort_partitions_ = queue_->num_sort_partitions(max_num_paths_,
                                                     device_scene_->data.max_shaders);

  integrator_state_gpu_.sort_partition_divisor = (int)divide_up(max_num_paths_,
                                                                num_sort_partitions_);

  if (num_sort_partitions_ > 1 && queue_->supports_local_atomic_sort()) {
    /* Allocate array for partitioned shader sorting using local atomics. */
    const int num_offsets = (device_scene_->data.max_shaders + 1) * num_sort_partitions_;
    if (integrator_shader_sort_partition_key_offsets_.size() < num_offsets) {
      integrator_shader_sort_partition_key_offsets_.alloc(num_offsets);
      integrator_shader_sort_partition_key_offsets_.zero_to_device();
    }
    integrator_state_gpu_.sort_partition_key_offsets =
        (int *)integrator_shader_sort_partition_key_offsets_.device_pointer;

    integrator_state_gpu_.sort_key_counter[DEVICE_KERNEL_INTEGRATOR_SHADE_SURFACE] = nullptr;
    integrator_state_gpu_.sort_key_counter[DEVICE_KERNEL_INTEGRATOR_SHADE_SURFACE_RAYTRACE] =
        nullptr;
  }
  else {
    /* Allocate arrays for shader sorting. */
    const int sort_buckets = device_scene_->data.max_shaders * num_sort_partitions_;
    if (integrator_shader_sort_counter_.size() < sort_buckets) {
      integrator_shader_sort_counter_.alloc(sort_buckets);
      integrator_shader_sort_counter_.zero_to_device();

      integrator_shader_sort_prefix_sum_.alloc(sort_buckets);
      integrator_shader_sort_prefix_sum_.zero_to_device();
    }

    const bool use_raytrace = device_scene_->data.kernel_features & KERNEL_FEATURE_NODE_RAYTRACE;
    if (use_raytrace && integrator_shader_raytrace_sort_counter_.size() < sort_buckets) {
      integrator_shader_raytrace_sort_counter_.alloc(sort_buckets);
      integrator_shader_raytrace_sort_counter_.zero_to_device();
    }

    integrator_state_gpu_.sort_key_counter[DEVICE_KERNEL_INTEGRATOR_SHADE_SURFACE] =
        (int *)integrator_shader_sort_counter_.device_pointer;
    integrator_state_gpu_.sort_key_counter[DEVICE_KERNEL_INTEGRATOR_SHADE_SURFACE_RAYTRACE] =
        (use_raytrace) ? (int *)integrator_shader_raytrace_sort_counter_.device_pointer : nullptr;
  }
}

void PathTraceWorkGPU::alloc_integrator_path_split()
{
  if (integrator_next_shadow_path_index_.size() == 0) {
    integrator_next_shadow_path_index_.alloc(1);
    integrator_next_shadow_path_index_.zero_to_device();

    integrator_state_gpu_.next_shadow_path_index =
        (int *)integrator_next_shadow_path_index_.device_pointer;
  }

  if (integrator_next_main_path_index_.size() == 0) {
    integrator_next_main_path_index_.alloc(1);
    integrator_next_shadow_path_index_.data()[0] = 0;
    integrator_next_main_path_index_.zero_to_device();

    integrator_state_gpu_.next_main_path_index =
        (int *)integrator_next_main_path_index_.device_pointer;
  }
}

void PathTraceWorkGPU::alloc_photon_caustics()
{
  const KernelIntegrator &integrator = device_scene_->data.integrator;
  if ((device_->info.type != DEVICE_OPTIX && device_->info.type != DEVICE_CUDA) ||
      !integrator.use_photon_caustics || integrator.photon_caustics_count <= 0 ||
      max_num_paths_ == 0)
  {
    photon_caustics_.free();
    photon_caustics_hash_.free();
    photon_caustics_query_hash_.free();
    photon_caustics_query_next_.free();
    photon_caustics_count_.free();
    photon_caustics_stored_indices_.free();
    photon_caustics_pilot_footprints_.free();
    photon_reconstruction_metadata_.free();
    photon_cell_statistics_.free();
    photon_beam_metadata_.free();
    photon_caustics_events_.free();
    photon_caustics_path_indices_.free();
    photon_guide_.free();
    photon_guide_statistics_.free();
    photon_guide_hits_.free();
    photon_area_guide_cdf_.free();
    photon_area_guide_statistics_.free();
    photon_area_guide_round_statistics_.free();
    integrator_state_gpu_.photon_area_guide_cdf = nullptr;
    integrator_state_gpu_.photon_area_guide_statistics = nullptr;
    integrator_state_gpu_.photon_area_guide_round_statistics = nullptr;
    photon_guide_reset_ = true;
    integrator_state_gpu_.photon_guide = nullptr;
    integrator_state_gpu_.photon_guide_statistics = nullptr;
    integrator_state_gpu_.photon_guide_hits = nullptr;
    integrator_state_gpu_.photon_guide_fraction = 0.0f;
    integrator_state_gpu_.photon_caustics = nullptr;
    integrator_state_gpu_.photon_caustics_hash = nullptr;
    integrator_state_gpu_.photon_caustics_query_hash = nullptr;
    integrator_state_gpu_.photon_caustics_query_next = nullptr;
    integrator_state_gpu_.photon_caustics_count = nullptr;
    integrator_state_gpu_.photon_caustics_stored_indices = nullptr;
    integrator_state_gpu_.photon_caustics_pilot_footprints = nullptr;
    integrator_state_gpu_.photon_cell_statistics = nullptr;
    integrator_state_gpu_.photon_reconstruction_metadata = nullptr;
    integrator_state_gpu_.photon_beam_metadata = nullptr;
    integrator_state_gpu_.photon_caustics_events = nullptr;
    integrator_state_gpu_.photon_caustics_path_indices = nullptr;
    integrator_state_gpu_.photon_caustics_capacity = 0;
    integrator_state_gpu_.photon_caustics_hash_size = 0;
    integrator_state_gpu_.photon_caustics_iteration = 0;
    integrator_state_gpu_.photon_caustics_radius = 0.0f;
    integrator_state_gpu_.photon_caustics_event_stride = 0;
    photon_caustics_valid_ = false;
    return;
  }

  /* One photon path borrows one SoA slot and stores at most one photon. A power-of-two hash
   * table at least twice as large keeps hash chains short without a second spatial structure. */
  const uint capacity = uint(min(integrator.photon_caustics_count, max_num_paths_));
  const bool query_index = integrator.use_photon_query_index;
  const bool ellipse_confidence =
      integrator.use_photon_adaptive_radius &&
      (integrator.photon_reconstruction_method == PHOTON_RECONSTRUCTION_ELLIPSE_COVERAGE ||
       (integrator.photon_reconstruction_method != PHOTON_RECONSTRUCTION_REGIONAL &&
        integrator.use_photon_gradient_guard && integrator.photon_gradient_strength > 0.0f &&
        integrator.photon_caustics_differentials));
  const bool path_filter = integrator.photon_reconstruction_filter &
                           PHOTON_RECONSTRUCTION_FILTER_PATH;
  /* Memory is prepared while the native matcher may be selected. Reserve history
   * for the later photon program too, including photon-only LPE graphs. Keep it
   * stable across phases so map reuse never reads a freed/null event buffer. */
  const uint event_stride = (diagnostic_slots_ || device_scene_->data.film.lpe_state_count ||
                             device_scene_->data.film.lpe_photon_count || path_filter) ?
                                uint(max(integrator.max_bounce, 1) +
                                     integrator.transparent_max_bounce) :
                                0;
  uint hash_size = 1;
  while (hash_size < 2u * capacity) {
    hash_size <<= 1;
  }
  if (integrator_state_gpu_.photon_caustics_capacity != capacity ||
      integrator_state_gpu_.photon_caustics_hash_size != hash_size ||
      integrator_state_gpu_.photon_caustics_event_stride != event_stride ||
      bool(integrator_state_gpu_.photon_caustics_query_hash) != query_index)
  {
    photon_caustics_valid_ = false;
  }
  const size_t path_index_bytes = event_stride ? size_t(capacity) * sizeof(uint) : 0;
  const size_t max_events = (std::numeric_limits<size_t>::max() - path_index_bytes) /
                            sizeof(KernelPhotonCausticsEvent);
  if (event_stride && size_t(capacity) > max_events / event_stride) {
    photon_caustics_valid_ = false;
    device_->set_error("Photon caustics LPE event allocation size overflow.");
    return;
  }
  const size_t event_count = size_t(capacity) * event_stride;
  try {
    const size_t model_size = size_t(integrator.num_lights) * PHOTON_GUIDE_COMPONENTS;
    if (integrator.use_photon_guiding) {
      if (photon_guide_.size() != model_size) {
        photon_guide_.alloc(model_size);
        photon_guide_statistics_.alloc(model_size);
        photon_guide_reset_ = true;
      }
    }
    else {
      photon_guide_.free();
      photon_guide_statistics_.free();
      /* Work memory is checked at every scheduler batch. A disabled guide must not invalidate
       * an otherwise reusable photon map. Re-enabling it allocates a fresh model above. */
    }
    if (integrator.use_photon_area_guiding) {
      const size_t size = size_t(integrator.num_lights) * PHOTON_AREA_GUIDE_CELLS;
      if (photon_area_guide_cdf_.size() != size) {
        photon_area_guide_cdf_.alloc(size);
        photon_area_guide_statistics_.alloc(size);
        photon_guide_reset_ = true;
      }
      photon_area_guide_round_statistics_.alloc_to_device(size);
    }
    else {
      photon_area_guide_cdf_.free();
      photon_area_guide_statistics_.free();
      photon_area_guide_round_statistics_.free();
    }
    if (integrator.use_photon_guiding || integrator.use_photon_area_guiding) {
      photon_guide_hits_.alloc_to_device(capacity);
    }
    else {
      photon_guide_hits_.free();
    }
    photon_caustics_.alloc_to_device(capacity);
    if (integrator.use_photon_cell_statistics) {
      photon_cell_statistics_.alloc_to_device(hash_size + 1);
    }
    else {
      photon_cell_statistics_.free();
    }
    photon_caustics_hash_.alloc_to_device(hash_size);
    if (query_index) {
      photon_caustics_query_hash_.alloc_to_device(hash_size);
      photon_caustics_query_next_.alloc_to_device(capacity);
    }
    else {
      photon_caustics_query_hash_.free();
      photon_caustics_query_next_.free();
    }
    if (ellipse_confidence) {
      photon_caustics_pilot_footprints_.alloc_to_device(capacity);
    }
    else {
      photon_caustics_pilot_footprints_.free();
    }
    if (integrator.use_photon_adaptive_radius) {
      photon_caustics_stored_indices_.alloc_to_device(capacity);
    }
    else {
      photon_caustics_stored_indices_.free();
    }
    if (photon_caustics_count_.size() == 0) {
      photon_caustics_count_.alloc(1);
      photon_caustics_count_.zero_to_device();
    }
    if (integrator.photon_reconstruction_filter != PHOTON_RECONSTRUCTION_FILTER_LEGACY) {
      photon_reconstruction_metadata_.alloc_to_device(capacity);
    }
    else {
      photon_reconstruction_metadata_.free();
    }
    if (integrator.use_photon_beam_continuity) {
      photon_beam_metadata_.alloc_to_device(capacity);
    }
    else {
      photon_beam_metadata_.free();
    }
    if (event_stride) {
      photon_caustics_events_.alloc_to_device(event_count);
      photon_caustics_path_indices_.alloc_to_device(capacity);
    }
    else {
      photon_caustics_events_.free();
      photon_caustics_path_indices_.free();
    }
  }
  catch (const std::bad_alloc &) {
    photon_caustics_valid_ = false;
    device_->set_error(
        "Unable to allocate the photon caustics cache and complete LPE event "
        "history. Reduce Photon Count or Max Bounces.");
    return;
  }
  if (device_->have_error() || !photon_caustics_.device_pointer ||
      !photon_caustics_hash_.device_pointer || !photon_caustics_count_.device_pointer ||
      (query_index && (!photon_caustics_query_hash_.device_pointer ||
                       !photon_caustics_query_next_.device_pointer)) ||
      (integrator.use_photon_adaptive_radius && !photon_caustics_stored_indices_.device_pointer) ||
      (ellipse_confidence && !photon_caustics_pilot_footprints_.device_pointer) ||
      (integrator.use_photon_cell_statistics && !photon_cell_statistics_.device_pointer) ||
      (integrator.photon_reconstruction_filter &&
       !photon_reconstruction_metadata_.device_pointer) ||
      (integrator.use_photon_beam_continuity && !photon_beam_metadata_.device_pointer) ||
      (event_stride &&
       (!photon_caustics_events_.device_pointer || !photon_caustics_path_indices_.device_pointer)))
  {
    photon_caustics_valid_ = false;
    device_->set_error(
        "Unable to allocate the photon caustics cache and complete LPE event "
        "history. Reduce Photon Count or Max Bounces.");
    return;
  }

  integrator_state_gpu_.photon_guide_hits = reinterpret_cast<float4 *>(
      photon_guide_hits_.device_pointer);
  integrator_state_gpu_.photon_caustics = reinterpret_cast<KernelPhotonCaustic *>(
      photon_caustics_.device_pointer);
  integrator_state_gpu_.photon_cell_statistics = reinterpret_cast<uint2 *>(
      photon_cell_statistics_.device_pointer);
  integrator_state_gpu_.photon_caustics_hash = reinterpret_cast<uint *>(
      photon_caustics_hash_.device_pointer);
  integrator_state_gpu_.photon_caustics_count = reinterpret_cast<uint *>(
      photon_caustics_count_.device_pointer);
  integrator_state_gpu_.photon_caustics_query_hash = reinterpret_cast<uint *>(
      photon_caustics_query_hash_.device_pointer);
  integrator_state_gpu_.photon_caustics_query_next = reinterpret_cast<uint *>(
      photon_caustics_query_next_.device_pointer);
  integrator_state_gpu_.photon_caustics_stored_indices = reinterpret_cast<uint *>(
      photon_caustics_stored_indices_.device_pointer);
  integrator_state_gpu_.photon_caustics_pilot_footprints = reinterpret_cast<float4 *>(
      photon_caustics_pilot_footprints_.device_pointer);
  integrator_state_gpu_.photon_beam_metadata = reinterpret_cast<KernelPhotonBeamMetadata *>(
      photon_beam_metadata_.device_pointer);
  integrator_state_gpu_.photon_reconstruction_metadata =
      reinterpret_cast<KernelPhotonReconstructionMetadata *>(
          photon_reconstruction_metadata_.device_pointer);
  integrator_state_gpu_.photon_caustics_events = reinterpret_cast<KernelPhotonCausticsEvent *>(
      photon_caustics_events_.device_pointer);
  integrator_state_gpu_.photon_caustics_path_indices = reinterpret_cast<uint *>(
      photon_caustics_path_indices_.device_pointer);
  integrator_state_gpu_.photon_caustics_capacity = capacity;
  integrator_state_gpu_.photon_caustics_hash_size = hash_size;
  integrator_state_gpu_.photon_caustics_radius = max(integrator.photon_caustics_radius, 1.0e-6f);
  integrator_state_gpu_.photon_caustics_event_stride = event_stride;
}

void PathTraceWorkGPU::alloc_gpu_guiding()
{
  const bool enabled = (device_->info.type == DEVICE_OPTIX || device_->info.type == DEVICE_CUDA) &&
      (device_scene_->data.kernel_features & KERNEL_FEATURE_GPU_PATH_GUIDING) &&
      device_scene_->data.integrator.use_gpu_guiding;
  if (!enabled) {
    gpu_guiding_cdf_.free();
    gpu_guiding_statistics_.free();
    gpu_guiding_ready_cells_.free();
    integrator_state_gpu_.gpu_guiding_cdf = nullptr;
    integrator_state_gpu_.gpu_guiding_statistics = nullptr;
    integrator_state_gpu_.gpu_guiding_ready_cells = nullptr;
    integrator_state_gpu_.gpu_guiding_active = false;
    integrator_state_gpu_.gpu_guiding_training = false;
    gpu_guiding_reset_ = true;
    return;
  }
  if (!gpu_guiding_cdf_.device_pointer) {
    gpu_guiding_cdf_.alloc_to_device(GPU_GUIDE_CELLS * GPU_GUIDE_STRIDE);
    gpu_guiding_statistics_.alloc_to_device(GPU_GUIDE_CELLS * GPU_GUIDE_STRIDE);
    gpu_guiding_ready_cells_.alloc(1);
    gpu_guiding_ready_cells_.zero_to_device();
    gpu_guiding_reset_ = true;
  }
  integrator_state_gpu_.gpu_guiding_cdf = (float *)gpu_guiding_cdf_.device_pointer;
  integrator_state_gpu_.gpu_guiding_statistics = (float *)gpu_guiding_statistics_.device_pointer;
  integrator_state_gpu_.gpu_guiding_ready_cells = (uint *)gpu_guiding_ready_cells_.device_pointer;
}

void PathTraceWorkGPU::alloc_work_memory()
{
  alloc_integrator_soa();
  alloc_integrator_queue();
  alloc_integrator_sorting();
  alloc_integrator_path_split();
  alloc_photon_caustics();
  alloc_gpu_guiding();
}

/* Each query has its own emission seed and photon event history. Keep those bindings
 * across spatial tiles while borrowing the resident viewport traversal structures. */
struct PathTraceWorkGPU::QueryWorkspace {
  DeviceScene scene;
  PathTraceWorkGPU work;
  uint64_t stream, revision;

  QueryWorkspace(PathTraceWorkGPU &owner, const QuerySlice &slice)
      : scene(owner.device_),
        work(owner.device_, owner.film_, &scene, owner.cancel_requested_flag_),
        stream(slice.stream),
        revision(slice.revision)
  {
  }
};

PathTraceWorkGPU::~PathTraceWorkGPU() = default;

void PathTraceWorkGPU::release_query_workspace(const uint64_t stream)
{
  if (query_workspace_ && (!stream || query_workspace_->stream == stream)) {
    query_workspace_.reset();
  }
}

bool PathTraceWorkGPU::supports_gpu_query() const
{
  const KernelData &data = device_scene_->data;
  return max_num_paths_ && data.film.pass_combined != PASS_UNUSED && !data.bake.use &&
         data.cam.type != CAMERA_CUSTOM && (!data.integrator.use_caustics || data.integrator.use_photon_caustics) &&
         !data.film.use_deep_output && !(data.kernel_features & (KERNEL_FEATURE_SHADOW_CATCHER |
             KERNEL_FEATURE_VOLUME | KERNEL_FEATURE_OSL_SHADING | KERNEL_FEATURE_OSL_CAMERA));
}

void PathTraceWorkGPU::query_slice(const QuerySlice &s, QuerySliceResult &out)
{
  const KernelData &original = device_scene_->data;
  if (!max_num_paths_) {
    out.error = "Viewport GPU scene is not resident";
    return;
  }
  if (original.cam.width != s.frame_w || original.cam.height != s.frame_h) {
    out.error = "Viewport projection does not match the selection";
    return;
  }
  if (!supports_gpu_query()) {
    out.error = "Scene or projection requires the CPU query";
    return;
  }
  const int pixels = s.w * s.h;
  const int64_t event_budget = int64_t(pixels) * s.count *
                               (original.integrator.max_bounce + original.integrator.transparent_max_bounce + 2) * 8;
  /* A single receiving point can gather many complete photon chains. Keep the pool when
   * retrying smaller regions so splitting an overflowing tile actually provides more room. */
  const int capacity = original.integrator.use_photon_caustics ? (1 << 18) :
                        int(std::clamp(event_budget, int64_t(1024), int64_t(1 << 18)));
  /* Emission borrows one state per photon just as it does in the viewport. Preserve its
   * effective photon count so the query also preserves the density and differential footprint. */
  const bool photons = original.integrator.use_photon_caustics;
  const int photon_count = photons ? min(original.integrator.photon_caustics_count, max_num_paths_) : 0;
  const int slots = max(1024, photon_count);
  const uint64_t features = (original.kernel_features | KERNEL_FEATURE_NODE_AOV) & ~KERNEL_FEATURE_GPU_PATH_GUIDING;
  const size_t soa = estimate_single_state_size(features, original.volume_stack_size, true) * slots;
  out.bytes = soa + capacity * sizeof(KernelQueryRecord) + divide_up(capacity, 128) * sizeof(uint4) +
              size_t(pixels) * original.film.pass_stride * sizeof(float) +
              (size_t(original.max_shaders) + slots) * 32 + (4 << 20);
  if (photons) {
    const size_t depth = max(original.integrator.max_bounce, 1) +
                         original.integrator.transparent_max_bounce;
    /* Each of the two hash tables can round up to four entries per photon. Include
     * optional reconstruction metadata and spatial-cell statistics in the upper bound. */
    const size_t per_photon = sizeof(KernelPhotonCaustic) + depth * sizeof(KernelPhotonCausticsEvent) +
                              sizeof(KernelPhotonReconstructionMetadata) + sizeof(KernelPhotonBeamMetadata) +
                              sizeof(float4) + 11 * sizeof(uint) + 4 * sizeof(uint2);
    out.bytes += size_t(photon_count) * per_photon + sizeof(uint2) + sizeof(uint);
  }
  const bool persistent = photons && s.stream;
  if (persistent && query_workspace_ &&
      (query_workspace_->stream != s.stream || query_workspace_->revision != s.revision ||
       query_workspace_->scene.photon_caustics_revision != device_scene_->photon_caustics_revision))
  {
    query_workspace_.reset();
  }
  const bool reuse_workspace = persistent && query_workspace_;
  /* The path states and photon index are already allocated on subsequent tiles. */
  const size_t tile_bytes = size_t(capacity) * sizeof(KernelQueryRecord) +
                            divide_up(capacity, 128) * sizeof(uint4) + sizeof(uint) +
                            size_t(pixels) * original.film.pass_stride * sizeof(float);
  const size_t required = reuse_workspace ? tile_bytes : out.bytes;
  size_t total = 0, free = 0;
  device_->get_device_memory_info(total, free);
  if (!free || free < required + max(total / 20, size_t(64 << 20))) {
    out.error = "Insufficient free memory for the GPU query";
    return;
  }

  /* The DeviceScene here owns no device geometry. Device array bindings and the top-level
   * traversal handle remain those of the viewport. Only a small private film and SoA are added. */
  unique_ptr<QueryWorkspace> transient;
  auto &workspace = persistent ? query_workspace_ : transient;
  if (!workspace) {
    workspace = make_unique<QueryWorkspace>(*this, s);
  }
  DeviceScene &local = workspace->scene;
  PathTraceWorkGPU &scratch = workspace->work;
  local.photon_caustics_revision = device_scene_->photon_caustics_revision;
  local.data = original;
  local.data.kernel_features = features;
  local.data.integrator.use_gpu_guiding = false;
  local.data.integrator.use_photon_guiding = local.data.integrator.use_photon_area_guiding = false;
  /* One private camera sample estimates regular lighting and gathers its photon contribution.
   * Keep the viewport film layout, but never redirect writes to its separate estimators. */
  local.data.film.photon_segment_stride = local.data.film.photon_compact = 0;
  local.data.film.pass_photon_sample_count = PASS_UNUSED;
  local.data.film.lpe_independent_photon = local.data.film.lpe_photon_count = 0;
  local.data.integrator.seed = hash_uint2(original.integrator.seed, 0x51e527u);
  local.data.film.lpe_state_count = local.data.film.lpe_shadow_count = 0;
  local.data.film.lpe_initial_low = local.data.film.lpe_initial_high = 0;
  local.data.film.pass_sample_count = local.data.film.pass_adaptive_aux_buffer = PASS_UNUSED;
  scratch.diagnostic_slots_ = slots;
  scratch.query_viewport_guide_ = &integrator_state_gpu_;
  device_vector<KernelQueryRecord> records(device_, "LPE query events", MEM_READ_WRITE);
  device_vector<uint4> pages(device_, "LPE query 128-bit pages", MEM_READ_WRITE);
  device_vector<uint> count(device_, "LPE query event count", MEM_READ_WRITE);
  records.alloc(capacity);
  records.zero_to_device();
  pages.alloc(divide_up(capacity, 128));
  pages.zero_to_device();
  count.alloc(1);
  count.zero_to_device();
  BufferParams p = effective_buffer_params_;
  p.full_x = s.x;
  p.full_y = s.y;
  p.width = p.window_width = s.w;
  p.height = p.window_height = s.h;
  p.window_x = p.window_y = 0;
  p.update_offset_stride();
  scratch.buffers_->reset(p);
  scratch.buffers_->zero();
  scratch.effective_buffer_params_ = p;
  if (!reuse_workspace) {
    scratch.alloc_work_memory();
  }
  scratch.integrator_state_gpu_.query_records = (KernelQueryRecord *)records.device_pointer;
  scratch.integrator_state_gpu_.query_count = (uint *)count.device_pointer;
  scratch.integrator_state_gpu_.query_pages = (uint4 *)pages.device_pointer;
  scratch.integrator_state_gpu_.query_capacity = capacity;
  scratch.integrator_state_gpu_.query_total_light = s.total_light;
  {
    struct Restore {
      std::function<void()> restore;
      ~Restore() { restore(); }
    } restore{[&]() {
      scratch.queue_->synchronize();
      scratch.integrator_state_gpu_.query_records = nullptr;
      scratch.integrator_state_gpu_.query_count = nullptr;
      scratch.integrator_state_gpu_.query_pages = nullptr;
      KernelData data = original;
      device_->const_copy_to("data", &data, sizeof(data));
      device_->const_copy_to("integrator_state", &integrator_state_gpu_, sizeof(integrator_state_gpu_));
    }};
    device_->const_copy_to("data", &local.data, sizeof(local.data));
    scratch.init_execution();
    RenderStatistics statistics;
    const uint64_t builds = scratch.photon_caustics_build_count_;
    const uint64_t reuses = scratch.photon_caustics_reuse_count_;
    const int interval = max(1, original.integrator.photon_caustics_update_interval);
    for (int start = s.sample; start < s.sample + s.count;) {
      const int remaining = s.sample + s.count - start;
      const int batch = photons ? min(remaining, interval - start % interval) : remaining;
      scratch.render_samples(statistics, start, batch, 0);
      start += batch;
    }
    out.photon_builds = scratch.photon_caustics_build_count_ - builds;
    out.photon_reuses = scratch.photon_caustics_reuse_count_ - reuses;
    if (!scratch.queue_->synchronize() || is_cancel_requested()) {
      out.error = "GPU contribution query interrupted";
      return;
    }
  }
  count.copy_from_device();
  if (count.data()[0] > uint(capacity)) {
    out.error = "GPU query event buffer full";
    return;
  }
  const uint used = count.data()[0];
  if (used) {
    records.copy_from_device(0, used, 1);
    pages.copy_from_device(0, divide_up(used, 128u), 1);
  }
  if (!scratch.buffers_->copy_from_device()) {
    out.error = "Cannot read GPU contribution query film";
    return;
  }
  for (uint i = 0; i < used; ++i) {
    if (!(reinterpret_cast<const uint *>(pages.data())[i / 32] & (1u << (i % 32)))) {
      out.error = "Incomplete GPU query page";
      return;
    }
    const KernelQueryRecord &rec = records.data()[i];
    /* Black world and zero emission can reach the film even with no light source. */
    if (rec.pixel < 0 || (rec.r == 0 && rec.g == 0 && rec.b == 0) ||
        !std::isfinite(rec.r) || !std::isfinite(rec.g) || !std::isfinite(rec.b)) {
      continue;
    }
    if (rec.pixel >= pixels) {
      out.error = "GPU query pixel outside crop";
      return;
    }
    const bool photon = rec.event == KernelQueryRecord::photon_source;
    QueryPath path{{rec.object, rec.shader, photon ? 0 : rec.event}};
    uint cursor = rec.parent, upper = i + 1;
    while (cursor) {
      if (cursor >= upper) {
        out.error = "Invalid GPU query path";
        return;
      }
      const KernelQueryRecord &v = records.data()[cursor-1];
      path.push_back({v.object, v.shader, v.event});
      upper = cursor;
      cursor = v.parent;
    }
    std::reverse(path.begin(), path.end());
    QueryRGB &sum = out.data.images[{path, photon}][{s.x + rec.pixel % s.w, s.y + rec.pixel / s.w}];
    sum[0] += rec.r;
    sum[1] += rec.g;
    sum[2] += rec.b;
  }
  for (int i = 0; i < pixels; ++i) {
    const float *rgb = scratch.buffers_->buffer.data() + i * p.pass_stride + original.film.pass_combined;
    const QueryPoint point{s.x + i % s.w, s.y + i / s.w};
    out.data.images[{{}, false}][point] = {rgb[0], rgb[1], rgb[2]};
    out.data.images[{{{-1, -1, -1}}, false}][point] = {double(s.count), 0, 0};
    if (photons) {
      out.data.images[{{{-1, -1, -1}}, true}][point] = {double(s.count), 0, 0};
    }
  }
  if (persistent && s.last_slice) {
    release_query_workspace(s.stream);
  }
}

uint PathTraceWorkGPU::vcm_batch_size(const uint requested) const
{
  return std::min(requested, uint(std::max(0, max_num_paths_)));
}

bool PathTraceWorkGPU::vcm_emit_batch(const VCMBatch &batch,
                                      const uint count,
                                      VCMLightPath *paths,
                                      VCMVertex *vertices)
{
  if (device_->info.type != DEVICE_OPTIX || count > uint(max_num_paths_)) {
    device_->set_error("VCM capture requires OptiX and enough path-state slots for one batch.");
    return false;
  }
  device_vector<VCMLightPath> headers(device_, "vcm_path_headers", MEM_READ_WRITE);
  device_vector<VCMVertex> events(device_, "vcm_path_events", MEM_READ_WRITE);
  device_vector<VCMLaunch> launch(device_, "vcm_launch", MEM_READ_ONLY);
  headers.alloc(count);
  events.alloc(size_t(count) * batch.max_depth);
  headers.zero_to_device();
  events.zero_to_device();
  launch.alloc(1);
  launch[0] = {batch,
               reinterpret_cast<VCMLightPath *>(headers.device_pointer),
               reinterpret_cast<VCMVertex *>(events.device_pointer)};
  launch.copy_to_device();
  for (int attempt = 0; attempt < 16; ++attempt) {
    if (is_cancel_requested() || device_->have_error() ||
        !queue_->enqueue(
            DEVICE_KERNEL_VCM_EMIT, int(count), DeviceKernelArguments(&launch.device_pointer)))
    {
      return false;
    }
    queue_->copy_from_device(headers);
    if (!queue_->synchronize()) {
      return false;
    }
    if (std::any_of(headers.data(), headers.data() + count, [](const VCMLightPath &path) {
          return path.status == VCM_PATH_CACHE_MISS;
        })) {
      /* Fully replay the same IDs after loading the requested texture tiles. */
      device_->image_load_requested_gpu(*queue_);
      continue;
    }
    std::copy_n(headers.data(), count, paths);
    if (std::any_of(paths, paths + count, [](const VCMLightPath &path) { return path.count != 0; })) {
      queue_->copy_from_device(events);
      if (!queue_->synchronize()) {
        return false;
      }
      for (uint i = 0; i < count; ++i) {
        /* Validate a device count before using it as a host copy length. */
        if (paths[i].count > batch.max_depth) {
          device_->set_error("VCM emitter returned an invalid vertex count.");
          return false;
        }
        const size_t offset = size_t(i) * batch.max_depth;
        std::copy_n(events.data() + offset, paths[i].count, vertices + offset);
      }
    }
    return !is_cancel_requested();
  }
  device_->set_error("VCM could not resolve texture residency for an emission batch.");
  return false;
}

bool PathTraceWorkGPU::vcm_render_rounds(const uint first, const uint end, const uint stride,
                                         const uint count, const uint depth, const bool area,
                                         const bool native_bsdf, const int planes, const vector<int> &routes, vector<float> &film)
{
  const KernelData &data = device_scene_->data;
  const size_t pixels = size_t(data.cam.width) * size_t(data.cam.height);
  const uint lanes = vcm_batch_size(std::min(count, 16384u));
  const size_t slots = size_t(count) * depth;
  const size_t film_pixels = pixels * (1 + size_t(planes));
  if (!lanes || slots > UINT_MAX || film_pixels > INT_MAX) {
    device_->set_error("OptiX VCM round exceeds the device index range.");
    return false;
  }
  uint bucket_count = 1;
  while (bucket_count < count * 2) {
    bucket_count *= 2;
  }
  const size_t lights = data.integrator.num_lights;
  if (first < stride || vcm_guide_targets_.size() != lights) {
    vcm_guide_targets_.assign(lights, zero_float4());
    vcm_guide_fractions_.assign(lights, 0.0f);
  }

  device_only_memory<VCMLightPath> paths(device_, "VCM light paths");
  device_only_memory<VCMVertex> vertices(device_, "VCM light vertices");
  device_only_memory<VCMConnectionNode> scratch(device_, "VCM camera and light prefixes");
  device_only_memory<uint> heads(device_, "VCM grid heads"), links(device_, "VCM grid links");
  device_only_memory<float> staging(device_, "VCM round film");
  device_vector<float> sum(device_, "VCM film", MEM_READ_WRITE);
  device_vector<uint> status(device_, "VCM round status", MEM_READ_WRITE);
  device_vector<VCMGuideMoments> moments(device_, "VCM emission guide moments", MEM_READ_WRITE);
  device_vector<VCMLaunch> args(device_, "VCM launch", MEM_READ_ONLY);

  paths.alloc_to_device(count);
  vertices.alloc_to_device(slots);
  heads.alloc_to_device(bucket_count);
  links.alloc_to_device(slots);
  if (area && !native_bsdf) {
    scratch.alloc_to_device(size_t(lanes) * 2 * (depth + 1));
  }
  staging.alloc_to_device(film_pixels * 3);
  const bool adaptive = data.film.pass_adaptive_aux_buffer != PASS_UNUSED;
  sum.alloc(3 * (film_pixels + (adaptive ? pixels : 0)));
  queue_->zero_to_device(sum);
  status.alloc(1);
  moments.alloc(std::max(size_t(1), lights));
  args.alloc(1);
  VCMLaunch &launch = args[0];
  launch = {};
  launch.paths = reinterpret_cast<VCMLightPath *>(paths.device_pointer);
  launch.vertices = reinterpret_cast<VCMVertex *>(vertices.device_pointer);
  launch.heads = reinterpret_cast<uint *>(heads.device_pointer);
  launch.links = reinterpret_cast<uint *>(links.device_pointer);
  launch.bucket_mask = bucket_count - 1;
  device_vector<int> source_routes(device_, "VCM pass routes", MEM_READ_ONLY);
  if (planes) {
    source_routes.alloc(routes.size());
    std::copy(routes.begin(), routes.end(), source_routes.data());
    source_routes.copy_to_device();
    launch.source_routes = reinterpret_cast<const int *>(source_routes.device_pointer);
  }
  launch.scratch = reinterpret_cast<VCMConnectionNode *>(scratch.device_pointer);
  launch.film = reinterpret_cast<float *>(staging.device_pointer);

  if (device_->have_error()) {
    return false;
  }
  auto enqueue = [&](int n) {
    queue_->copy_to_device(args);
    if (device_->have_error() || !queue_->enqueue(
            DEVICE_KERNEL_VCM_EMIT, n, DeviceKernelArguments(&args.device_pointer))) {
      return false;
    }
    /* On mapped memory the kernel reads the host allocation itself. Do not
     * modify the next launch descriptor while that dispatch is using it. */
    return !(args.is_shared(device_) && args.host_pointer == args.shared_pointer) ||
           queue_->synchronize();
  };
  for (uint round = first; round < end; round += stride) {
    launch.batch = {0, round, uint(data.integrator.seed), depth};
    launch.batch.native_bsdf = native_bsdf;
    for (size_t j = 0; j < lights; ++j) {
      const int i = int((round + j) % lights);
      if (vcm_guide_fractions_[i] > 0) {
        launch.batch.guide_light = i;
        launch.batch.guide_fraction = vcm_guide_fractions_[i];
        launch.batch.guide_target = make_float3(vcm_guide_targets_[i]);
        launch.batch.guide_sigma = vcm_guide_targets_[i].w;
        break;
      }
    }
    launch.radius = 0.02 * std::pow(round + 1.0, -1.0 / 6.0);
    launch.eta = count * M_PI * launch.radius * launch.radius;
    bool accepted = false;
    for (int attempt = 0; attempt < 16 && !accepted; ++attempt) {
      queue_->zero_to_device(heads);
      queue_->zero_to_device(staging);
      queue_->zero_to_device(status);
      queue_->zero_to_device(moments);
      launch.status = reinterpret_cast<uint *>(status.device_pointer);
      launch.moments = reinterpret_cast<VCMGuideMoments *>(moments.device_pointer);
      launch.accumulate = nullptr;
      for (int phase = 0; phase < 2; ++phase) {
        launch.connect = phase == 1;
        for (uint offset = 0; offset < count; offset += lanes) {
          if (is_cancel_requested()) {
            queue_->synchronize();
            return false;
          }
          launch.batch.first = offset;
          if (!enqueue(int(std::min(lanes, count - offset)))) {
            return false;
          }
        }
        /* A bad emission batch must never be used as a completed photon table. */
        queue_->copy_from_device(status);
        if (!queue_->synchronize()) {
          return false;
        }
        if (status[0]) {
          break;
        }
      }
      if (status[0] & ~1u) {
        device_->set_error(status[0] & 4u ? "OptiX VCM photon exceeds spatial grid precision." :
                                       "OptiX VCM encountered an unsupported emitter or surface.");
        return false;
      }
      if (status[0]) {
        device_->image_load_requested_gpu(*queue_);
        continue;
      }
      /* Commit only complete rounds; texture page faults replay the same IDs and RNG. */
      launch.accumulate = reinterpret_cast<float *>(sum.device_pointer);
      launch.adaptive_sum = adaptive && vcm_adaptive_half(round, data.integrator.seed) ? launch.accumulate + 3 * film_pixels : nullptr;
      if (!enqueue(int(film_pixels))) {
        return false;
      }
      if ((!area || native_bsdf) && lights) {
        queue_->copy_from_device(moments);
      }
      if (!queue_->synchronize()) {
        return false;
      }
      for (size_t i = 0; (!area || native_bsdf) && i < lights; ++i) {
        const VCMGuideMoments &m = moments[i];
        if (m.squared_weight <= 0) {
          continue;
        }
        const double ess = m.weight * m.weight / m.squared_weight;
        if (ess < 16) {
          continue;
        }
        const float3 mean = make_float3(float(m.first[0]/m.weight), float(m.first[1]/m.weight),
                                        float(m.first[2]/m.weight));
        const double d2 = len_squared(mean);
        const float sigma = float(std::sqrt(std::max((m.second/m.weight - d2)/3.0, d2*1e-4)));
        if (isfinite_safe(mean) && std::isfinite(sigma) && sigma > 0) {
          vcm_guide_targets_[i] = make_float4(mean, sigma);
          vcm_guide_fractions_[i] = float(0.5 * ess / (ess + 64.0));
        }
      }
      accepted = true;
    }
    if (!accepted) {
      device_->set_error("OptiX VCM could not resolve texture residency for this round.");
      return false;
    }
  }
  queue_->copy_from_device(sum);
  if (!queue_->synchronize()) {
    return false;
  }
  film.assign(sum.data(), sum.data() + sum.size());
  return !is_cancel_requested();
}

void PathTraceWorkGPU::init_execution()
{
  queue_->init_execution();

  /* Separate physical devices have separate constants. The sequential test restores
   * these, together with integrator_state, when switching logical replicas. */
  if (photon_replica_ && device_scene_->data.integrator.use_photon_caustics) {
    KernelData data = device_scene_->data;
    data.integrator.seed = photon_sampling_seed();
    device_->const_copy_to("data", &data, sizeof(data));
  }

  /* Copy to device side struct in constant memory. */
  device_->const_copy_to(
      "integrator_state", &integrator_state_gpu_, sizeof(integrator_state_gpu_));
}

bool PathTraceWorkGPU::update_queue_counter_and_cache()
{
  /* Copy stats from the device. */
  queue_->copy_from_device(integrator_queue_counter_);

  if (!queue_->synchronize()) {
    return false;
  }

  /* Update image cache if needed. */
  /* TODO: If the number of kernels with cache misses is small compared to the total
   * number of queued kernels, we could try asynchronously updating the image cache
   * while continuing to work on the majority of states? */
  IntegratorQueueCounter *queue_counter = integrator_queue_counter_.data();
  if (queue_counter->cache_miss) {
    LOG_DEBUG << "Image cache miss in GPU kernel, updating to load requested tiles";
    device_->image_load_requested_gpu(*queue_);
    queue_counter->cache_miss = 0;
    queue_->copy_to_device(integrator_queue_counter_);
  }

  return true;
}

void PathTraceWorkGPU::render_samples(RenderStatistics &statistics,
                                      const int start_sample,
                                      const int samples_num,
                                      const int sample_offset)
{
  if (!gpu_guiding_cdf_.device_pointer || photon_only_) {
    if (device_scene_->data.film.deep_physical) {
      for (int i = 0; i < samples_num && !is_cancel_requested() && !device_->have_error(); ++i)
        render_samples_batch(statistics, start_sample + i, 1, sample_offset);
    }
    else render_samples_batch(statistics, start_sample, samples_num, sample_offset);
    return;
  }
  if (gpu_guiding_reset_) {
    queue_->zero_to_device(gpu_guiding_cdf_);
    queue_->zero_to_device(gpu_guiding_statistics_);
    gpu_guiding_reset_ = false;
    gpu_guiding_frozen_ = false;
    gpu_guiding_next_publish_ = 32;
    gpu_guiding_update_seconds_ = 0.0;
    LOG_INFO << "GPU surface guiding: " << GPU_GUIDE_CELLS << " cells, " << GPU_GUIDE_BINS
             << " direction bins, " << (gpu_guiding_cdf_.memory_size() +
                                        gpu_guiding_statistics_.memory_size())
             << " field bytes; per-path state allocated separately";
  }
  const int training_end = device_scene_->data.integrator.gpu_guiding_training_samples;
  for (int done = 0; done < samples_num && !is_cancel_requested() && !device_->have_error();) {
    const int begin = start_sample + done;
    /* Subsets start with an empty local guide. Keep the global sample index for rendering
     * and use the index within the subset for training and publishing. */
    const int local_begin = begin - sample_offset;
    const bool training = !gpu_guiding_frozen_ && local_begin < training_end;
    int count = training ? min(samples_num - done, min(32, training_end - local_begin)) :
                           samples_num - done;
    if (device_scene_->data.film.deep_physical) count = min(count, 1);
    integrator_state_gpu_.gpu_guiding_training = training;
    render_samples_batch(statistics, begin, count, sample_offset);
    if (training && (local_begin + count >= gpu_guiding_next_publish_ ||
                     local_begin + count >= training_end) &&
        !device_->have_error() && !is_cancel_requested()) {
      publish_gpu_guiding(begin + count);
      gpu_guiding_next_publish_ = ((local_begin + count) / 32 + 1) * 32;
      gpu_guiding_frozen_ = local_begin + count >= training_end;
    }
    done += count;
  }
}

void PathTraceWorkGPU::publish_gpu_guiding(const int completed_sample)
{
  scoped_timer timer;
  queue_->zero_to_device(gpu_guiding_ready_cells_);
  DeviceKernelArguments args;
  queue_->enqueue(DEVICE_KERNEL_GPU_GUIDING_PUBLISH, GPU_GUIDE_CELLS, args);
  queue_->copy_from_device(gpu_guiding_ready_cells_);
  if (!queue_->synchronize()) {
    return;
  }
  const double seconds = timer.get_time();
  gpu_guiding_update_seconds_ += seconds;
  LOG_INFO << "GPU surface guiding: sample " << completed_sample << ", ready cells "
           << gpu_guiding_ready_cells_.data()[0] << ", publication seconds " << seconds
           << ", accumulated publication seconds " << gpu_guiding_update_seconds_;
}

void PathTraceWorkGPU::render_samples_batch(RenderStatistics &statistics,
                                            const int start_sample,
                                            const int samples_num,
                                            const int sample_offset)
{
  /* Limit number of states for the tile and rely on a greedy scheduling of tiles. This allows to
   * add more work (because tiles are smaller, so there is higher chance that more paths will
   * become busy after adding new tiles). This is especially important for the shadow catcher which
   * schedules work in halves of available number of paths. */
  work_tile_scheduler_.set_max_num_path_states(max_num_paths_ / 8);
  work_tile_scheduler_.set_accelerated_rt(
      (device_->get_bvh_layout_mask(device_scene_->data.kernel_features) & BVH_LAYOUT_OPTIX) != 0);
  work_tile_scheduler_.reset(effective_buffer_params_,
                             start_sample,
                             samples_num,
                             sample_offset,
                             device_scene_->data.integrator.scrambling_distance);

  /* Build the current emission round before camera paths occupy the borrowed SoA slots, or reuse
   * that round. Reset queues and slot markers before entering the camera scheduler. */
  if (gpu_guiding_cdf_.device_pointer) {
    integrator_state_gpu_.gpu_guiding_active = false;
    device_->const_copy_to("integrator_state", &integrator_state_gpu_, sizeof(integrator_state_gpu_));
  }
  if ((!device_scene_->data.film.photon_segment_stride || photon_only_) &&
      !enqueue_photon_caustics(start_sample))
  {
    statistics.occupancy = 0.0f;
    return;
  }
  if (gpu_guiding_cdf_.device_pointer) {
    integrator_state_gpu_.gpu_guiding_active = !photon_only_;
    device_->const_copy_to("integrator_state", &integrator_state_gpu_, sizeof(integrator_state_gpu_));
  }
  enqueue_reset();

  int num_iterations = 0;
  uint64_t num_busy_accum = 0;

  /* TODO: set a hard limit in case of undetected kernel failures? */
  while (true) {
    /* Enqueue work from the scheduler, on start or when there are not enough
     * paths to keep the device occupied. */
    bool finished;
    if (enqueue_work_tiles(finished)) {
      if (!update_queue_counter_and_cache()) {
        break; /* Stop on error. */
      }
    }

    if (is_cancel_requested()) {
      break;
    }

    /* Stop if no more work remaining. */
    if (finished) {
      break;
    }

    /* Enqueue on of the path iteration kernels. */
    if (enqueue_path_iteration()) {
      if (!update_queue_counter_and_cache()) {
        break; /* Stop on error. */
      }
    }

    if (is_cancel_requested()) {
      break;
    }

    num_busy_accum += num_active_main_paths_paths();
    ++num_iterations;
  }

  if (device_scene_->data.film.deep_physical && !device_->have_error() && !is_cancel_requested()) {
    const auto &p = effective_buffer_params_;
    DeviceKernelArguments args(&p.full_x, &p.full_y, &p.width, &p.height, &p.offset, &p.stride);
    queue_->enqueue(DEVICE_KERNEL_DEEP_PHYSICAL_FINISH, p.width * p.height, args);
    queue_->synchronize();
  }
  if (num_iterations) {
    statistics.occupancy = float(num_busy_accum) / num_iterations / max_num_paths_;
  }
  else {
    statistics.occupancy = 0.0f;
  }
}

DeviceKernel PathTraceWorkGPU::get_most_queued_kernel() const
{
  const IntegratorQueueCounter *queue_counter = integrator_queue_counter_.data();

  int max_num_queued = 0;
  DeviceKernel kernel = DEVICE_KERNEL_NUM;

  for (int i = 0; i < DEVICE_GPU_KERNEL_INTEGRATOR_NUM; i++) {
    /* SHADOW_PATH_MNEE_PENDING is a sentinel marker on shadow slots holding an MNEE precompute
     * payload; there is no kernel to dispatch for it. The slot transitions to a real shadow
     * kernel (or terminates) when integrator_shade_surface runs on the main path. */
    if (i == DEVICE_KERNEL_INTEGRATOR_SHADOW_PATH_MNEE_PENDING) {
      continue;
    }
    if (queue_counter->num_queued[i] > max_num_queued) {
      kernel = (DeviceKernel)i;
      max_num_queued = queue_counter->num_queued[i];
    }
  }

  return kernel;
}

bool PathTraceWorkGPU::enqueue_photon_caustics(const int start_sample)
{
  if (is_cancel_requested() || device_->have_error()) {
    return false;
  }
  if (!device_scene_->data.integrator.use_photon_caustics ||
      integrator_state_gpu_.photon_caustics_capacity == 0)
  {
    photon_caustics_valid_ = false;
    return true;
  }

  const KernelIntegrator &integrator = device_scene_->data.integrator;
  const char *backend = device_->info.type == DEVICE_OPTIX ? "OptiX" : "CUDA";
  const int update_interval = max(integrator.photon_caustics_update_interval, 1);
  const uint round = uint(start_sample / update_interval);
  if (photon_caustics_valid_ && !photon_guide_reset_ &&
      photon_caustics_revision_ == device_scene_->photon_caustics_revision &&
      photon_caustics_seed_ == photon_sampling_seed() &&
      photon_caustics_requested_count_ == integrator.photon_caustics_count &&
      photon_caustics_round_ == round && photon_caustics_update_interval_ == update_interval &&
      photon_caustics_capacity_ == integrator_state_gpu_.photon_caustics_capacity &&
      photon_caustics_radius_ == integrator_state_gpu_.photon_caustics_radius &&
      photon_caustics_event_stride_ == integrator_state_gpu_.photon_caustics_event_stride)
  {
    ++photon_caustics_reuse_count_;
    if ((photon_caustics_reuse_count_ & (photon_caustics_reuse_count_ - 1)) == 0) {
      LOG_DEBUG << backend << " photon caustics cache: builds " << photon_caustics_build_count_
                << ", reused batches " << photon_caustics_reuse_count_ << ", build seconds "
                << photon_caustics_build_seconds_;
    }
    return true;
  }

  const scoped_timer guide_timer;
  const bool guide_restart = photon_guide_reset_ ||
                             photon_caustics_revision_ !=
                                 device_scene_->photon_caustics_revision ||
                             photon_caustics_seed_ != photon_sampling_seed() ||
                             round < photon_caustics_round_;
  if (integrator.use_photon_guiding) {
    if (guide_restart) {
      photon_guiding_initialize(photon_guide_.data(), integrator.num_lights);
      photon_guide_.copy_to_device();
      photon_guide_statistics_.zero_to_device();
      integrator_state_gpu_.photon_guide_fraction = 0.0f;
      photon_guide_epoch_ = round / 32;
    }
    else if (round / 32 != photon_guide_epoch_) {
      queue_->copy_from_device(photon_guide_statistics_);
      if (!queue_->synchronize()) {
        return false;
      }
      if (photon_guiding_update(
              photon_guide_.data(), photon_guide_statistics_.data(), integrator.num_lights))
      {
        integrator_state_gpu_.photon_guide_fraction = .5f;
        photon_guide_.copy_to_device();
        LOG_DEBUG << backend << " photon guiding: fitted epoch " << round / 32;
      }
      photon_guide_statistics_.copy_to_device();
      photon_guide_epoch_ = round / 32;
    }
    integrator_state_gpu_.photon_guide = reinterpret_cast<KernelPhotonGuide *>(
        photon_guide_.device_pointer);
    integrator_state_gpu_.photon_guide_statistics =
        reinterpret_cast<KernelPhotonGuideStatistics *>(photon_guide_statistics_.device_pointer);
  }
  else {
    integrator_state_gpu_.photon_guide = nullptr;
    integrator_state_gpu_.photon_guide_statistics = nullptr;
    integrator_state_gpu_.photon_guide_fraction = 0.0f;
  }
  if (integrator.use_photon_area_guiding) {
    if (guide_restart) {
      photon_area_guiding_initialize(photon_area_guide_cdf_.data(), integrator.num_lights);
      photon_area_guide_cdf_.copy_to_device();
      photon_area_guide_statistics_.zero_to_device();
      photon_area_guide_epoch_ = round / 32;
    }
    else if (round / 32 != photon_area_guide_epoch_) {
      queue_->copy_from_device(photon_area_guide_statistics_);
      if (!queue_->synchronize())
        return false;
      if (photon_area_guiding_update(photon_area_guide_cdf_.data(),
                                     photon_area_guide_statistics_.data(),
                                     integrator.num_lights))
      {
        photon_area_guide_cdf_.copy_to_device();
        LOG_DEBUG << backend << " photon area guiding: fitted epoch " << round / 32;
      }
      photon_area_guide_statistics_.copy_to_device();
      photon_area_guide_epoch_ = round / 32;
    }
  }
  integrator_state_gpu_.photon_area_guide_cdf = reinterpret_cast<float *>(
      photon_area_guide_cdf_.device_pointer);
  integrator_state_gpu_.photon_area_guide_statistics = reinterpret_cast<float4 *>(
      photon_area_guide_statistics_.device_pointer);
  integrator_state_gpu_.photon_area_guide_round_statistics = reinterpret_cast<float2 *>(
      photon_area_guide_round_statistics_.device_pointer);
  const double guide_seconds = guide_timer.get_time();
  if (query_viewport_guide_) {
    /* Borrow only the published distributions. Camera samples, photon maps and training writes
     * belong to the individual session; neither job mutates the other's sampling history. */
    integrator_state_gpu_.photon_guide = query_viewport_guide_->photon_guide;
    integrator_state_gpu_.photon_guide_fraction = query_viewport_guide_->photon_guide_fraction;
    integrator_state_gpu_.photon_area_guide_cdf = query_viewport_guide_->photon_area_guide_cdf;
  }
  photon_guide_reset_ = false;
  photon_caustics_valid_ = false;
  const scoped_timer build_timer;
  /* The absolute sample range determines the round, independent of scheduler batch boundaries.
   * Power remains normalized per map; the camera film averages the independent map estimates. */
  integrator_state_gpu_.photon_caustics_iteration = hash_uint2(round, photon_sampling_seed());
  integrator_state_gpu_.photon_caustics_wavelength_random =
      getenv("CYCLES_PHOTON_WAVELENGTH_LEGACY") ?
          hash_uint2_to_float(integrator_state_gpu_.photon_caustics_iteration, 0xa54ff53au) :
          photon_caustics_wavelength_random(round, photon_sampling_seed());
  integrator_state_gpu_.photon_caustics_wavelength_groups = photon_wavelength_groups_from_environment();
  device_->const_copy_to(
      "integrator_state", &integrator_state_gpu_, sizeof(integrator_state_gpu_));
  /* Texture requests can terminate light paths before they deposit a photon. Never publish such
   * a partial map: load the missing tiles and replay every emission with the same random seed. */
  constexpr int max_attempts = 16;
  for (int attempt = 0; attempt < max_attempts; attempt++) {
    if (is_cancel_requested()) {
      return false;
    }
    queue_->zero_to_device(integrator_queue_counter_);
    /* Only photons deposited in this attempt can reach the guiding feedback pass.
     * Their first-hit records are overwritten by emission; unused slots need no clear. */
    queue_->zero_to_device(photon_caustics_hash_);
    if (integrator_state_gpu_.photon_caustics_query_hash) {
      queue_->zero_to_device(photon_caustics_query_hash_);
    }
    queue_->zero_to_device(photon_caustics_count_);
    if (integrator.use_photon_area_guiding) {
      queue_->zero_to_device(photon_area_guide_round_statistics_);
    }

    /* No positional kernel arguments: both GPU entry points read iteration/buffers from
     * integrator_state. The CUDA entry also checks the capacity for padded launch threads. */
    if (!queue_->enqueue(DEVICE_KERNEL_INTEGRATOR_PHOTON_CAUSTICS_EMIT,
                         int(integrator_state_gpu_.photon_caustics_capacity),
                         DeviceKernelArguments()))
    {
      return false;
    }
    queue_->copy_from_device(integrator_queue_counter_);
    queue_->copy_from_device(photon_caustics_count_);
    if (!queue_->synchronize()) {
      return false;
    }
    if (is_cancel_requested()) {
      return false;
    }

    if (integrator_queue_counter_.data()->cache_miss == 0) {
      const double emission_seconds = build_timer.get_time();
      if (integrator.use_photon_area_guiding &&
          !queue_->enqueue(DEVICE_KERNEL_INTEGRATOR_PHOTON_AREA_GUIDE_COMMIT,
                           int(photon_area_guide_cdf_.size()),
                           DeviceKernelArguments()))
      {
        return false;
      }
      if (integrator_state_gpu_.photon_cell_statistics && photon_caustics_count_.data()[0] != 0) {
        queue_->zero_to_device(photon_cell_statistics_);
        if (!queue_->enqueue(DEVICE_KERNEL_INTEGRATOR_PHOTON_CAUSTICS_CELLS,
                             int(photon_caustics_count_.data()[0]),
                             DeviceKernelArguments()))
        {
          return false;
        }
      }
      if (integrator.use_photon_adaptive_radius && photon_caustics_count_.data()[0] != 0 &&
          (!queue_->enqueue(DEVICE_KERNEL_INTEGRATOR_PHOTON_CAUSTICS_RECONSTRUCT,
                            int(photon_caustics_count_.data()[0]),
                            DeviceKernelArguments()) ||
           !queue_->synchronize()))
      {
        return false;
      }
      photon_caustics_revision_ = device_scene_->photon_caustics_revision;
      photon_caustics_seed_ = photon_sampling_seed();
      photon_caustics_requested_count_ = integrator.photon_caustics_count;
      photon_caustics_round_ = round;
      photon_caustics_update_interval_ = update_interval;
      photon_caustics_capacity_ = integrator_state_gpu_.photon_caustics_capacity;
      photon_caustics_radius_ = integrator_state_gpu_.photon_caustics_radius;
      photon_caustics_event_stride_ = integrator_state_gpu_.photon_caustics_event_stride;
      photon_caustics_valid_ = true;
      ++photon_caustics_build_count_;
      const double build_seconds = build_timer.get_time();
      photon_caustics_build_seconds_ += build_seconds;
      LOG_DEBUG << backend << " photon caustics: emitted " << photon_caustics_capacity_
                << ", stored " << photon_caustics_count_.data()[0] << ", radius "
                << photon_caustics_radius_ << ", build " << photon_caustics_build_count_
                << ", scene revision " << photon_caustics_revision_ << ", build seconds "
                << build_seconds << ", emission seconds " << emission_seconds
                << ", reconstruction seconds " << build_seconds - emission_seconds
                << ", guiding update seconds " << guide_seconds << ", round " << round
                << ", sample interval " << update_interval << ", seed "
                << integrator_state_gpu_.photon_caustics_iteration << ", attempts " << attempt + 1
                << ", LPE event stride " << photon_caustics_event_stride_ << ", LPE cache bytes "
                << photon_caustics_events_.memory_size() +
                       photon_caustics_path_indices_.memory_size()
                << ", query index bytes " << photon_caustics_query_hash_.memory_size() +
                                               photon_caustics_query_next_.memory_size();
      return true;
    }
    if (attempt + 1 < max_attempts) {
      LOG_DEBUG << "Image cache miss in photon caustics prepass, loading tiles and replaying";
      device_->image_load_requested_gpu(*queue_);
      if (device_->have_error()) {
        return false;
      }
    }
  }

  device_->set_error(
      "Photon caustics image cache did not converge after 16 attempts. The photon map was not "
      "used.");
  return false;
}

void PathTraceWorkGPU::enqueue_reset()
{
  const DeviceKernelArguments args(&max_num_paths_);

  queue_->enqueue(DEVICE_KERNEL_INTEGRATOR_RESET, max_num_paths_, args);
  queue_->zero_to_device(integrator_queue_counter_);
  if (integrator_shader_sort_counter_.size() != 0) {
    queue_->zero_to_device(integrator_shader_sort_counter_);
  }
  if (device_scene_->data.kernel_features & KERNEL_FEATURE_NODE_RAYTRACE &&
      integrator_shader_raytrace_sort_counter_.size() != 0)
  {
    queue_->zero_to_device(integrator_shader_raytrace_sort_counter_);
  }

  /* Tiles enqueue need to know number of active paths, which is based on this counter. Zero the
   * counter on the host side because `zero_to_device()` is not doing it. */
  if (integrator_queue_counter_.host_pointer) {
    memset(integrator_queue_counter_.data(), 0, integrator_queue_counter_.memory_size());
  }

  /* No active paths, reset to zero to avoid unnecessary work and exceeding
   * new max number of paths if it changed. */
  max_active_main_path_index_ = 0;
}

bool PathTraceWorkGPU::enqueue_path_iteration()
{
  /* Find kernel to execute, with max number of queued paths. */
  const IntegratorQueueCounter *queue_counter = integrator_queue_counter_.data();

  int num_active_paths = 0;
  for (int i = 0; i < DEVICE_GPU_KERNEL_INTEGRATOR_NUM; i++) {
    num_active_paths += queue_counter->num_queued[i];
  }

  if (num_active_paths == 0) {
    return false;
  }

  /* Find kernel to execute, with max number of queued paths. */
  const DeviceKernel kernel = get_most_queued_kernel();
  if (kernel == DEVICE_KERNEL_NUM) {
    return false;
  }

  /* For kernels that add shadow paths, check if there is enough space available.
   * If not, schedule shadow kernels first to clear out the shadow paths. */
  int num_paths_limit = INT_MAX;

  if (kernel_creates_shadow_paths(kernel)) {
    compact_shadow_paths();

    /* Number of shadow paths that may be created for every scheduled path. */
    const int shadow_paths_per_path = kernel_creates_ao_paths(kernel) ? 2 : 1;

    int available_shadow_paths = max_num_paths_ - integrator_next_shadow_path_index_.data()[0];

    if (available_shadow_paths < queue_counter->num_queued[kernel] * shadow_paths_per_path) {
      if (queue_counter->num_queued[DEVICE_KERNEL_INTEGRATOR_SHADE_LIGHT_NEE]) {
        enqueue_path_iteration(DEVICE_KERNEL_INTEGRATOR_SHADE_LIGHT_NEE);
        return true;
      }
      if (queue_counter->num_queued[DEVICE_KERNEL_INTEGRATOR_INTERSECT_SHADOW]) {
        enqueue_path_iteration(DEVICE_KERNEL_INTEGRATOR_INTERSECT_SHADOW);
        return true;
      }
      if (queue_counter->num_queued[DEVICE_KERNEL_INTEGRATOR_SHADE_SHADOW]) {
        enqueue_path_iteration(DEVICE_KERNEL_INTEGRATOR_SHADE_SHADOW);
        return true;
      }

      /* Rare corner case where there are not even enough available shadow paths
       * to run the kernel with 1 path, in this case we must force compaction. */
      if (available_shadow_paths < shadow_paths_per_path) {
        compact_shadow_paths(true);
        available_shadow_paths = max_num_paths_ - integrator_next_shadow_path_index_.data()[0];
      }
    }

    /* Limit number of scheduled paths to the available shadow path space. */
    num_paths_limit = available_shadow_paths / shadow_paths_per_path;

    if (kernel == DEVICE_KERNEL_INTEGRATOR_INTERSECT_MNEE) {
      /* MNEE should not fill in more than half the available shadow paths,
       * so there is room to run integrator_shade_surface, which is the only
       * kernel that can release MNEE shadow paths. */
      const int num_mnee_paths =
          queue_counter->num_queued[DEVICE_KERNEL_INTEGRATOR_SHADOW_PATH_MNEE_PENDING];
      num_paths_limit = min(num_paths_limit, max(max_num_paths_ / 2 - num_mnee_paths, 1));
    }
  }

  /* Schedule kernel with maximum number of queued items. */
  enqueue_path_iteration(kernel, num_paths_limit);

  /* Update next shadow path index for kernels that can add shadow paths. */
  if (kernel_creates_shadow_paths(kernel)) {
    queue_->copy_from_device(integrator_next_shadow_path_index_);
  }

  return true;
}

void PathTraceWorkGPU::enqueue_path_iteration(DeviceKernel kernel, const int num_paths_limit)
{
  const device_ptr d_trace_buffer = trace_buffer_device_pointer();
  device_ptr d_path_index = 0;

  /* Create array of path indices for which this kernel is queued to be executed. */
  int work_size = kernel_max_active_main_path_index(kernel);

  IntegratorQueueCounter *queue_counter = integrator_queue_counter_.data();
  const int num_queued = queue_counter->num_queued[kernel];

  if (kernel_uses_sorting(kernel)) {
    /* Compute array of active paths, sorted by shader. */
    work_size = num_queued;
    d_path_index = queued_paths_.device_pointer;

    compute_sorted_queued_paths(kernel, num_paths_limit);
  }
  else if (num_queued < work_size) {
    work_size = num_queued;
    d_path_index = queued_paths_.device_pointer;

    if (kernel_is_shadow_path(kernel)) {
      /* Compute array of active shadow paths for specific kernel. */
      compute_queued_paths(DEVICE_KERNEL_INTEGRATOR_QUEUED_SHADOW_PATHS_ARRAY, kernel);
    }
    else {
      /* Compute array of active paths for specific kernel. */
      compute_queued_paths(DEVICE_KERNEL_INTEGRATOR_QUEUED_PATHS_ARRAY, kernel);
    }
  }

  work_size = min(work_size, num_paths_limit);

  DCHECK_LE(work_size, max_num_paths_);

  switch (kernel) {
    case DEVICE_KERNEL_INTEGRATOR_INIT_FROM_CAMERA: {
      const int num_path_indices = work_size;
      device_ptr null_tiles = 0;
      int zero = 0;
      const DeviceKernelArguments args(
          &null_tiles, &zero, &d_trace_buffer, &zero, &d_path_index, &num_path_indices);

      queue_->enqueue(kernel, work_size, args);
      break;
    }
    case DEVICE_KERNEL_INTEGRATOR_INTERSECT_CLOSEST: {
      /* Closest ray intersection kernels with integrator state and render buffer. */
      const DeviceKernelArguments args(&d_path_index, &d_trace_buffer, &work_size);

      queue_->enqueue(kernel, work_size, args);
      break;
    }

    case DEVICE_KERNEL_INTEGRATOR_INTERSECT_SHADOW:
    case DEVICE_KERNEL_INTEGRATOR_INTERSECT_SUBSURFACE:
    case DEVICE_KERNEL_INTEGRATOR_INTERSECT_VOLUME_STACK:
    case DEVICE_KERNEL_INTEGRATOR_INTERSECT_DEDICATED_LIGHT:
    case DEVICE_KERNEL_INTEGRATOR_INTERSECT_MNEE: {
      /* Ray intersection kernels with integrator state. */
      const DeviceKernelArguments args(&d_path_index, &work_size);

      queue_->enqueue(kernel, work_size, args);
      break;
    }
    case DEVICE_KERNEL_INTEGRATOR_SHADE_BACKGROUND:
    case DEVICE_KERNEL_INTEGRATOR_SHADE_LIGHT_NEE:
    case DEVICE_KERNEL_INTEGRATOR_SHADE_LIGHT_FORWARD:
    case DEVICE_KERNEL_INTEGRATOR_SHADE_SHADOW:
    case DEVICE_KERNEL_INTEGRATOR_SHADE_SURFACE:
    case DEVICE_KERNEL_INTEGRATOR_SHADE_SURFACE_RAYTRACE:
    case DEVICE_KERNEL_INTEGRATOR_SHADE_VOLUME:
    case DEVICE_KERNEL_INTEGRATOR_SHADE_VOLUME_RAY_MARCHING:
    case DEVICE_KERNEL_INTEGRATOR_SHADE_DEDICATED_LIGHT: {
      /* Shading kernels with integrator state and render buffer. */
      const DeviceKernelArguments args(&d_path_index, &d_trace_buffer, &work_size);

      queue_->enqueue(kernel, work_size, args);
      break;
    }
    default:
      LOG_FATAL << "Unhandled kernel " << device_kernel_as_string(kernel)
                << " used for path iteration, should never happen.";
      break;
  }
}

void PathTraceWorkGPU::compute_sorted_queued_paths(DeviceKernel queued_kernel,
                                                   const int num_paths_limit)
{
  int d_queued_kernel = queued_kernel;

  /* Launch kernel to fill the active paths arrays. */
  if (num_sort_partitions_ > 1 && queue_->supports_local_atomic_sort()) {
    const int work_size = kernel_max_active_main_path_index(queued_kernel);
    device_ptr d_queued_paths = queued_paths_.device_pointer;

    int partition_size = (int)integrator_state_gpu_.sort_partition_divisor;

    const DeviceKernelArguments args(
        &work_size, &partition_size, &num_paths_limit, &d_queued_paths, &d_queued_kernel);

    queue_->enqueue(DEVICE_KERNEL_INTEGRATOR_SORT_BUCKET_PASS,
                    GPU_PARALLEL_SORT_BLOCK_SIZE * num_sort_partitions_,
                    args);
    queue_->enqueue(DEVICE_KERNEL_INTEGRATOR_SORT_WRITE_PASS,
                    GPU_PARALLEL_SORT_BLOCK_SIZE * num_sort_partitions_,
                    args);
    return;
  }

  device_ptr d_counter = (device_ptr)integrator_state_gpu_.sort_key_counter[d_queued_kernel];
  device_ptr d_prefix_sum = integrator_shader_sort_prefix_sum_.device_pointer;
  assert(d_counter != 0 && d_prefix_sum != 0);

  /* Compute prefix sum of number of active paths with each shader. */
  {
    const int work_size = 1;
    int sort_buckets = device_scene_->data.max_shaders * num_sort_partitions_;

    const DeviceKernelArguments args(&d_counter, &d_prefix_sum, &sort_buckets);

    queue_->enqueue(DEVICE_KERNEL_PREFIX_SUM, work_size, args);
  }

  queue_->zero_to_device(num_queued_paths_);

  /* Launch kernel to fill the active paths arrays. */
  {
    /* TODO: this could be smaller for terminated paths based on amount of work we want
     * to schedule, and also based on num_paths_limit.
     *
     * Also, when the number paths is limited it may be better to prefer paths from the
     * end of the array since compaction would need to do less work. */
    const int work_size = kernel_max_active_main_path_index(queued_kernel);

    device_ptr d_queued_paths = queued_paths_.device_pointer;
    device_ptr d_num_queued_paths = num_queued_paths_.device_pointer;

    const DeviceKernelArguments args(&work_size,
                                     &num_paths_limit,
                                     &d_queued_paths,
                                     &d_num_queued_paths,
                                     &d_counter,
                                     &d_prefix_sum,
                                     &d_queued_kernel);

    queue_->enqueue(DEVICE_KERNEL_INTEGRATOR_SORTED_PATHS_ARRAY, work_size, args);
  }
}

void PathTraceWorkGPU::compute_queued_paths(DeviceKernel kernel, DeviceKernel queued_kernel)
{
  int d_queued_kernel = queued_kernel;

  /* Launch kernel to fill the active paths arrays. */
  const int work_size = kernel_max_active_main_path_index(queued_kernel);
  device_ptr d_queued_paths = queued_paths_.device_pointer;
  device_ptr d_num_queued_paths = num_queued_paths_.device_pointer;

  const DeviceKernelArguments args(
      &work_size, &d_queued_paths, &d_num_queued_paths, &d_queued_kernel);

  queue_->zero_to_device(num_queued_paths_);
  queue_->enqueue(kernel, work_size, args);
}

void PathTraceWorkGPU::compact_main_paths(const int num_active_paths)
{
  /* Early out if there is nothing that needs to be compacted. */
  if (num_active_paths == 0) {
    max_active_main_path_index_ = 0;
    return;
  }

  const int min_compact_paths = 32;
  if (max_active_main_path_index_ == num_active_paths ||
      max_active_main_path_index_ < min_compact_paths)
  {
    return;
  }

  /* Compact. */
  compact_paths(num_active_paths,
                max_active_main_path_index_,
                DEVICE_KERNEL_INTEGRATOR_TERMINATED_PATHS_ARRAY,
                DEVICE_KERNEL_INTEGRATOR_COMPACT_PATHS_ARRAY,
                DEVICE_KERNEL_INTEGRATOR_COMPACT_STATES);

  /* Adjust max active path index now we know which part of the array is actually used. */
  max_active_main_path_index_ = num_active_paths;
}

void PathTraceWorkGPU::compact_shadow_paths(const bool force)
{
  IntegratorQueueCounter *queue_counter = integrator_queue_counter_.data();
  const int num_active_paths =
      queue_counter->num_queued[DEVICE_KERNEL_INTEGRATOR_SHADE_LIGHT_NEE] +
      queue_counter->num_queued[DEVICE_KERNEL_INTEGRATOR_INTERSECT_SHADOW] +
      queue_counter->num_queued[DEVICE_KERNEL_INTEGRATOR_SHADE_SHADOW] +
      queue_counter->num_queued[DEVICE_KERNEL_INTEGRATOR_SHADOW_PATH_MNEE_PENDING];

  /* Early out if there is nothing that needs to be compacted. */
  if (num_active_paths == 0) {
    if (integrator_next_shadow_path_index_.data()[0] != 0) {
      integrator_next_shadow_path_index_.data()[0] = 0;
      queue_->copy_to_device(integrator_next_shadow_path_index_);
    }
    return;
  }

  /* Compact if we can reduce the space used by half. Not always unless forced, since
   * compaction has a cost. */
  const float max_overhead_factor = 2.0f;
  const int min_compact_paths = 32;
  const int num_total_paths = integrator_next_shadow_path_index_.data()[0];
  if (!force && (num_total_paths < num_active_paths * max_overhead_factor ||
                 num_total_paths < min_compact_paths))
  {
    return;
  }

  /* Compact. */
  compact_paths(num_active_paths,
                num_total_paths,
                DEVICE_KERNEL_INTEGRATOR_TERMINATED_SHADOW_PATHS_ARRAY,
                DEVICE_KERNEL_INTEGRATOR_COMPACT_SHADOW_PATHS_ARRAY,
                DEVICE_KERNEL_INTEGRATOR_COMPACT_SHADOW_STATES);

  /* Adjust max active path index now we know which part of the array is actually used. */
  integrator_next_shadow_path_index_.data()[0] = num_active_paths;
  queue_->copy_to_device(integrator_next_shadow_path_index_);
}

void PathTraceWorkGPU::compact_paths(const int num_active_paths,
                                     const int max_active_path_index,
                                     DeviceKernel terminated_paths_kernel,
                                     DeviceKernel compact_paths_kernel,
                                     DeviceKernel compact_kernel)
{
  /* Compact fragmented path states into the start of the array, moving any paths
   * with index higher than the number of active paths into the gaps. */
  device_ptr d_compact_paths = queued_paths_.device_pointer;
  device_ptr d_num_queued_paths = num_queued_paths_.device_pointer;

  /* Create array with terminated paths that we can write to. */
  {
    /* TODO: can the work size be reduced here? */
    int offset = num_active_paths;
    const int work_size = num_active_paths;

    const DeviceKernelArguments args(&work_size, &d_compact_paths, &d_num_queued_paths, &offset);

    queue_->zero_to_device(num_queued_paths_);
    queue_->enqueue(terminated_paths_kernel, work_size, args);
  }

  /* Create array of paths that we need to compact, where the path index is bigger
   * than the number of active paths. */
  {
    const int work_size = max_active_path_index;

    const DeviceKernelArguments args(
        &work_size, &d_compact_paths, &d_num_queued_paths, &num_active_paths);

    queue_->zero_to_device(num_queued_paths_);
    queue_->enqueue(compact_paths_kernel, work_size, args);
  }

  queue_->copy_from_device(num_queued_paths_);
  queue_->synchronize();

  const int num_compact_paths = num_queued_paths_.data()[0];

  /* Move paths into gaps. */
  if (num_compact_paths > 0) {
    int work_size = num_compact_paths;
    int active_states_offset = 0;
    int terminated_states_offset = num_active_paths;

    const DeviceKernelArguments args(
        &d_compact_paths, &active_states_offset, &terminated_states_offset, &work_size);

    queue_->enqueue(compact_kernel, work_size, args);
  }
}

bool PathTraceWorkGPU::enqueue_work_tiles(bool &finished)
{
  /* If there are existing paths wait them to go to intersect closest kernel, which will align the
   * wavefront of the existing and newly added paths. */
  /* TODO: Check whether counting new intersection kernels here will have positive affect on the
   * performance. */
  const DeviceKernel kernel = get_most_queued_kernel();
  if (kernel != DEVICE_KERNEL_NUM && kernel != DEVICE_KERNEL_INTEGRATOR_INTERSECT_CLOSEST) {
    return false;
  }

  const int num_active_paths = num_active_main_paths_paths();

  /* Don't schedule more work if canceling. */
  if (is_cancel_requested()) {
    if (num_active_paths == 0) {
      finished = true;
    }
    return false;
  }

  finished = false;

  vector<KernelWorkTile> work_tiles;

  int max_num_camera_paths = max_num_paths_;
  int num_predicted_splits = 0;

  if (has_shadow_catcher()) {
    /* When there are shadow catchers in the scene bounce from them will split the state. So we
     * make sure there is enough space in the path states array to fit split states.
     *
     * Basically, when adding N new paths we ensure that there is 2*N available path states, so
     * that all the new paths can be split.
     *
     * Note that it is possible that some of the current states can still split, so need to make
     * sure there is enough space for them as well. */

    /* Number of currently in-flight states which can still split. */
    const int num_scheduled_possible_split = shadow_catcher_count_possible_splits();

    const int num_available_paths = max_num_paths_ - num_active_paths;
    const int num_new_paths = num_available_paths / 2;
    max_num_camera_paths = max(num_active_paths,
                               num_active_paths + num_new_paths - num_scheduled_possible_split);
    num_predicted_splits += num_scheduled_possible_split + num_new_paths;
  }

  /* Schedule when we're out of paths or there are too few paths to keep the
   * device occupied. */
  int num_paths = num_active_paths;
  if (num_paths == 0 || num_paths < min_num_active_main_paths_) {
    /* Get work tiles until the maximum number of path is reached. */
    while (num_paths < max_num_camera_paths) {
      KernelWorkTile work_tile;
      if (work_tile_scheduler_.get_work(&work_tile, max_num_camera_paths - num_paths)) {
        work_tiles.push_back(work_tile);
        num_paths += work_tile.w * work_tile.h * work_tile.num_samples;
      }
      else {
        break;
      }
    }

    /* If we couldn't get any more tiles, we're done. */
    if (work_tiles.empty() && num_paths == 0) {
      finished = true;
      return false;
    }
  }

  /* Initialize paths from work tiles. */
  if (work_tiles.empty()) {
    return false;
  }

  /* Compact state array when number of paths becomes small relative to the
   * known maximum path index, which makes computing active index arrays slow. */
  compact_main_paths(num_active_paths);

  if (has_shadow_catcher()) {
    integrator_next_main_path_index_.data()[0] = num_paths;
    queue_->copy_to_device(integrator_next_main_path_index_);
  }

  enqueue_work_tiles((device_scene_->data.bake.use) ? DEVICE_KERNEL_INTEGRATOR_INIT_FROM_BAKE :
                                                      DEVICE_KERNEL_INTEGRATOR_INIT_FROM_CAMERA,
                     work_tiles.data(),
                     work_tiles.size(),
                     num_active_paths,
                     num_predicted_splits);

  return true;
}

void PathTraceWorkGPU::enqueue_work_tiles(DeviceKernel kernel,
                                          const KernelWorkTile work_tiles[],
                                          const int num_work_tiles,
                                          const int num_active_paths,
                                          const int num_predicted_splits)
{
  /* Copy work tiles to device. */
  if (work_tiles_.size() < num_work_tiles) {
    work_tiles_.alloc(num_work_tiles);
  }

  int path_index_offset = num_active_paths;
  int max_tile_work_size = 0;
  for (int i = 0; i < num_work_tiles; i++) {
    KernelWorkTile &work_tile = work_tiles_.data()[i];
    work_tile = work_tiles[i];
    if (photon_only_) {
      work_tile.sample_offset = ~work_tile.sample_offset;
    }

    const int tile_work_size = work_tile.w * work_tile.h * work_tile.num_samples;

    work_tile.path_index_offset = path_index_offset;
    work_tile.work_size = tile_work_size;

    path_index_offset += tile_work_size;

    max_tile_work_size = max(max_tile_work_size, tile_work_size);
  }

  queue_->copy_to_device(work_tiles_);

  const device_ptr d_work_tiles = work_tiles_.device_pointer;
  device_ptr d_render_buffer = trace_buffer_device_pointer();

  /* Launch kernel. */
  device_ptr null_ptr = 0;
  int zero = 0;
  const DeviceKernelArguments args(
      &d_work_tiles, &num_work_tiles, &d_render_buffer, &max_tile_work_size, &null_ptr, &zero);

  queue_->enqueue(kernel, max_tile_work_size * num_work_tiles, args);

  max_active_main_path_index_ = path_index_offset + num_predicted_splits;
}

int PathTraceWorkGPU::num_active_main_paths_paths()
{
  IntegratorQueueCounter *queue_counter = integrator_queue_counter_.data();

  int num_paths = 0;
  for (int i = 0; i < DEVICE_GPU_KERNEL_INTEGRATOR_NUM; i++) {
    DCHECK_GE(queue_counter->num_queued[i], 0)
        << "Invalid number of queued states for kernel "
        << device_kernel_as_string(static_cast<DeviceKernel>(i));

    if (!kernel_is_shadow_path((DeviceKernel)i)) {
      num_paths += queue_counter->num_queued[i];
    }
  }

  return num_paths;
}

bool PathTraceWorkGPU::should_use_graphics_interop(PathTraceDisplay *display)
{
  /* There are few aspects with the graphics interop when using multiple devices caused by the fact
   * that the PathTraceDisplay has a single texture:
   *
   *   CUDA will return `CUDA_ERROR_NOT_SUPPORTED` from `cuGraphicsGLRegisterBuffer()` when
   *   attempting to register OpenGL PBO which has been mapped. Which makes sense, because
   *   otherwise one would run into a conflict of where the source of truth is. */
  if (has_multiple_works()) {
    return false;
  }

  if (!interop_use_checked_) {
    Device *device = queue_->device;
    interop_use_ = device->should_use_graphics_interop(display->graphics_interop_get_device(),
                                                       true);

    if (interop_use_) {
      LOG_INFO << "Using graphics interop GPU display update.";
    }
    else {
      LOG_INFO << "Using naive GPU display update.";
    }

    interop_use_checked_ = true;
  }

  return interop_use_;
}

void PathTraceWorkGPU::copy_to_display(PathTraceDisplay *display,
                                       PassMode pass_mode,
                                       const int num_samples)
{
  if (device_->have_error()) {
    /* Don't attempt to update GPU display if the device has errors: the error state will make
     * wrong decisions to happen about interop, causing more chained bugs. */
    return;
  }

  if (!buffers_->buffer.device_pointer) {
    LOG_WARNING << "Request for GPU display update without allocated render buffers.";
    return;
  }

  if (should_use_graphics_interop(display)) {
    if (copy_to_display_interop(display, pass_mode, num_samples)) {
      return;
    }

    /* If error happens when trying to use graphics interop fallback to the native implementation
     * and don't attempt to use interop for the further updates. */
    interop_use_ = false;
  }

  copy_to_display_naive(display, pass_mode, num_samples);
}

void PathTraceWorkGPU::copy_to_display_naive(PathTraceDisplay *display,
                                             PassMode pass_mode,
                                             const int num_samples)
{
  const BufferParams &effective_big_tile_params = (pass_mode == PassMode::DENOISED) ?
                                                      effective_denoised_big_tile_params_ :
                                                      effective_big_tile_params_;
  const BufferParams &effective_buffer_params = (pass_mode == PassMode::DENOISED) ?
                                                    effective_denoised_buffer_params_ :
                                                    effective_buffer_params_;

  const int full_x = effective_buffer_params.full_x;
  const int full_y = effective_buffer_params.full_y;
  const int width = effective_buffer_params.window_width;
  const int height = effective_buffer_params.window_height;
  const int final_width = buffers_->params.window_width;
  const int final_height = buffers_->params.window_height;

  const int texture_x = full_x - effective_big_tile_params.full_x +
                        effective_buffer_params.window_x - effective_big_tile_params.window_x;
  const int texture_y = full_y - effective_big_tile_params.full_y +
                        effective_buffer_params.window_y - effective_big_tile_params.window_y;

  /* Re-allocate display memory if needed, and make sure the device pointer is allocated.
   *
   * NOTE: allocation happens to the final resolution so that no re-allocation happens on every
   * change of the resolution divider. However, if the display becomes smaller, shrink the
   * allocated memory as well. */
  if (display_rgba_half_.data_width != final_width ||
      display_rgba_half_.data_height != final_height)
  {
    display_rgba_half_.alloc(final_width, final_height);
    /* TODO(sergey): There should be a way to make sure device-side memory is allocated without
     * transferring zeroes to the device. */
    queue_->zero_to_device(display_rgba_half_);
  }

  const PassType type = get_display_pass_access_info(pass_mode).type;
  PassAccessor::Destination destination(type == PASS_NONE ? film_->get_display_pass() : type,
                                        pass_mode);
  destination.d_pixels_half_rgba = display_rgba_half_.device_pointer;

  get_render_tile_film_pixels(destination, pass_mode, num_samples);

  queue_->copy_from_device(display_rgba_half_);
  queue_->synchronize();

  display->copy_pixels_to_texture(display_rgba_half_.data(), texture_x, texture_y, width, height);
}

bool PathTraceWorkGPU::copy_to_display_interop(PathTraceDisplay *display,
                                               PassMode pass_mode,
                                               const int num_samples)
{
  if (!device_graphics_interop_) {
    device_graphics_interop_ = queue_->graphics_interop_create();
  }

  GraphicsInteropBuffer &interop_buffer = display->graphics_interop_get_buffer();
  device_graphics_interop_->set_buffer(interop_buffer);

  const device_ptr d_rgba_half = device_graphics_interop_->map();
  if (!d_rgba_half) {
    return false;
  }

  PassAccessor::Destination destination = get_display_destination_template(display, pass_mode);
  destination.d_pixels_half_rgba = d_rgba_half;

  get_render_tile_film_pixels(destination, pass_mode, num_samples);

  device_graphics_interop_->unmap();

  return true;
}

void PathTraceWorkGPU::destroy_gpu_resources(PathTraceDisplay *display)
{
  if (!device_graphics_interop_) {
    return;
  }
  display->graphics_interop_activate();
  device_graphics_interop_ = nullptr;
  display->graphics_interop_deactivate();
}

void PathTraceWorkGPU::get_render_tile_film_pixels(const PassAccessor::Destination &destination,
                                                   PassMode pass_mode,
                                                   const int num_samples)
{
  const KernelFilm &kfilm = device_scene_->data.film;

  const PassAccessor::PassAccessInfo pass_access_info = get_display_pass_access_info(pass_mode);
  if (pass_access_info.type == PASS_NONE) {
    return;
  }

  const BufferParams &effective_buffer_params = (pass_mode == PassMode::DENOISED) ?
                                                    effective_denoised_buffer_params_ :
                                                    effective_buffer_params_;

  const PassAccessorGPU pass_accessor(queue_.get(), pass_access_info, kfilm.exposure, num_samples);

  pass_accessor.get_render_tile_pixels(buffers_.get(), effective_buffer_params, destination);
}

int PathTraceWorkGPU::adaptive_sampling_converge_filter_count_active_impl(const float threshold,
                                                                          bool reset)
{
  const int num_active_pixels = adaptive_sampling_convergence_check_count_active(threshold, reset);

  if (num_active_pixels) {
    enqueue_adaptive_sampling_filter_x();
    enqueue_adaptive_sampling_filter_y();
    queue_->synchronize();
  }

  return num_active_pixels;
}

int PathTraceWorkGPU::adaptive_sampling_convergence_check_count_active(const float threshold,
                                                                       bool reset)
{
  device_vector<uint> num_active_pixels(device_, "num_active_pixels", MEM_READ_WRITE);
  num_active_pixels.alloc(1);

  queue_->zero_to_device(num_active_pixels);

  const int work_size = effective_buffer_params_.width * effective_buffer_params_.height;
  if (!work_size) {
    return 0;
  }

  const int reset_int = reset; /* No bool kernel arguments. */

  const device_ptr d_trace_buffer = trace_buffer_device_pointer();
  const DeviceKernelArguments args(&d_trace_buffer,
                                   &effective_buffer_params_.full_x,
                                   &effective_buffer_params_.full_y,
                                   &effective_buffer_params_.width,
                                   &effective_buffer_params_.height,
                                   &threshold,
                                   &reset_int,
                                   &effective_buffer_params_.offset,
                                   &effective_buffer_params_.stride,
                                   &num_active_pixels.device_pointer);

  queue_->enqueue(DEVICE_KERNEL_ADAPTIVE_SAMPLING_CONVERGENCE_CHECK, work_size, args);

  queue_->copy_from_device(num_active_pixels);
  queue_->synchronize();

  return num_active_pixels.data()[0];
}

void PathTraceWorkGPU::enqueue_adaptive_sampling_filter_x()
{
  const int work_size = effective_buffer_params_.height;
  DCHECK_GT(work_size, 0);

  const device_ptr d_trace_buffer = trace_buffer_device_pointer();
  const DeviceKernelArguments args(&d_trace_buffer,
                                   &effective_buffer_params_.full_x,
                                   &effective_buffer_params_.full_y,
                                   &effective_buffer_params_.width,
                                   &effective_buffer_params_.height,
                                   &effective_buffer_params_.offset,
                                   &effective_buffer_params_.stride);

  queue_->enqueue(DEVICE_KERNEL_ADAPTIVE_SAMPLING_CONVERGENCE_FILTER_X, work_size, args);
}

void PathTraceWorkGPU::enqueue_adaptive_sampling_filter_y()
{
  const int work_size = effective_buffer_params_.width;
  DCHECK_GT(work_size, 0);

  const device_ptr d_trace_buffer = trace_buffer_device_pointer();
  const DeviceKernelArguments args(&d_trace_buffer,
                                   &effective_buffer_params_.full_x,
                                   &effective_buffer_params_.full_y,
                                   &effective_buffer_params_.width,
                                   &effective_buffer_params_.height,
                                   &effective_buffer_params_.offset,
                                   &effective_buffer_params_.stride);

  queue_->enqueue(DEVICE_KERNEL_ADAPTIVE_SAMPLING_CONVERGENCE_FILTER_Y, work_size, args);
}

void PathTraceWorkGPU::merge_photon_estimates()
{
  if (!device_scene_->data.film.photon_segment_stride) {
    return;
  }
  const int count = effective_buffer_params_.width * effective_buffer_params_.height;
  if (count) {
    const DeviceKernelArguments args(&buffers_->buffer.device_pointer, &count);
    queue_->enqueue(DEVICE_KERNEL_PHOTON_MERGE, count, args);
    /* Denoising can use a different queue; publish the completed merged film first. */
    queue_->synchronize();
  }
}

void PathTraceWorkGPU::cryptomatte_postproces()
{
  const int work_size = effective_buffer_params_.width * effective_buffer_params_.height;
  if (!work_size) {
    return;
  }

  const bool photon_metadata = estimator_sample_offset_ >=
                               device_scene_->data.integrator.photon_native_samples;
  const device_ptr d_native_buffer = buffers_->buffer.device_pointer +
                                     sizeof(float) *
                                         device_scene_->data.film.photon_segment_stride *
                                         (photon_metadata ? 2 : 1);
  const DeviceKernelArguments args(&d_native_buffer,
                                   &work_size,
                                   &effective_buffer_params_.offset,
                                   &effective_buffer_params_.stride);

  queue_->enqueue(DEVICE_KERNEL_CRYPTOMATTE_POSTPROCESS, work_size, args);
}

void PathTraceWorkGPU::denoise_volume_guiding_buffers()
{
  if (effective_buffer_params_.width == 0 || effective_buffer_params_.height == 0) {
    return;
  }

  const DeviceKernelArguments args(&buffers_->buffer.device_pointer,
                                   &effective_buffer_params_.full_x,
                                   &effective_buffer_params_.full_y,
                                   &effective_buffer_params_.width,
                                   &effective_buffer_params_.height,
                                   &effective_buffer_params_.offset,
                                   &effective_buffer_params_.stride);

  {
    const int work_size = effective_buffer_params_.width * effective_buffer_params_.height;
    DCHECK_GT(work_size, 0);
    queue_->enqueue(DEVICE_KERNEL_VOLUME_GUIDING_FILTER_X, work_size, args);
  }

  {
    const int work_size = effective_buffer_params_.width;
    DCHECK_GT(work_size, 0);
    queue_->enqueue(DEVICE_KERNEL_VOLUME_GUIDING_FILTER_Y, work_size, args);
  }
}

bool PathTraceWorkGPU::read_preview_pass(const PassAccessor::PassAccessInfo &info,
                                         float exposure, int samples, int channels, float *pixels)
{
  const BufferParams &slice = effective_buffer_params_;
  const BufferParams &tile = effective_big_tile_params_;
  const int x = slice.full_x + slice.window_x - tile.full_x - tile.window_x;
  const int y = slice.full_y + slice.window_y - tile.full_y - tile.window_y;
  if (slice.window_width == 0 || slice.window_height == 0) {
    return true;
  }
  if (x < 0 || y < 0 || x + slice.window_width > tile.window_width ||
      y + slice.window_height > tile.window_height || !buffers_->buffer.device_pointer) {
    return false;
  }
  const size_t row = size_t(slice.window_width) * channels;
  if (!preview_pass_.alloc(row * slice.window_height)) {
    return false;
  }
  if (!preview_pass_.device_pointer) {
    queue_->zero_to_device(preview_pass_);
  }
  if (!preview_pass_.device_pointer) {
    return false;
  }
  PassAccessor::Destination destination;
  destination.num_components = channels;
  destination.d_pixels = preview_pass_.device_pointer;
  const PassAccessorGPU accessor(queue_.get(), info, exposure, samples);
  if (!accessor.get_render_tile_pixels(buffers_.get(), slice, destination)) {
    return false;
  }
  queue_->copy_from_device(preview_pass_);
  if (!queue_->synchronize()) {
    return false;
  }
  for (int i = 0; i < slice.window_height; ++i) {
    std::copy_n(preview_pass_.data() + i * row, row,
                pixels + (size_t(y + i) * tile.window_width + x) * channels);
  }
  return true;
}

bool PathTraceWorkGPU::copy_render_buffers_from_device()
{
  /* May not exist if cancelled before rendering started. */
  if (!buffers_->buffer.device_pointer) {
    return false;
  }

  queue_->copy_from_device(buffers_->buffer);

  /* Synchronize so that the CPU-side buffer is available at the exit of this function. */
  return queue_->synchronize();
}

bool PathTraceWorkGPU::copy_render_buffers_to_device()
{
  queue_->copy_to_device(buffers_->buffer);

  /* NOTE: The direct device access to the buffers only happens within this path trace work. The
   * rest of communication happens via API calls which involves `copy_render_buffers_from_device()`
   * which will perform synchronization as needed. */

  return true;
}

bool PathTraceWorkGPU::copy_convergence_masks(const RenderBuffers &source)
{
  /* Reuse the strided float-copy film kernel. Only one float per pixel and estimator
   * crosses PCIe; accumulated lighting and photon maps remain on their own device. */
  if (!convergence_buffer_) {
    convergence_buffer_ = make_unique<RenderBuffers>(device_);
  }
  BufferParams params;
  params.width = params.window_width = source.params.width;
  params.height = params.window_height = source.params.height;
  params.pass_stride = 1;
  params.update_offset_stride();
  convergence_buffer_->reset(params);
  const KernelFilm &film = device_scene_->data.film;
  const int stride = source.params.pass_stride;
  const int segment = film.photon_segment_stride ? film.photon_segment_stride : stride;
  const size_t pixels = source.buffer.size() / stride;
  PassAccessor::PassAccessInfo info;
  info.type = PASS_AOV_VALUE;
  info.mode = PassMode::NOISY;
  info.offset = 0;
  info.use_sample_count = false;
  const PassAccessorGPU accessor(queue_.get(), info, 1.0f, 1);
  PassAccessor::Destination destination;
  destination.d_pixels = buffers_->buffer.device_pointer;
  destination.num_components = 1;
  destination.pixel_stride = stride;
  for (int offset = film.photon_segment_stride; offset < stride; offset += segment) {
    destination.pixel_offset = offset + film.pass_adaptive_aux_buffer + 3;
    for (size_t pixel = 0; pixel < pixels; ++pixel) {
      convergence_buffer_->buffer.data()[pixel] =
          source.buffer.data()[pixel * stride + destination.pixel_offset];
    }
    queue_->copy_to_device(convergence_buffer_->buffer);
    if (!accessor.get_render_tile_pixels(convergence_buffer_.get(), destination)) {
      return false;
    }
  }
  return !device_->have_error();
}

bool PathTraceWorkGPU::zero_render_buffers()
{
  gpu_guiding_reset_ = true;
  photon_guide_reset_ = true;
  reset_photon_sampling();
  queue_->zero_to_device(buffers_->buffer);

  return true;
}

bool PathTraceWorkGPU::has_shadow_catcher() const
{
  return device_scene_->data.integrator.has_shadow_catcher;
}

int PathTraceWorkGPU::shadow_catcher_count_possible_splits()
{
  if (max_active_main_path_index_ == 0) {
    return 0;
  }

  if (!has_shadow_catcher()) {
    return 0;
  }

  queue_->zero_to_device(num_queued_paths_);

  const int work_size = max_active_main_path_index_;
  device_ptr d_num_queued_paths = num_queued_paths_.device_pointer;

  const DeviceKernelArguments args(&work_size, &d_num_queued_paths);

  queue_->enqueue(DEVICE_KERNEL_INTEGRATOR_SHADOW_CATCHER_COUNT_POSSIBLE_SPLITS, work_size, args);
  queue_->copy_from_device(num_queued_paths_);
  queue_->synchronize();

  return num_queued_paths_.data()[0];
}

bool PathTraceWorkGPU::kernel_uses_sorting(DeviceKernel kernel)
{
  return (kernel == DEVICE_KERNEL_INTEGRATOR_SHADE_SURFACE ||
          kernel == DEVICE_KERNEL_INTEGRATOR_SHADE_SURFACE_RAYTRACE);
}

bool PathTraceWorkGPU::kernel_creates_shadow_paths(DeviceKernel kernel)
{
  return (kernel == DEVICE_KERNEL_INTEGRATOR_SHADE_SURFACE ||
          kernel == DEVICE_KERNEL_INTEGRATOR_SHADE_SURFACE_RAYTRACE ||
          kernel == DEVICE_KERNEL_INTEGRATOR_INTERSECT_MNEE ||
          kernel == DEVICE_KERNEL_INTEGRATOR_SHADE_VOLUME ||
          kernel == DEVICE_KERNEL_INTEGRATOR_SHADE_VOLUME_RAY_MARCHING ||
          kernel == DEVICE_KERNEL_INTEGRATOR_SHADE_DEDICATED_LIGHT);
}

bool PathTraceWorkGPU::kernel_creates_ao_paths(DeviceKernel kernel)
{
  return (device_scene_->data.kernel_features & KERNEL_FEATURE_AO) &&
         (kernel == DEVICE_KERNEL_INTEGRATOR_SHADE_SURFACE ||
          kernel == DEVICE_KERNEL_INTEGRATOR_SHADE_SURFACE_RAYTRACE);
}

bool PathTraceWorkGPU::kernel_is_shadow_path(DeviceKernel kernel)
{
  return (kernel == DEVICE_KERNEL_INTEGRATOR_INTERSECT_SHADOW ||
          kernel == DEVICE_KERNEL_INTEGRATOR_SHADE_SHADOW ||
          kernel == DEVICE_KERNEL_INTEGRATOR_SHADE_LIGHT_NEE);
}

int PathTraceWorkGPU::kernel_max_active_main_path_index(DeviceKernel kernel)
{
  return (kernel_is_shadow_path(kernel)) ? integrator_next_shadow_path_index_.data()[0] :
                                           max_active_main_path_index_;
}

CCL_NAMESPACE_END
