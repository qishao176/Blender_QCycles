/* SPDX-FileCopyrightText: 2024 Blender Authors
 *
 * SPDX-License-Identifier: Apache-2.0 */

#pragma once

#include "kernel/film/write.h"
#include "util/atomic.h"
#include "kernel/film/deep_physical.h"

CCL_NAMESPACE_BEGIN

/* -------------------------------------------------------------------- */
/** \name Deep Sample Writing
 *
 * Functions for writing deep samples during path tracing.
 * Each sample stores RGBA color and depth information.
 * \{ */

#ifdef __DEEP_OUTPUT__

/* Keep these flags and pack/unpack helpers in sync with session/deep_buffers.h. */
constexpr uint32_t DEEP_SAMPLE_FLAG_HARD_SURFACE_METADATA = (1u << 0);
constexpr uint32_t DEEP_SAMPLE_INFO_FLAG_MASK = 0xffu;
constexpr uint32_t DEEP_SAMPLE_INFO_CAMERA_SAMPLE_SHIFT = 8u;
constexpr uint32_t DEEP_SAMPLE_INFO_CAMERA_SAMPLE_MASK =
    0xffffffffu & ~DEEP_SAMPLE_INFO_FLAG_MASK;
constexpr uint32_t DEEP_INVALID_SAMPLE_INDEX = 0xffffffffu;

/* In Physical mode the existing path fields hold a source kind and a depth anchor,
 * rather than a slot in the legacy append buffer. Anchors survive bounded compaction. */
ccl_device_inline void film_deep_bind(KernelGlobals kg,
                                      ConstIntegratorState state,
                                      const float z,
                                      const float back,
                                      const float alpha,
                                      const uint kind,
                                      const uint object = DEEP_BOUNDED_NONSURFACE,
                                      const uint shader = 0,
                                      const uint normal = 0)
{
  if (!kernel_data.film.deep_physical_ptr ||
      INTEGRATOR_STATE(state, path, bounce) != 0 ||
      (INTEGRATOR_STATE(state, path, flag) & PATH_RAY_SHADOW_CATCHER_PASS)) return;
#ifndef __KERNEL_GPU__
  IntegratorState writable = const_cast<IntegratorStateCPU *>(state);
#else
  IntegratorState writable = state;
#endif
  deep_physical_append(kernel_data.film.deep_physical_ptr,
                       INTEGRATOR_STATE(state, path, render_pixel_index),
                       kernel_data.film.deep_lpe_channels,
                       z, back, alpha, object, shader, normal);
  INTEGRATOR_STATE_WRITE(writable, path, deep_surface_sample_idx) = kind;
  INTEGRATOR_STATE_WRITE(writable, path, deep_volume_z) = fmaxf(z, 1e-8f);
}

/**
 * Deep sample data structure for kernel use.
 * Matches DeepSampleData in session/deep_buffers.h.
 */
struct ccl_align(16) KernelDeepSample {
  float r, g, b, a;
  float z;
  float z_back;
  uint32_t surface_object;
  uint32_t surface_prim;
  uint32_t surface_shader;
  uint32_t packed_geometric_normal;
  uint32_t flags;
};

static_assert(sizeof(KernelDeepSample) == 48, "KernelDeepSample layout must match host.");

ccl_device_inline void deep_make_surface_key(const int object,
                                             const int prim,
                                             const int shader,
                                             uint32_t &surface_object,
                                             uint32_t &surface_prim,
                                             uint32_t &surface_shader)
{
  surface_object = uint32_t(object);
  surface_prim = uint32_t(prim);
  surface_shader = uint32_t(shader);
}

ccl_device_inline float2 deep_encode_octahedral_normal(const float3 normal)
{
  const float3 n = safe_normalize(normal);
  const float inv_l1 = 1.0f / (fabsf(n.x) + fabsf(n.y) + fabsf(n.z) + 1e-20f);
  float2 encoded = make_float2(n.x * inv_l1, n.y * inv_l1);

  if (n.z < 0.0f) {
    encoded = make_float2(copysignf(1.0f - fabsf(encoded.y), encoded.x),
                          copysignf(1.0f - fabsf(encoded.x), encoded.y));
  }

  return encoded;
}

ccl_device_inline uint32_t deep_pack_unorm_16(const float value)
{
  return (uint32_t)clamp(int((clamp(value, 0.0f, 1.0f) * 65535.0f) + 0.5f), 0, 65535);
}

ccl_device_inline uint32_t deep_pack_geometric_normal(const float3 normal)
{
  const float2 encoded = deep_encode_octahedral_normal(normal);
  const float2 unorm = encoded * 0.5f + make_float2(0.5f, 0.5f);
  return deep_pack_unorm_16(unorm.x) | (deep_pack_unorm_16(unorm.y) << 16);
}

ccl_device_inline uint32_t deep_pack_sample_info(const uint32_t flags,
                                                 const uint32_t camera_sample)
{
  return (flags & DEEP_SAMPLE_INFO_FLAG_MASK) |
         ((camera_sample << DEEP_SAMPLE_INFO_CAMERA_SAMPLE_SHIFT) &
          DEEP_SAMPLE_INFO_CAMERA_SAMPLE_MASK);
}

ccl_device_inline uint32_t film_write_deep_sample_with_metadata(
    KernelGlobals kg,
    const uint32_t pixel_index,
    ccl_global KernelDeepSample *ccl_restrict deep_samples,
    ccl_global uint32_t *ccl_restrict sample_counts,
    const float alpha,
    const float z,
    const float z_back,
    const uint32_t surface_object,
    const uint32_t surface_prim,
    const uint32_t surface_shader,
    const uint32_t packed_geometric_normal,
    const uint32_t sample_info)
{
  /* Bounds check: ensure pixel index is within allocated buffer. */
  const uint32_t num_pixels = kernel_data.film.deep_width * kernel_data.film.deep_height;
  if (pixel_index >= num_pixels) {
    return DEEP_INVALID_SAMPLE_INDEX;
  }

  const uint32_t sample_idx = atomic_fetch_and_add_uint32(&sample_counts[pixel_index], 1);

  if (sample_idx >= kernel_data.film.deep_max_samples) {
    atomic_fetch_and_add_uint32(&sample_counts[pixel_index], -1);
    return DEEP_INVALID_SAMPLE_INDEX;
  }

  const uint64_t offset = uint64_t(pixel_index) * kernel_data.film.deep_max_samples + sample_idx;

  deep_samples[offset].r = 0.0f;
  deep_samples[offset].g = 0.0f;
  deep_samples[offset].b = 0.0f;
  deep_samples[offset].a = alpha;
  deep_samples[offset].z = z;
  deep_samples[offset].z_back = z_back;
  deep_samples[offset].surface_object = surface_object;
  deep_samples[offset].surface_prim = surface_prim;
  deep_samples[offset].surface_shader = surface_shader;
  deep_samples[offset].packed_geometric_normal = packed_geometric_normal;
  deep_samples[offset].flags = sample_info;

  return sample_idx;
}

/**
 * Write a deep sample for the current pixel.
 *
 * This appends a new sample to the pixel's sample list. If the maximum
 * number of samples has been reached, the sample is dropped.
 *
 * \param kg: Kernel globals
 * \param pixel_index: Pixel index for the current sample
 * \param deep_samples: Pointer to deep sample buffer
 * \param sample_counts: Pointer to per-pixel sample counts
 * \param z: Front depth (distance from camera)
 * \param z_back: Back depth (same as z for surfaces, different for volumes)
 */
ccl_device_inline uint32_t film_write_deep_sample(
    KernelGlobals kg,
    const uint32_t pixel_index,
    ccl_global KernelDeepSample *ccl_restrict deep_samples,
    ccl_global uint32_t *ccl_restrict sample_counts,
    const float z,
    const float z_back)
{
  return film_write_deep_sample_with_metadata(
      kg, pixel_index, deep_samples, sample_counts, 1.0f, z, z_back, 0, 0, 0, 0, 0);
}

/**
 * Write a deep sample with transparency (for transparent surfaces).
 *
 * \param alpha: Explicit alpha value (0 = fully transparent, 1 = fully opaque)
 */
ccl_device_inline uint32_t film_write_deep_sample_transparent(
    KernelGlobals kg,
    const uint32_t pixel_index,
    ccl_global KernelDeepSample *ccl_restrict deep_samples,
    ccl_global uint32_t *ccl_restrict sample_counts,
    const float alpha,
    const float z,
    const float z_back)
{
  return film_write_deep_sample_with_metadata(
      kg, pixel_index, deep_samples, sample_counts, alpha, z, z_back, 0, 0, 0, 0, 0);
}

ccl_device_inline uint32_t film_write_deep_surface_sample_transparent(
    KernelGlobals kg,
    const uint32_t pixel_index,
    ccl_global KernelDeepSample *ccl_restrict deep_samples,
    ccl_global uint32_t *ccl_restrict sample_counts,
    const float alpha,
    const float z,
    const float z_back,
    const uint32_t surface_object,
    const uint32_t surface_prim,
    const uint32_t surface_shader,
    const uint32_t packed_geometric_normal,
    const uint32_t camera_sample)
{
  return film_write_deep_sample_with_metadata(
      kg,
      pixel_index,
      deep_samples,
      sample_counts,
      alpha,
      z,
      z_back,
      surface_object,
      surface_prim,
      surface_shader,
      packed_geometric_normal,
      deep_pack_sample_info(DEEP_SAMPLE_FLAG_HARD_SURFACE_METADATA, camera_sample));
}

/**
 * Write a volumetric deep sample.
 *
 * Volumes have a front and back depth representing the entry and exit points.
 *
 * \param z_entry: Depth where ray enters the volume.
 * \param z_exit: Depth where ray exits the volume.
 */
ccl_device_inline void film_write_deep_sample_volume(
    KernelGlobals kg,
    const uint32_t pixel_index,
    ccl_global KernelDeepSample *ccl_restrict deep_samples,
    ccl_global uint32_t *ccl_restrict sample_counts,
    const float alpha,
    const float z_entry,
    const float z_exit)
{
  /* Volumes use z_back to represent the volume thickness. */
  film_write_deep_sample_transparent(
      kg, pixel_index, deep_samples, sample_counts, alpha, z_entry, z_exit);
}

ccl_device_inline void film_accumulate_deep_surface_rgb(
    KernelGlobals kg,
    const uint32_t pixel_index,
    const uint32_t sample_idx,
    ccl_global KernelDeepSample *ccl_restrict deep_samples,
    const Spectrum contribution)
{
  if (!kernel_data.film.use_deep_output || deep_samples == nullptr ||
      sample_idx == DEEP_INVALID_SAMPLE_INDEX)
  {
    return;
  }

  const uint64_t offset = uint64_t(pixel_index) * kernel_data.film.deep_max_samples + sample_idx;
  const float3 contribution_rgb = spectrum_to_rgb(contribution);
  atomic_add_and_fetch_float(&deep_samples[offset].r, contribution_rgb.x);
  atomic_add_and_fetch_float(&deep_samples[offset].g, contribution_rgb.y);
  atomic_add_and_fetch_float(&deep_samples[offset].b, contribution_rgb.z);
}

/* Fixed depth bins accumulate ALL camera samples, independently of the native
 * append buffer's capacity. RGB already includes camera throughput. Alpha is
 * owned by native volume intervals and must never be added a second time. */
ccl_device_inline void film_accumulate_deep_photon(KernelGlobals kg,
                                                   const uint pixel,
                                                   const float z,
                                                   const Spectrum contribution)
{
  if (!kernel_data.film.deep_photon_ptr || kernel_data.film.deep_photon_bins == 0 ||
      pixel >= uint(kernel_data.film.deep_width * kernel_data.film.deep_height) ||
      !isfinite_safe(z))
  {
    return;
  }
  const int bins = kernel_data.film.deep_photon_bins;
  const float lo = kernel_data.film.deep_photon_z_min;
  const float hi = kernel_data.film.deep_photon_z_max;
  const int bin = clamp(int((z - lo) * bins / (hi - lo)), 0, bins - 1);
  ccl_global float4 *data = (ccl_global float4 *)kernel_data.film.deep_photon_ptr;
  ccl_global float4 *dst = data + size_t(pixel) * bins + bin;
  const float3 rgb = spectrum_to_rgb(contribution) * kernel_data.film.exposure;
  atomic_add_and_fetch_float(&dst->x, rgb.x);
  atomic_add_and_fetch_float(&dst->y, rgb.y);
  atomic_add_and_fetch_float(&dst->z, rgb.z);
  atomic_add_and_fetch_float(&dst->w, average(rgb) * z);
}

/** \} */

#endif /* __DEEP_OUTPUT__ */

CCL_NAMESPACE_END
