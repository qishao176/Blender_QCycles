/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: Apache-2.0 */

#pragma once

#include "kernel/closure/bsdf.h"
#include "kernel/integrator/state.h"
#include "util/hash.h"

CCL_NAMESPACE_BEGIN

/* Bounded NVIDIA surface guiding. A frozen CDF is queried during a batch;
 * disjoint statistics are published only after every main/shadow queue drains.
 * These are proposal distributions, never cached radiance contributions. */
#ifdef __GPU_PATH_GUIDING__
ccl_device_inline bool gpu_guiding_active(KernelGlobals kg)
{
  return (kernel_data.kernel_features & KERNEL_FEATURE_GPU_PATH_GUIDING) &&
         kernel_integrator_state.gpu_guiding_active;
}

ccl_device_inline bool gpu_guiding_surface_supported(const ccl_private ShaderData *sd)
{
  bool found = false;
  for (int i = 0; i < sd->num_closure; ++i) {
    const ccl_private ShaderClosure *sc = &sd->closure[i];
    if (!CLOSURE_IS_BSDF_OR_BSSRDF(sc->type) || sc->sample_weight <= 0.0f) {
      continue;
    }
    /* Mixed delta/transmission/BSSRDF and hair closures use the original sampler.
     * Do not assign a continuous density to a discrete event. */
    switch (sc->type) {
      case CLOSURE_BSDF_DIFFUSE_ID:
      case CLOSURE_BSDF_OREN_NAYAR_ID:
      case CLOSURE_BSDF_BURLEY_ID:
      case CLOSURE_BSDF_SHEEN_ID:
      case CLOSURE_BSDF_ASHIKHMIN_VELVET_ID:
        break;
      case CLOSURE_BSDF_MICROFACET_GGX_ID:
      case CLOSURE_BSDF_MICROFACET_BECKMANN_ID:
      case CLOSURE_BSDF_ASHIKHMIN_SHIRLEY_ID:
        if (bsdf_get_specular_roughness_squared(sc) < 1e-4f) {
          return false;
        }
        break;
      default:
        return false;
    }
    found = true;
  }
  return found;
}

ccl_device_inline int gpu_guiding_cell(KernelGlobals kg, const ccl_private ShaderData *sd)
{
  const float3 p = (sd->P - make_float3(kernel_data.integrator.gpu_guiding_min)) *
                   make_float3(kernel_data.integrator.gpu_guiding_scale);
  if (min(p.x, min(p.y, p.z)) < 0.0f || max(p.x, max(p.y, p.z)) >= GPU_GUIDE_GRID) {
    return -1;
  }
  const float3 n = sd->Ng;
  const float3 an = fabs(n);
  const int side = an.x >= an.y && an.x >= an.z ? (n.x >= 0.0f ? 0 : 1) :
                   an.y >= an.z ? (n.y >= 0.0f ? 2 : 3) : (n.z >= 0.0f ? 4 : 5);
  return ((int(p.z) * GPU_GUIDE_GRID + int(p.y)) * GPU_GUIDE_GRID + int(p.x)) * 6 + side;
}

ccl_device_inline int gpu_guiding_bin(const float3 d)
{
  const float phi = atan2f(d.y, d.x) * M_1_2PI_F + 0.5f;
  const int u = clamp(int(phi * 8.0f), 0, 7);
  const int v = clamp(int((d.z * 0.5f + 0.5f) * 8.0f), 0, 7);
  return v * 8 + u;
}

ccl_device_inline float gpu_guiding_pdf(const int cell, const float3 d)
{
  const ccl_global float *cdf = kernel_integrator_state.gpu_guiding_cdf +
                                cell * GPU_GUIDE_STRIDE;
  const int bin = gpu_guiding_bin(d);
  return (cdf[bin] - (bin ? cdf[bin - 1] : 0.0f)) * (GPU_GUIDE_BINS * M_1_4PI_F);
}

ccl_device_inline float gpu_guiding_fraction(KernelGlobals kg,
                                             const ccl_private ShaderData *sd,
                                             ccl_private int *cell)
{
  *cell = -1;
  if (!gpu_guiding_active(kg) || !gpu_guiding_surface_supported(sd)) {
    return 0.0f;
  }
  *cell = gpu_guiding_cell(kg, sd);
  if (*cell < 0 || kernel_integrator_state.gpu_guiding_cdf[
                       *cell * GPU_GUIDE_STRIDE + GPU_GUIDE_BINS] == 0.0f)
  {
    return 0.0f;
  }
  return kernel_data.integrator.gpu_guiding_probability;
}

ccl_device_inline float3 gpu_guiding_sample(const int cell, const float2 rand)
{
  const ccl_global float *cdf = kernel_integrator_state.gpu_guiding_cdf +
                                cell * GPU_GUIDE_STRIDE;
  const float r = min(rand.x, 0.99999994f);
  int lo = 0, hi = GPU_GUIDE_BINS - 1;
  while (lo < hi) {
    const int mid = (lo + hi) / 2;
    if (r < cdf[mid]) {
      hi = mid;
    }
    else {
      lo = mid + 1;
    }
  }
  const float begin = lo ? cdf[lo - 1] : 0.0f;
  const float u = clamp((r - begin) / (cdf[lo] - begin), 0.0f, 0.99999994f);
  const float phi = ((float(lo % 8) + u) / 8.0f - 0.5f) * M_2PI_F;
  const float z = 2.0f * (float(lo / 8) + rand.y) / 8.0f - 1.0f;
  const float radius = safe_sqrtf(1.0f - z * z);
  return make_float3(radius * cosf(phi), radius * sinf(phi), z);
}

/* One selected anchor bounds history storage to 16 B/path, also copied to
 * shadow branches. Sampling depth is independent of path contributions. The
 * learned tail omits direct NEE at the anchor itself, not at later vertices. */
ccl_device_inline void gpu_guiding_record_anchor(KernelGlobals kg,
                                                 IntegratorState state,
                                                 const ccl_private ShaderData *sd,
                                                 const float3 direction,
                                                 const float pdf)
{
  if (!gpu_guiding_active(kg) || !kernel_integrator_state.gpu_guiding_training ||
      INTEGRATOR_STATE(state, gpu_guiding, bin) != -1 ||
      !gpu_guiding_surface_supported(sd))
  {
    return;
  }
  const uint target = hash_uint2(INTEGRATOR_STATE(state, path, rng_pixel),
                                 INTEGRATOR_STATE(state, path, sample) ^ 0x67756964u) & 3u;
  if (INTEGRATOR_STATE(state, path, bounce) < target) {
    return;
  }
  const int cell = gpu_guiding_cell(kg, sd);
  if (cell < 0 || pdf <= 0.0f) {
    return;
  }
  const Spectrum scale = INTEGRATOR_STATE(state, path, throughput) * pdf;
  const Spectrum inverse = safe_divide_color(one_spectrum(), scale);
  INTEGRATOR_STATE_WRITE(state, gpu_guiding, inverse_weight) = inverse;
  INTEGRATOR_STATE_WRITE(state, gpu_guiding, bin) = cell * GPU_GUIDE_STRIDE +
                                                  gpu_guiding_bin(direction);
  atomic_add_and_fetch_float(&kernel_integrator_state.gpu_guiding_statistics[
                                cell * GPU_GUIDE_STRIDE + GPU_GUIDE_BINS], 1.0f);
}

ccl_device_inline void gpu_guiding_add(const int bin,
                                      const Spectrum inverse,
                                      const Spectrum contribution)
{
  if (bin >= 0) {
    const float value = average(max(contribution * inverse, zero_spectrum()));
    if (isfinite_safe(value) && value > 0.0f) {
      /* Training-only outlier guard. The actual film contribution is untouched. */
      atomic_add_and_fetch_float(&kernel_integrator_state.gpu_guiding_statistics[bin],
                                 min(value, 1e8f));
    }
  }
}

ccl_device_inline void gpu_guiding_record_radiance(KernelGlobals kg,
                                                   IntegratorState state,
                                                   const Spectrum radiance)
{
  if (gpu_guiding_active(kg) && kernel_integrator_state.gpu_guiding_training) {
    gpu_guiding_add(INTEGRATOR_STATE(state, gpu_guiding, bin),
                    INTEGRATOR_STATE(state, gpu_guiding, inverse_weight),
                    INTEGRATOR_STATE(state, path, throughput) * radiance);
  }
}

ccl_device_inline void gpu_guiding_record_shadow(KernelGlobals kg, IntegratorShadowState state)
{
  if (gpu_guiding_active(kg) && kernel_integrator_state.gpu_guiding_training) {
    gpu_guiding_add(INTEGRATOR_STATE(state, shadow_path, gpu_guiding_bin),
                    INTEGRATOR_STATE(state, shadow_path, gpu_guiding_inverse_weight),
                    INTEGRATOR_STATE(state, shadow_path, throughput));
  }
}

ccl_device_inline void gpu_guiding_publish(const int cell)
{
  if (cell >= GPU_GUIDE_CELLS) {
    return;
  }
  ccl_global float *cdf = kernel_integrator_state.gpu_guiding_cdf + cell * GPU_GUIDE_STRIDE;
  const ccl_global float *stats = kernel_integrator_state.gpu_guiding_statistics +
                                  cell * GPU_GUIDE_STRIDE;
  float sum = 0.0f;
  for (int i = 0; i < GPU_GUIDE_BINS; ++i) {
    sum += stats[i];
  }
  if (stats[GPU_GUIDE_BINS] < 32.0f || !(sum > 0.0f) || !isfinite_safe(sum)) {
    cdf[GPU_GUIDE_BINS] = 0.0f;
    return;
  }
  float cumulative = 0.0f;
  for (int i = 0; i < GPU_GUIDE_BINS; ++i) {
    cumulative += 0.95f * (stats[i] / sum) + 0.05f / GPU_GUIDE_BINS;
    cdf[i] = cumulative;
  }
  cdf[GPU_GUIDE_BINS - 1] = 1.0f;
  cdf[GPU_GUIDE_BINS] = 1.0f;
  atomic_fetch_and_add_uint32(kernel_integrator_state.gpu_guiding_ready_cells, 1u);
}
#endif

CCL_NAMESPACE_END
