/* SPDX-FileCopyrightText: 2026 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#pragma once

#include "kernel/sample/sobol_burley.h"
#include "util/hash.h"
#include "util/math.h"

CCL_NAMESPACE_BEGIN

/* Keep a spatial Morton ordering even when the scene uses a white-noise sampler.
 * Do not hash this index separately for each pixel: the shared sequence needs
 * neighboring pixels to retain its spatial stratification. */
ccl_device_inline uint photon_noise_pixel_index(const int x, const int y, const uint seed)
{
  return nested_uniform_scramble_base4(morton2d(uint(x), uint(y)), seed ^ 0x70686e73u);
}

/* A fresh 1-SPP blue-noise realization for each camera sample, photon and bounce.
 * This is the native spatial Owen/Sobol construction, with independent temporal
 * scrambles. Absolute samples and emission rounds survive tiling and subsets. */
ccl_device_inline float photon_noise_sample(const uint pixel_index,
                                            const uint sample,
                                            const uint photon_index,
                                            const uint iteration,
                                            const uint bounce)
{
  const uint seed = hash_uint3(photon_index, iteration, hash_uint2(sample, bounce));
  return sobol_burley_sample_1D(pixel_index, 0x50484e53u, seed, 0xffffffffu);
}

/* Stochastic rounding of a normalized kernel value to multiples of strength.
 * E[round(value/strength)] * strength = value. The same scalar weights all RGB
 * components and every output pass. At strength 1 a smooth unit peak becomes
 * equal-amplitude grains, with density given by the original kernel profile.
 * This adds variance, not blur, and averages back to the original finite kernel. */
ccl_device_inline float photon_noise_quantize(const float value,
                                              const float strength,
                                              const float random)
{
  if (!(strength > 1e-6f)) {
    return value;
  }
  const float scaled = value / strength;
  const float base = floorf(scaled);
  return strength * (base + float(random < scaled - base));
}

CCL_NAMESPACE_END
