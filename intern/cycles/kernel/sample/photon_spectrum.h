/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: Apache-2.0 */

#pragma once

#include "kernel/sample/util.h"
#include "util/hash.h"

CCL_NAMESPACE_BEGIN

/* A photon and its spectral camera query must describe the same monochromatic
 * transport problem. Multiple coherent groups cover the spectrum within a map;
 * glass on both sides of a merge must still use the SAME group's wavelength.
 * Use the absolute emission round, NOT its hash, to preserve stratification.
 * A scrambled base-2 sequence covers every equal-probability wavelength stratum
 * once in a power-of-two prefix. Different seeds/devices scramble independently.
 * This changes the sample sequence, not the native wavelength PDF or RGB weight. */
ccl_device_inline float photon_caustics_wavelength_random(const uint round, const uint seed)
{
  const uint bits = reverse_integer_bits(reversed_bit_owen(round, hash_uint2(seed, 0xa54ff53au)));
  /* Exact 24-bit conversion: uint_to_float_excl can round a boundary sample
   * into the preceding stratum. Keep [0,1) without shrinking every interval. */
  return float(bits >> 8) * (1.0f / 16777216.0f);
}

constexpr uint PHOTON_WAVELENGTH_GROUPS = 8;

/* Stable emission slots form shuffled blocks: exactly one photon per group in
 * every complete block. No per-photon field, atomic counter or extra map is needed.
 * groups is a power of two (one selects the diagnostic single-wavelength mode). */
ccl_device_inline uint photon_wavelength_group(const uint index,
                                               const uint iteration,
                                               const uint groups)
{
  return (index + hash_uint2(index / groups, iteration ^ 0x510e527fu)) & (groups - 1u);
}

ccl_device_inline uint photon_camera_wavelength_group(const float random, const uint groups)
{
  return min(uint(random * float(groups)), groups - 1u);
}

/* Stratify the native wavelength CDF, not RGB channels. Jitter changes every
 * round; these are not eight fixed wavelengths. Camera group selection is
 * uniform and photon flux remains divided by ALL emission attempts. */
ccl_device_inline float photon_group_wavelength_random(const float jitter,
                                                       const uint group,
                                                       const uint groups)
{
  /* Preserve each stratum's open upper endpoint after float rounding. */
  return min((float(group) + jitter) / float(groups),
             float(group + 1u) / float(groups) - 1.0f / 16777216.0f);
}

/* Achromatic suffixes are valid at every wavelength. Spectral suffixes require
 * an exact group match when the camera prefix is spectral; compensate the 1/G
 * subset only in that case. The native RGB response is applied separately once. */
ccl_device_inline float photon_wavelength_merge_weight(const bool spectral_photon,
                                                       const int camera_group,
                                                       const uint photon_group,
                                                       const uint groups)
{
  return !spectral_photon || camera_group < 0 ? 1.0f :
         (uint(camera_group) == photon_group ? float(groups) : 0.0f);
}

CCL_NAMESPACE_END
