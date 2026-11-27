/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: Apache-2.0 */
#pragma once

#include "util/hash.h"

CCL_NAMESPACE_BEGIN

/* The first word is the existing emission counter. The other words are used only
 * by the viewport probe on selected maps, sharing the normal map publication fence. */
enum PhotonMapCounter {
  PHOTON_MAP_STORED,
  PHOTON_PROBE_ENABLED,
  PHOTON_PROBE_SURFACES,
  PHOTON_PROBE_ISOLATED,
  PHOTON_PROBE_WIDER_HELPED,
  PHOTON_PROBE_IN_FRAME,
  PHOTON_PROBE_SUBPIXEL,
  PHOTON_PROBE_SUBPIXEL_RADIUS_LIMITED,
  PHOTON_PROBE_PIXEL_SCALE_SUM,
  PHOTON_PROBE_VOLUME_STORED,
  PHOTON_MAP_COUNTERS
};

/* After the first four rounds, visit one round per block of 16. Across sixteen
 * blocks, visit every low-four-bit offset exactly once. A fixed offset aliases
 * the bit-reversed wavelength jitter of dispersive photons. All replicas probe
 * the same rounds, using their own transport seeds for the actual emission. */
ccl_device_inline bool photon_probe_round(const uint round, const uint scene_seed)
{
  if (round < 4) {
    return true;
  }
  const uint index = round - 4;
  const uint block = index / 16;
  const uint offset = (block + hash_uint2(block / 16, scene_seed)) & 15u;
  return (index & 15u) == offset;
}

/* The normalized parabolic kernel has peak 2 / (pi * a * b). */
ccl_device_inline float photon_probe_projected_area(const float peak,
                                                    const float light_scale,
                                                    const float tangent_jacobian)
{
  return fabsf(tangent_jacobian) * (2.0f / peak) * sqr(light_scale);
}

CCL_NAMESPACE_END
