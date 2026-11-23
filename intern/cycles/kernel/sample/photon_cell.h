/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: Apache-2.0 */
#pragma once

#include "util/math.h"

CCL_NAMESPACE_BEGIN

/* Fixed point tent deposits give order-independent occupancy sums. These are
 * query statistics, not photon energy. Up to 4M deposits cannot overflow uint. */
constexpr uint PHOTON_CELL_MASS = 256;
constexpr uint PHOTON_CELL_PROBES = 32;

ccl_device_inline float photon_cell_weight(const float3 f, const uint corner)
{
  return ((corner & 1) ? f.x : 1 - f.x) * ((corner & 2) ? f.y : 1 - f.y) *
         ((corner & 4) ? f.z : 1 - f.z);
}

ccl_device_inline void photon_cell_weights(const float3 f, ccl_private uint *weights)
{
  uint sum = 0, largest = 0;
  float maximum = -1;
  for (uint c = 0; c < 8; c++) {
    const float w = photon_cell_weight(f, c);
    weights[c] = uint(floorf(w * PHOTON_CELL_MASS));
    sum += weights[c];
    if (w > maximum) {
      maximum = w;
      largest = c;
    }
  }
  weights[largest] = uint(int(weights[largest]) + int(PHOTON_CELL_MASS) - int(sum));
}

ccl_device_inline float photon_cell_pilot_radius(const float occupancy,
                                                 const float cell_width,
                                                 const float minimum,
                                                 const float maximum,
                                                 const float target)
{
  /* Occupancy is a proposal for query work, not an area/volume density estimate.
   * A compatible-neighbor check refines it after the first query. */
  return clamp(cell_width * sqrtf(4 * target / max(occupancy, 1.0f)), minimum, maximum);
}

ccl_device_inline float photon_cell_refine_radius(const float radius,
                                                  const float maximum,
                                                  const float compatible_mass,
                                                  const float target)
{
  const float t = saturatef(1 - compatible_mass / max(target, 1.0f));
  return mix(radius, maximum, t * t * (3 - 2 * t));
}

CCL_NAMESPACE_END
