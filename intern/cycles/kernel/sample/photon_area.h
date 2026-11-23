/* SPDX-FileCopyrightText: 2026 Blender Foundation
 * SPDX-License-Identifier: Apache-2.0 */
#pragma once
#include "kernel/types.h"
#include "util/math.h"
CCL_NAMESPACE_BEGIN

/* Equal-area square coordinates. For ellipses these are mapped through the
 * concentric disk map, not interpreted as the shader's UV coordinates. */
ccl_device_inline float2 photon_area_sample(const ccl_global float *cdf,
                                            const float2 random,
                                            ccl_private uint *cell,
                                            ccl_private float *pdf)
{
  constexpr int resolution = PHOTON_AREA_GUIDE_RESOLUTION;
  constexpr int cells = PHOTON_AREA_GUIDE_CELLS;
  if (!cdf) {
    *cell = min(int(random.x * resolution), resolution - 1) +
            resolution * min(int(random.y * resolution), resolution - 1);
    *pdf = 1.0f;
    return random;
  }
  int lo = 0, hi = cells - 1;
  while (lo < hi) {
    const int mid = (lo + hi) / 2;
    if (random.x < cdf[mid])
      hi = mid;
    else
      lo = mid + 1;
  }
  *cell = lo;
  const float previous = lo ? cdf[lo - 1] : 0.0f;
  const float mass = cdf[lo] - previous;
  *pdf = mass * cells;
  const float u = clamp((random.x - previous) / mass, 0.0f, 1.0f);
  return make_float2((float(lo % resolution) + u) / resolution,
                     (float(lo / resolution) + random.y) / resolution);
}
CCL_NAMESPACE_END
