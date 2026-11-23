/* SPDX-FileCopyrightText: 2026 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#pragma once

#include "util/math.h"

CCL_NAMESPACE_BEGIN

/* Directional marginal of an isotropic spatial Gaussian, not its spatial PDF.
 * Online Photon Guiding with 3D Gaussians for Caustics Rendering (2024), Eq. 13-14.
 * A sampled 3D point is projected onto the unit sphere about the light origin.
 * mu and sigma can be expressed in any shared, uniformly scaled coordinate frame. */
ccl_device_inline float photon_gaussian_direction_pdf(const float3 delta,
                                                      const float sigma,
                                                      const float3 direction)
{
  if (!(sigma > 0.0f)) {
    return 0.0f;
  }
  const float3 v = delta / sigma;
  const float a = dot(v, direction);
  const float perpendicular_squared = len_squared(v - a * direction);
  constexpr float normalization = 0.06349363593424097f; /* (2*pi)^(-3/2) */
  if (a >= 0.0f) {
    return normalization * expf(-0.5f * perpendicular_squared) *
           (a * expf(-0.5f * a * a) +
            1.2533141373155f * (1.0f + a * a) * erfcf(-a * 0.70710678118655f));
  }
  /* The two terms of Eq. 13 cancel in the back hemisphere. Evaluate the
   * scaled radial integral in double precision there, and use its asymptotic
   * expansion far from the mean. This branch is cold for concentrated models. */
  const double b = -double(a);
  double radial;
  if (b < 10.0) {
    radial = (1.0 + b * b) * 1.2533141373155003 * ::exp(0.5 * b * b) *
                 erfc(b * 0.7071067811865475) -
             b;
  }
  else {
    double term = 1.0, series = 1.0;
    for (int n = 1; n <= 12; ++n) {
      term *= -double((2 * n + 2) * (2 * n + 1)) / (2.0 * n * b * b);
      series += term;
    }
    radial = 2.0 * series / (b * b * b);
  }
  return normalization * expf(-0.5f * (perpendicular_squared + a * a)) * float(radial);
}

/* Project onto the plane normal to direction (Sun emission).
 * The result is a normalized density per unit area on that plane. */
ccl_device_inline float photon_gaussian_plane_pdf(const float3 delta,
                                                  const float sigma,
                                                  const float3 direction)
{
  const float3 tangent = delta - direction * dot(delta, direction);
  return expf(-0.5f * len_squared(tangent) / sqr(sigma)) * (0.5f * M_1_PI_F / sqr(sigma));
}

/* Four independent uniform variates; Box-Muller gives three standard normals.
 * Finite endpoint handling is shared by sampling and the numerical tests. */
ccl_device_inline float3 photon_gaussian_sample(const float4 random)
{
  const float r0 = sqrtf(-2.0f * logf(max(1.0f - random.x, FLT_MIN)));
  const float r1 = sqrtf(-2.0f * logf(max(1.0f - random.z, FLT_MIN)));
  const float a0 = M_2PI_F * random.y, a1 = M_2PI_F * random.w;
  return make_float3(r0 * cosf(a0), r0 * sinf(a0), r1 * cosf(a1));
}

CCL_NAMESPACE_END
