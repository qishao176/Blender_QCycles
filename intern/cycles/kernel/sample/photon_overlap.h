/* SPDX-FileCopyrightText: 2026 Blender Foundation
 * SPDX-License-Identifier: Apache-2.0 */
#pragma once

#include "util/math.h"

CCL_NAMESPACE_BEGIN

/* Three trial multipliers of each ORIGINAL ellipse, not a shared world-space
 * radius. The model changes bandwidth only; transported flux is never averaged.
 * For an Epanechnikov ellipse with support matrix S, its second moment is S/6.
 * The product integral of two moment-matched Gaussians is proportional to
 * exp(-3 d^T (s^2 S_i + S_j)^-1 d) / sqrt(det(s^2 S_i + S_j)).
 * This is a bounded overlap proxy, NOT the exact compact-kernel intersection
 * integral. The published kernel remains the original Epanechnikov ellipse. */
struct PhotonOverlapStatistics {
  float weight[3][3], squared[3][3]; /* trial scale, RGB */
};

ccl_device_inline float3 photon_overlap_covariance(const float4 footprint)
{
  const float det = footprint.x * footprint.z - sqr(footprint.y);
  if (!(det > 0.0f) || !isfinite_safe(det)) {
    return zero_float3();
  }
  return make_float3(footprint.z, -footprint.y, footprint.x) / det;
}

/* Bring the neighbor's covariance into the query's tangent frame. Comparing
 * raw Q coefficients from different receiver normals would confuse orientation. */
ccl_device_inline float3 photon_overlap_transport(const float3 covariance,
                                                  const float3 other_normal,
                                                  const float3 U,
                                                  const float3 V)
{
  float3 A, B;
  make_orthonormals(other_normal, &A, &B);
  const float a = dot(U, A), b = dot(U, B), c = dot(V, A), d = dot(V, B);
  return make_float3(a * a * covariance.x + 2.0f * a * b * covariance.y + b * b * covariance.z,
                     a * c * covariance.x + (a * d + b * c) * covariance.y + b * d * covariance.z,
                     c * c * covariance.x + 2.0f * c * d * covariance.y + d * d * covariance.z);
}

ccl_device_inline float photon_overlap_weight(const float3 own,
                                              const float3 other,
                                              const float2 delta,
                                              const float scale_squared)
{
  const float3 sum = scale_squared * own + other;
  const float det = sum.x * sum.z - sqr(sum.y);
  const float own_det = own.x * own.z - sqr(own.y);
  if (!(det > 0.0f && own_det > 0.0f && other.x * other.z > sqr(other.y)) || !isfinite_safe(det)) {
    return 0.0f;
  }
  const float q = (sum.z * sqr(delta.x) - 2.0f * sum.y * delta.x * delta.y +
                   sum.x * sqr(delta.y)) /
                  det;
  /* The Minkowski sum of the two support ellipses is contained in
   * 2*(s^2 S_i + S_j). Outside it even the compact supports cannot overlap. */
  if (!isfinite_safe(q) || q >= 2.0f) {
    return 0.0f;
  }
  return 2.0f * sqrtf(own_det / det) * expf(-3.0f * max(q, 0.0f));
}

ccl_device_inline void photon_overlap_add(ccl_private PhotonOverlapStatistics *statistics,
                                          const float3 own,
                                          const float3 other,
                                          const float2 delta,
                                          const float3 relative_power,
                                          const float compatibility,
                                          const float maximum,
                                          const float selection_weight = 1.0f)
{
  const float powers[3] = {relative_power.x, relative_power.y, relative_power.z};
  const float scale_squared[3] = {1.0f, maximum, sqr(maximum)};
  for (int trial = 0; trial < 3; ++trial) {
    const float overlap = compatibility *
                          photon_overlap_weight(own, other, delta, scale_squared[trial]);
    for (int c = 0; c < 3; ++c) {
      const float w = overlap * clamp(powers[c], 0.0f, 1e6f);
      statistics->weight[trial][c] += selection_weight * w;
      statistics->squared[trial][c] += selection_weight * w * w;
    }
  }
}

ccl_device_inline float photon_overlap_effective_count(
    const ccl_private PhotonOverlapStatistics *statistics, const int trial)
{
  const float total = statistics->weight[trial][0] + statistics->weight[trial][1] +
                      statistics->weight[trial][2];
  float count = FLT_MAX;
  for (int c = 0; c < 3; ++c) {
    const float weight = statistics->weight[trial][c];
    const float squared = statistics->squared[trial][c];
    if (weight > 1e-4f * total && squared > 0.0f) {
      count = min(count, weight * weight / squared);
    }
  }
  return count == FLT_MAX ? 0.0f : count;
}

ccl_device_inline float photon_overlap_scale(const ccl_private PhotonOverlapStatistics *statistics,
                                             const float target,
                                             const float maximum,
                                             const float gradient_strength,
                                             const float gradient)
{
  float previous_count = photon_overlap_effective_count(statistics, 0);
  if (maximum <= 1.0f || previous_count >= target) {
    return 1.0f;
  }
  float growth = maximum;
  const float half_log_max = .5f * logf(maximum);
  for (int trial = 1; trial < 3; ++trial) {
    const float count = photon_overlap_effective_count(statistics, trial);
    if (count >= target) {
      /* Interpolate the first observed crossing. With unequal powers N_eff
       * need not be monotone; never skip an adequate smaller trial because a
       * later trial admits a dominant rare photon. */
      const float fraction = saturatef((target - previous_count) /
                                       max(count - previous_count, 1e-8f));
      growth = expf((float(trial - 1) + fraction) * half_log_max);
      break;
    }
    previous_count = count;
  }
  if (!(gradient_strength > 0.0f && gradient > 0.0f)) {
    return growth;
  }
  return 1.0f + (growth - 1.0f) / (1.0f + 8.0f * gradient_strength * max(gradient, 0.0f));
}

CCL_NAMESPACE_END
