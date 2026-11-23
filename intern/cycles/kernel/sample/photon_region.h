/* SPDX-FileCopyrightText: 2026 Blender Foundation
 * SPDX-License-Identifier: Apache-2.0 */
#pragma once

#include "util/math.h"

CCL_NAMESPACE_BEGIN

/* Experimental density bandwidth estimation. Flux is an integration weight,
 * never treated as an irradiance observation. All offsets use the independent
 * statistical radius; RGB moments protect equal-luminance color boundaries. */
struct PhotonRegionStatistics {
  float weight[3], squared[3];
  float2 first[3];
  float3 second[3]; /* xx, xy, yy */
};

ccl_device_inline void photon_region_add(ccl_private PhotonRegionStatistics *s,
                                         const float2 offset,
                                         const float3 power,
                                         const float compatibility)
{
  const float rgb[3] = {power.x, power.y, power.z};
  for (int c = 0; c < 3; ++c) {
    const float w = compatibility * clamp(rgb[c], 0.0f, 1e6f);
    s->weight[c] += w;
    s->squared[c] += w * w;
    s->first[c] += w * offset;
    s->second[c] += w * make_float3(sqr(offset.x), offset.x * offset.y, sqr(offset.y));
  }
}

/* Fit dN/dP in the interface tangent frame from the already traced ray bundle.
 * A degenerate bundle has no trustworthy derivative; it is not a flat surface.
 * No intersections or material evaluations are added here. */
ccl_device_inline bool photon_region_normal_derivative(const float3 normal,
                                                       const float3 dP[4],
                                                       const float3 dN[4],
                                                       ccl_private float4 *derivative)
{
  float3 U, V;
  make_orthonormals(normal, &U, &V);
  float xx = 0, xy = 0, yy = 0;
  float4 cross = zero_float4();
  for (int i = 0; i < 4; ++i) {
    const float2 p = make_float2(dot(dP[i], U), dot(dP[i], V));
    const float2 n = make_float2(dot(dN[i], U), dot(dN[i], V));
    xx += p.x * p.x;
    xy += p.x * p.y;
    yy += p.y * p.y;
    cross += make_float4(n.x * p.x, n.x * p.y, n.y * p.x, n.y * p.y);
  }
  const float det = xx * yy - xy * xy;
  if (!(det > 1e-6f * sqr(xx + yy)) || !isfinite_safe(det)) {
    *derivative = zero_float4();
    return false;
  }
  *derivative = make_float4(cross.x * yy - cross.y * xy,
                            cross.y * xx - cross.x * xy,
                            cross.z * yy - cross.w * xy,
                            cross.w * xx - cross.z * xy) /
                det;
  return isfinite_safe(*derivative);
}

ccl_device_inline float3 photon_region_predict_normal(const float3 normal,
                                                      const float4 derivative,
                                                      const float3 displacement)
{
  float3 U, V;
  make_orthonormals(normal, &U, &V);
  const float x = dot(displacement, U), y = dot(displacement, V);
  return safe_normalize(normal + (derivative.x * x + derivative.y * y) * U +
                        (derivative.z * x + derivative.w * y) * V);
}

ccl_device_inline float photon_region_continuity(const float3 position_a,
                                                 const float3 normal_a,
                                                 const float4 derivative_a,
                                                 const bool valid_a,
                                                 const float3 position_b,
                                                 const float3 normal_b,
                                                 const float4 derivative_b,
                                                 const bool valid_b,
                                                 const float tolerance_cos)
{
  const float3 displacement = position_b - position_a;
  const float3 predicted_b = valid_a ? photon_region_predict_normal(
                                           normal_a, derivative_a, displacement) :
                                       normal_a;
  const float3 predicted_a = valid_b ? photon_region_predict_normal(
                                           normal_b, derivative_b, -displacement) :
                                       normal_b;
  const float residual = max(1.0f - clamp(dot(predicted_b, normal_b), -1.0f, 1.0f),
                             1.0f - clamp(dot(predicted_a, normal_a), -1.0f, 1.0f));
  /* Smoothly downweight discontinuities, not fixed-angle bins on a curved mesh.
   * With missing derivatives this conservatively falls back to normal agreement. */
  const float t = residual / max(1.0f - tolerance_cos, 1e-6f);
  return 1.0f / (1.0f + t * t);
}

/* Covariance addition changes both axes independently, unlike scalar footprint
 * scaling. (I + T)^-1 is positive definite; normalizing the resulting ellipse
 * preserves each photon's integral, including rough-footprint sign tagging.
 * This is a biased bandwidth heuristic, not a gradient-domain transport solver. */
ccl_device_inline float4 photon_region_footprint(const float4 footprint,
                                                 const ccl_private PhotonRegionStatistics *s,
                                                 const float pilot_radius,
                                                 const float support_limit,
                                                 const float target,
                                                 const float gradient_strength,
                                                 const bool anisotropic,
                                                 const float ellipse_guard)
{
  const float det_q = footprint.x * footprint.z - sqr(footprint.y);
  if (!(det_q > 0.0f) || !isfinite_safe(det_q)) {
    return footprint;
  }
  const float3 base = make_float3(footprint.z, -footprint.y, footprint.x) / det_q;
  const float base_hi = .5f * (base.x + base.z + hypotf(base.x - base.z, 2.0f * base.y));
  float effective = FLT_MAX;
  float3 tensor = zero_float3();
  float total_power = 0.0f;
  for (int c = 0; c < 3; ++c) {
    total_power += s->weight[c];
  }
  for (int c = 0; c < 3; ++c) {
    if (!(s->weight[c] > 1e-4f * total_power && s->squared[c] > 0.0f)) {
      continue;
    }
    const float n = sqr(s->weight[c]) / s->squared[c];
    effective = min(effective, n);
    if (!anisotropic || !(gradient_strength > 0.0f) || n < 4.0f) {
      continue;
    }
    const float2 mean = s->first[c] / s->weight[c];
    const float3 moment = s->second[c] / s->weight[c];
    const float3 covariance = moment - make_float3(sqr(mean.x), mean.x * mean.y, sqr(mean.y));
    const float variance = max(covariance.x + covariance.z, 0.0f);
    const float magnitude = len(mean);
    const float signal = 4.0f * max(magnitude - 2.0f * sqrtf(variance / n), 0.0f);
    if (magnitude > 1e-8f) {
      const float2 axis = mean / magnitude;
      tensor += signal * make_float3(sqr(axis.x), axis.x * axis.y, sqr(axis.y));
    }
    /* A centered thin band has zero first moment. Its regional second moment
     * supplies the missing transverse direction. Remove a sampling noise floor. */
    const float gap = hypotf(covariance.x - covariance.z, 2.0f * covariance.y);
    const float trace = max(covariance.x + covariance.z, 1e-10f);
    const float structure = max(gap / trace - 2.0f / sqrtf(n), 0.0f);
    if (gap > 1e-8f && structure > 0.0f) {
      const float hi = .5f * (trace + gap);
      tensor += (2.0f * structure / gap) *
                make_float3(hi - covariance.x, -covariance.y, hi - covariance.z);
    }
  }
  if (effective == FLT_MAX) {
    return footprint;
  }
  /* N_eff measures uncertainty in a weighted density estimate, not the number
   * of independent paths. It is deliberately a heuristic for guided samples. */
  const float desired = sqr(pilot_radius) * target / max(effective, 1.0f);
  const float extra = min(max(desired - sqrtf(base.x * base.z - sqr(base.y)), 0.0f),
                          max(sqr(support_limit) - base_hi, 0.0f));
  if (!(extra > 0.0f)) {
    return footprint;
  }
  tensor *= 8.0f * gradient_strength;
  const float guard = 1.0f + (anisotropic ? 8.0f * gradient_strength * ellipse_guard : 0.0f);
  const float xx = guard + tensor.x, xy = tensor.y, yy = guard + tensor.z;
  const float det_t = xx * yy - xy * xy;
  const float3 covariance = base + (extra / det_t) * make_float3(yy, -xy, xx);
  const float det = covariance.x * covariance.z - sqr(covariance.y);
  if (!(det > 0.0f) || !isfinite_safe(det)) {
    return footprint;
  }
  return make_float4(covariance.z / det,
                     -covariance.y / det,
                     covariance.x / det,
                     copysignf(2.0f * M_1_PI_F / sqrtf(det), footprint.w));
}

CCL_NAMESPACE_END
