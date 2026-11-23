/* SPDX-FileCopyrightText: 2026 Blender Foundation
 * SPDX-License-Identifier: Apache-2.0 */
#pragma once

#include "util/math.h"

CCL_NAMESPACE_BEGIN

/* A bounded pilot-density bandwidth heuristic, not an image-space denoiser or
 * an optimal APPM estimator. All offsets are normalized by the pilot radius.
 * Track RGB independently so equal-luminance color boundaries remain visible. */
struct PhotonRadiusStatistics {
  int count;
  float weight[3], weight_squared[3], second[3];
  float3 first[3];
};

/* Evidence from the original differential kernels, never from kernels being
 * resized by other workers. This only controls bandwidth, not photon power. */
struct PhotonEllipseStatistics {
  float weight, weight_squared, second, crossing;
  float3 first;
};

struct PhotonEllipseFrame {
  float3 axis;
  float anisotropy, radius, peak;
};

ccl_device_inline PhotonEllipseFrame photon_ellipse_frame(const float4 footprint,
                                                          const float3 normal)
{
  PhotonEllipseFrame frame = {};
  const float trace = footprint.x + footprint.z;
  const float det = footprint.x * footprint.z - sqr(footprint.y);
  if (!(footprint.w > 0.0f && trace > 0.0f && det > 0.0f) ||
      !isfinite_safe(trace) || !isfinite_safe(det) || !isfinite_safe(footprint.w))
  {
    return frame;
  }
  const float gap = hypotf(footprint.x - footprint.z, 2.0f * footprint.y);
  const float lo = .5f * (trace - gap);
  float2 axis = make_float2(-footprint.y, footprint.x - lo);
  if (len_squared(axis) < 1e-20f) {
    axis = footprint.x <= footprint.z ? make_float2(1, 0) : make_float2(0, 1);
  }
  else {
    axis = normalize(axis);
  }
  float3 U, V;
  make_orthonormals(normal, &U, &V);
  frame.axis = axis.x * U + axis.y * V;
  frame.anisotropy = saturatef(gap / trace);
  frame.radius = sqrtf(2.0f * M_1_PI_F / footprint.w);
  frame.peak = footprint.w;
  return frame;
}

ccl_device_inline void photon_ellipse_add(ccl_private PhotonEllipseStatistics *s,
                                          const PhotonEllipseFrame own,
                                          const float4 footprint,
                                          const float3 normal,
                                          const float3 query_minus_center,
                                          const float relative_power,
                                          const float selection_weight = 1.0f)
{
  if (!(own.peak > 0.0f && footprint.w > 0.0f && relative_power > 0.0f)) {
    return;
  }
  float3 U, V;
  make_orthonormals(normal, &U, &V);
  const float u = dot(query_minus_center, U), v = dot(query_minus_center, V);
  const float qu = footprint.x * u + footprint.y * v;
  const float qv = footprint.y * u + footprint.z * v;
  const float shape = 1.0f - (u * qu + v * qv);
  if (!(shape > 1e-6f) || !isfinite_safe(shape)) {
    return;
  }
  const PhotonEllipseFrame other = photon_ellipse_frame(footprint, normal);
  if (!(other.peak > 0.0f)) {
    return;
  }
  const float w = clamp(relative_power * (footprint.w / own.peak) * shape, 0.0f, 1e6f);
  if (!isfinite_safe(w)) {
    return;
  }
  /* Analytic log-density derivative of the Epanechnikov ellipse. Bound only
   * the pilot derivative near its support edge, where log derivatives diverge. */
  float3 gradient = (-2.0f * own.radius / shape) * (qu * U + qv * V);
  gradient *= min(1.0f, 8.0f / max(len(gradient), 1e-20f));
  const float disagreement = own.anisotropy * other.anisotropy *
                            max(1.0f - sqr(dot(own.axis, other.axis)), 0.0f);
  s->weight += selection_weight * w;
  s->weight_squared += selection_weight * w * w;
  s->first += selection_weight * w * gradient;
  s->second += selection_weight * w * len_squared(gradient);
  s->crossing += selection_weight * w * disagreement;
}

ccl_device_inline float photon_ellipse_evidence(const ccl_private PhotonEllipseStatistics *s)
{
  if (!s || !(s->weight > 0.0f && s->weight_squared > 0.0f)) {
    return 0.0f;
  }
  const float count = sqr(s->weight) / s->weight_squared;
  if (count < 4.0f) {
    return 0.0f;
  }
  const float3 mean = s->first / s->weight;
  const float variance = max(s->second / s->weight - len_squared(mean), 0.0f);
  const float signal = max(len(mean) - 2.0f * sqrtf(variance / count), 0.0f);
  /* Crossed elongated kernels can have zero centroid/gradient at their shared
   * peak. Orientation disagreement protects that peak without treating a broad,
   * consistently oriented footprint as an edge. Few overlaps are low confidence. */
  return saturatef((count - 3.0f) / 5.0f) *
         (signal + 2.0f * s->crossing / s->weight);
}

ccl_device_inline void photon_radius_add(ccl_private PhotonRadiusStatistics *s,
                                         const float3 offset,
                                         const float3 power,
                                         const float selection_weight = 1.0f)
{
  ++s->count;
  const float weights[3] = {power.x, power.y, power.z};
  for (int c = 0; c < 3; ++c) {
    /* This only bounds pilot statistics, never transported photon energy. */
    const float w = clamp(weights[c], 0.0f, 1e6f);
    s->weight[c] += selection_weight * w;
    s->weight_squared[c] += selection_weight * w * w;
    s->first[c] += selection_weight * w * offset;
    s->second[c] += selection_weight * w * len_squared(offset);
  }
}

ccl_device_inline float photon_radius_gradient(
    const ccl_private PhotonRadiusStatistics *s,
    const bool volume,
    const ccl_private PhotonEllipseStatistics *ellipse = nullptr)
{
  const float dimension = volume ? 3.0f : 2.0f;
  float gradient = 0.0f;
  for (int c = 0; c < 3; ++c) {
    if (!(s->weight[c] > 0.0f && s->weight_squared[c] > 0.0f)) {
      continue;
    }
    const float effective_count = sqr(s->weight[c]) / s->weight_squared[c];
    if (effective_count < 4.0f) {
      continue;
    }
    const float3 mean = s->first[c] / s->weight[c];
    const float variance = max(s->second[c] / s->weight[c] - len_squared(mean), 0.0f);
    /* Remove a two-standard-error noise floor before interpreting displacement
     * of the flux centroid as a density/color gradient. This is a conservative
     * heuristic, not a confidence interval for correlated photon paths. */
    const float signal = max(len(mean) - 2.0f * sqrtf(variance / effective_count), 0.0f);
    gradient = max(gradient, (dimension + 2.0f) * signal);
  }
  if (!volume) {
    gradient = max(gradient, photon_ellipse_evidence(ellipse));
  }
  return gradient;
}

ccl_device_inline float photon_radius_scale(
    const ccl_private PhotonRadiusStatistics *s,
    const float target,
    const float maximum,
    const float gradient_strength,
    const bool volume,
    const ccl_private PhotonEllipseStatistics *ellipse = nullptr,
    const float statistical_scale = 0.0f,
    const ccl_private PhotonRadiusStatistics *local_gradient = nullptr)
{
  const float dimension = volume ? 3.0f : 2.0f;
  /* Correct for the actual sampled area/volume. Enlarging a pilot region must
   * not be mistaken for increasing the local photon density. Zero retains the
   * footprint-local policy used before the independent pilot was enabled. */
  const float pilot_scale = statistical_scale > 0.0f ? statistical_scale : maximum;
  const float growth = clamp(
      pilot_scale * powf(target / max(float(s->count), 1.0f), 1.0f / dimension), 1.0f, maximum);
  if (growth <= 1.0f || !(gradient_strength > 0.0f)) {
    return growth;
  }
  return 1.0f +
         (growth - 1.0f) / (1.0f + 8.0f * gradient_strength *
                                       photon_radius_gradient(
                                           local_gradient ? local_gradient : s, volume, ellipse));
}

CCL_NAMESPACE_END
