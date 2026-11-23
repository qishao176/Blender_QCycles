/* SPDX-FileCopyrightText: 2026 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */
#pragma once

#include "kernel/types.h"
#include "util/math.h"

CCL_NAMESPACE_BEGIN

/* Fixed-size, per-light models. Devices fit independent proposals; photon power
 * always uses the local proposal PDF. Film merging remains sample-weighted.
 * Small sufficient-statistic buffers are transferred once per 32 emission rounds.
 * The current trainer is weighted online EM, not the paper's Adam optimizer. */
inline void photon_guiding_initialize(KernelPhotonGuide *guide, const int lights)
{
  for (int light = 0; light < lights; ++light) {
    for (int i = 0; i < PHOTON_GUIDE_COMPONENTS; ++i) {
      auto &g = guide[light * PHOTON_GUIDE_COMPONENTS + i];
      g.mean_sigma = make_float4(
          (i & 1) ? .25f : -.25f, (i & 2) ? .25f : -.25f, (i & 4) ? .25f : -.25f, .5f);
      g.weight = make_float4(1.0f / PHOTON_GUIDE_COMPONENTS, 0, 0, 0);
    }
  }
}

inline bool photon_guiding_update(KernelPhotonGuide *guide,
                                  KernelPhotonGuideStatistics *statistics,
                                  const int lights)
{
  bool trained = false;
  for (int light = 0; light < lights; ++light) {
    const int offset = light * PHOTON_GUIDE_COMPONENTS;
    double total = 0.0, count = 0.0;
    for (int i = 0; i < PHOTON_GUIDE_COMPONENTS; ++i) {
      total += statistics[offset + i].position_weight.w;
      count += statistics[offset + i].second_count.y;
    }
    if (!(total > 0.0) || !std::isfinite(total) || count < 16.0) {
      /* Keep rare feedback across epochs until this light has enough observations. */
      continue;
    }
    float weight_sum = 0.0f;
    for (int i = 0; i < PHOTON_GUIDE_COMPONENTS; ++i) {
      auto &g = guide[offset + i];
      const auto &s = statistics[offset + i];
      if (s.position_weight.w > 0.0f) {
        const float3 mean = make_float3(s.position_weight) / s.position_weight.w;
        const float second = s.second_count.x / s.position_weight.w;
        if (isfinite_safe(mean) && isfinite_safe(second)) {
          const float3 previous = make_float3(g.mean_sigma);
          const float3 blended = .5f * (previous + mean);
          const float moment = .5f * (len_squared(previous) + 3 * sqr(g.mean_sigma.w) +
                                      max(second, len_squared(mean)));
          const float variance = max((moment - len_squared(blended)) / 3.0f, 1e-6f);
          g.mean_sigma = make_float4(blended, clamp(sqrtf(variance), .001f, 2.0f));
        }
      }
      g.weight.x = max(.5f * (g.weight.x + float(s.position_weight.w / total)), .001f);
      weight_sum += g.weight.x;
    }
    for (int i = 0; i < PHOTON_GUIDE_COMPONENTS; ++i) {
      guide[offset + i].weight.x /= weight_sum;
      statistics[offset + i] = {};
    }
    /* The first component's spare weight.y marks this light as ready for proposal sampling.
     * Keep collecting sparse feedback without changing the light's native emission proposal. */
    guide[offset].weight.y = 1.0f;
    trained = true;
  }
  return trained;
}

/* Position proposals are trained from actual shader-evaluated emitted flux and
 * camera-visible flux. Every cell keeps at least 25% of uniform probability.
 * Each round's emitted statistics are committed only after texture residency is
 * complete; camera statistics are collected only from the published map. */
inline void photon_area_guiding_initialize(float *cdf, const int lights)
{
  for (int i = 0; i < lights * PHOTON_AREA_GUIDE_CELLS; ++i) {
    cdf[i] = float(i % PHOTON_AREA_GUIDE_CELLS + 1) / PHOTON_AREA_GUIDE_CELLS;
  }
}
inline bool photon_area_guiding_update(float *cdf, float4 *stats, const int lights)
{
  bool trained = false;
  for (int light = 0; light < lights; ++light) {
    const int offset = light * PHOTON_AREA_GUIDE_CELLS;
    double emitted = 0, visible = 0, count = 0, visible_count = 0;
    for (int i = 0; i < PHOTON_AREA_GUIDE_CELLS; ++i) {
      emitted += stats[offset + i].x;
      count += stats[offset + i].y;
      visible += stats[offset + i].z;
      visible_count += stats[offset + i].w;
    }
    if (!(emitted > 0) || !std::isfinite(emitted) || !std::isfinite(visible) || count < 16) {
      continue;
    }
    const double visible_fraction = visible > 0 && visible_count >= 16 ? .5 : 0.0;
    double cumulative = 0;
    float previous = 0;
    for (int i = 0; i < PHOTON_AREA_GUIDE_CELLS; ++i) {
      const float old_cdf = cdf[offset + i];
      const double target = (1 - visible_fraction) * stats[offset + i].x / emitted +
                            (visible_fraction ? visible_fraction * stats[offset + i].z / visible :
                                                0);
      cumulative += .5 * (old_cdf - previous) +
                    .5 * (.25 / PHOTON_AREA_GUIDE_CELLS + .75 * target);
      previous = old_cdf;
      cdf[offset + i] = float(cumulative);
      stats[offset + i] = zero_float4();
    }
    cdf[offset + PHOTON_AREA_GUIDE_CELLS - 1] = 1.0f;
    trained = true;
  }
  return trained;
}
CCL_NAMESPACE_END
