/* SPDX-FileCopyrightText: 2026 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */
#pragma once

#include "kernel/types.h"
#include "util/math.h"

#include <algorithm>


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

/* Fit conditional emission directions from pairs of actual, camera-observed surface paths.
 * The model never stores radiance. A hash collision can only make a worse proposal:
 * transport and LPE classification still use the newly traced path. */
inline int photon_field_guiding_fit(KernelPhotonField *fields,
                                    const KernelPhotonFieldObservation *observations,
                                    const int lights,
                                    const float radius,
                                    const float caster_radius)
{
  const float distance = clamp(6.0f * radius / max(caster_radius, 1e-8f), .005f, .15f);
  const float limit = sqr(distance);
  int fitted = 0;
  for (int light = 0; light < lights; ++light) {
    const auto *samples = observations + size_t(light) * PHOTON_FIELD_OBSERVATIONS;
    auto *model = fields + size_t(light) * PHOTON_FIELD_COMPONENTS;
    struct Candidate {
      int index, partner, neighbors;
      float error, priority;
    };
    Candidate candidates[PHOTON_FIELD_OBSERVATIONS];
    int observed = 0;
    for (int i = 0; i < PHOTON_FIELD_OBSERVATIONS; ++i) {
      if (samples[i].occupied) {
        candidates[observed++] = {i, -1, 0, FLT_MAX, 0.0f};
      }
    }
    for (int i = 0; i < observed; ++i) {
      const auto &a = samples[candidates[i].index];
      for (int j = i + 1; j < observed; ++j) {
        const auto &b = samples[candidates[j].index];
        if (a.iteration == b.iteration || a.object != b.object ||
            a.route != b.route || a.band != b.band || a.time_bin != b.time_bin)
        {
          continue;
        }
        const float uv_distance = len_squared(a.uv - b.uv);
        if (uv_distance < .0001f || uv_distance > .25f) {
          continue;
        }
        const float first = len_squared(float3(a.first) - float3(b.first));
        const float previous = len_squared(float3(a.previous) - float3(b.previous));
        const float landing = len_squared(float3(a.receiver) - float3(b.receiver));
        if (max(first, max(previous, landing)) >= limit) {
          continue;
        }
        const float3 da = float3(a.receiver) - float3(a.previous);
        const float3 db = float3(b.receiver) - float3(b.previous);
        const float cosine = dot(da, db);
        if (cosine <= 0.0f || sqr(cosine) < .75f * len_squared(da) * len_squared(db)) {
          continue;
        }
        const float error = first + previous + landing;
        candidates[i].neighbors++;
        candidates[j].neighbors++;
        if (error < candidates[i].error) {
          candidates[i].partner = candidates[j].index;
          candidates[i].error = error;
        }
        if (error < candidates[j].error) {
          candidates[j].partner = candidates[i].index;
          candidates[j].error = error;
        }
      }
    }
    /* Favor undercovered routes with corroboration. Isolated hits cannot train;
     * a dense patch should not consume all eight proposals just because it was seen first. */
    for (int i = 0; i < observed; ++i) {
      Candidate &c = candidates[i];
      c.priority = float(min(c.neighbors, 4)) / float(max(c.neighbors, 4));
    }
    std::sort(
        candidates, candidates + observed, [](const Candidate &a, const Candidate &b) {
          return a.priority == b.priority ? a.index < b.index : a.priority > b.priority;
        });
    int selected[PHOTON_FIELD_COMPONENTS], count = 0;
    for (int i = 0; i < observed; ++i) {
      const Candidate &c = candidates[i];
      if (count == PHOTON_FIELD_COMPONENTS || c.partner < 0) {
        break;
      }
      const auto &a = samples[c.index];
      bool covered = false;
      for (int k = 0; k < count; ++k) {
        const auto &b = samples[selected[k]];
        if (a.route == b.route && a.object == b.object && a.band == b.band &&
            a.time_bin == b.time_bin &&
            len_squared(float3(a.receiver) - float3(b.receiver)) < limit &&
            len_squared(float3(a.first) - float3(b.first)) < limit &&
            len_squared(a.uv - b.uv) < .0625f)
        {
          covered = true;
          break;
        }
      }
      if (covered) {
        continue;
      }
      const auto &b = samples[c.partner];
      auto &field = model[count];
      const float sigma = max(.25f * distance, .5f * len(float3(a.first) - float3(b.first)));
      field.first_sigma = make_float4(float3(a.first), sigma);
      field.last = make_float4(float3(b.first), 0.0f);
      field.uv = make_float4(a.uv.x, a.uv.y, b.uv.x, b.uv.y);
      field.band = a.band;
      field.time_bin = a.time_bin;
      selected[count++] = c.index;
    }
    for (int k = count; k < PHOTON_FIELD_COMPONENTS; ++k) {
      model[k] = {};
    }
    fitted += count;
  }
  return fitted;
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
