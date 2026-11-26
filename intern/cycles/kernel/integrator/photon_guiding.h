/* SPDX-FileCopyrightText: 2026 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */
#pragma once

#include "kernel/sample/lcg.h"
#include "kernel/sample/photon_area.h"
#include "kernel/sample/photon_gaussian.h"
#include "kernel/sample/photon_spectrum.h"
#include "kernel/integrator/photon_time.h"
#include "kernel/types.h"
#include "util/atomic.h"

CCL_NAMESPACE_BEGIN

ccl_device_inline bool photon_guide_light_supported(const ccl_global KernelLight *light)
{
  /* Area emission uses a source-position/direction joint density. Zero spread
   * has a delta direction and is guided in position only. Finite Point/Spot
   * sources retain their target-conditioned native sampler. */
  return light->type == LIGHT_SUN ||
         (light->type == LIGHT_AREA && light->area.tan_half_spread > 0.0f) ||
         ((light->type == LIGHT_POINT || light->type == LIGHT_SPOT) && light->spot.radius == 0.0f);
}

ccl_device_inline float photon_guide_light_fraction(const KernelPhotonCausticsCache cache,
                                                    const int light)
{
  /* Skip the model read during warm-up. Each light becomes eligible after its own fit. */
  return cache.guide && cache.guide_fraction > 0.0f &&
                 cache.guide[light * PHOTON_GUIDE_COMPONENTS].weight.y > 0.0f ?
             cache.guide_fraction :
             0.0f;
}

ccl_device_inline float3 photon_guide_sample(const KernelPhotonCausticsCache cache,
                                             const int light,
                                             ccl_private uint *rng)
{
  const ccl_global KernelPhotonGuide *model = cache.guide + light * PHOTON_GUIDE_COMPONENTS;
  float u = lcg_step_float(rng);
  int component = PHOTON_GUIDE_COMPONENTS - 1;
  for (int i = 0; i < PHOTON_GUIDE_COMPONENTS - 1; ++i) {
    if (u < model[i].weight.x) {
      component = i;
      break;
    }
    u -= model[i].weight.x;
  }
  const float4 g = model[component].mean_sigma;
  const float3 random = lcg_step_float3(rng);
  const float4 xi = make_float4(random, lcg_step_float(rng));
  return make_float3(g) + g.w * photon_gaussian_sample(xi);
}

ccl_device_inline float photon_guide_pdf(const KernelPhotonCausticsCache cache,
                                         const int light,
                                         const float3 position,
                                         const float3 direction,
                                         const bool plane)
{
  float pdf = 0.0f;
  for (int i = 0; i < PHOTON_GUIDE_COMPONENTS; ++i) {
    const ccl_global KernelPhotonGuide *g = &cache.guide[light * PHOTON_GUIDE_COMPONENTS + i];
    const float3 delta = make_float3(g->mean_sigma) - position;
    pdf += g->weight.x * (plane ?
                              photon_gaussian_plane_pdf(delta, g->mean_sigma.w, direction) :
                              photon_gaussian_direction_pdf(delta, g->mean_sigma.w, direction));
  }
  return pdf;
}

/* One eligible slot in every block, changing per map. The optional side buffer can
 * therefore be indexed without an atomic or one extra record per emitted photon. */
ccl_device_inline bool photon_field_slot(const uint index, const uint iteration)
{
  return (index & 15u) == (hash_uint2(index >> 4, iteration) & 15u);
}

ccl_device_inline uint photon_field_band(const KernelPhotonCausticsCache cache, const uint index)
{
  const uint group = photon_wavelength_group(index, cache.iteration, cache.wavelength_groups);
  const float u = photon_group_wavelength_random(
      cache.wavelength_random, group, cache.wavelength_groups);
  return min(uint(u * 8.0f), 7u);
}

/* Choose one local proposal conditional on the source UV before sampling direction.
 * For every possible choice the native/global/local mixture is normalized and keeps
 * full native support. Evaluate exactly that same conditional mixture for the weight. */
ccl_device_inline float photon_field_select(const ccl_global KernelPhotonField *fields,
                                            const float2 uv,
                                            const uint band,
                                            const float time,
                                            ccl_private uint *rng,
                                            ccl_private float4 *gaussian)
{
  float weights[PHOTON_FIELD_COMPONENTS], along[PHOTON_FIELD_COMPONENTS];
  float sum = 0.0f;
  int count = 0;
  const uint time_bin = min(uint(saturatef(time) * 4.0f), 3u);
  for (; count < PHOTON_FIELD_COMPONENTS; ++count) {
    const ccl_global KernelPhotonField *f = &fields[count];
    if (f->first_sigma.w == 0.0f) {
      break;
    }
    weights[count] = 0.0f;
    if ((f->band != ~0u && f->band != band) || f->time_bin != time_bin) {
      continue;
    }
    const float2 a = make_float2(f->uv.x, f->uv.y);
    const float2 ab = make_float2(f->uv.z, f->uv.w) - a;
    const float t = saturatef(dot(uv - a, ab) / max(len_squared(ab), 1e-8f));
    along[count] = t;
    weights[count] = sqr(max(0.0f, 1.0f - 16.0f * len_squared(uv - (a + t * ab))));
    sum += weights[count];
  }
  if (!(sum > 0.0f)) {
    return 0.0f;
  }
  float u = lcg_step_float(rng) * sum;
  for (int i = 0; i < count; ++i) {
    if (u < weights[i]) {
      const float3 mean = mix(
          make_float3(fields[i].first_sigma), make_float3(fields[i].last), along[i]);
      *gaussian = make_float4(mean, fields[i].first_sigma.w);
      return .25f * min(sum, 1.0f);
    }
    u -= weights[i];
  }
  return 0.0f;
}

/* Visibility feedback is collected at camera gathers, including gathers reached
 * through mirror/refraction chains, but not after a diffuse camera event.
 * The existing ancestry flag persists through later specular/transparent events.
 * Reject before claiming the per-map hit: an indirect observation must not block
 * a subsequent eligible observation of the same photon. This filters training
 * only; the caller still evaluates its full lighting/LPE contribution.
 * Emitted photons merely hitting geometry are
 * not training successes. Keep this independent of selected LPE/pass outputs.
 * Uniform thinning bounds atomic traffic; each retained photon contributes once
 * per map, so many camera samples reusing one map do not multiply training weight.
 * Flux already contains the inverse emission proposal and light selection PDF. */
ccl_device_inline void photon_guide_record(KernelGlobals kg,
                                           ConstIntegratorState state,
                                           const KernelPhotonCausticsCache cache,
                                           const uint index,
                                           const ccl_global KernelPhotonCaustic *photon)
{
  if ((!cache.guide && !cache.area_guide_statistics) || !cache.guide_hits ||
      (INTEGRATOR_STATE(state, path, flag) & PATH_RAY_DIFFUSE_ANCESTOR) ||
      is_zero(INTEGRATOR_STATE(state, path, throughput)))
  {
    return;
  }
  const bool global_slot = (hash_uint2(index, cache.iteration) & 15u) == 0;
  const bool field_slot = cache.field_observations && photon->receiver_object >= 0 &&
                          photon_field_slot(index, cache.iteration);
  if (!global_slot && !field_slot) {
    return;
  }
  const ccl_global KernelLight *light = &kernel_data_fetch(lights, photon->light);
  const bool local = field_slot && light->type == LIGHT_AREA && light->area.tan_half_spread > 0.0f;
  if (!photon_guide_light_supported(light) &&
      !(cache.area_guide_statistics && light->type == LIGHT_AREA))
  {
    return;
  }
  ccl_global float4 *hit = &cache.guide_hits[index];
  constexpr uint field_bit = 1u << 30;
  const uint claims = (global_slot ? 1u : 0u) | (local ? field_bit : 0u);
  if (!claims) {
    return;
  }
  const uint tag = atomic_fetch_and_or_uint32((ccl_global uint *)&hit->w, claims);
  const uint fresh = claims & ~tag;
  if (!fresh) {
    return;
  }
  const float3 p = make_float3(hit->x, hit->y, hit->z);
  const float weight = average(float3(photon->power)) * float(cache.capacity);
  if (!(weight > 0.0f) || !isfinite_safe(weight * (1.0f + len_squared(p)))) {
    return;
  }
  const uint source_cell = (tag >> 1) & 511u;
  if ((fresh & field_bit) && source_cell) {
    /* Spread the bounded bank across maps before taking first-writer slots.
     * Read only after camera work finishes. Failed texture retries never train it. */
    const uint slot = (hash_uint(index) % (PHOTON_FIELD_OBSERVATIONS / 16)) * 16 +
                      (cache.iteration & 15u);
    ccl_global KernelPhotonFieldObservation *o =
        &cache.field_observations[size_t(photon->light) * PHOTON_FIELD_OBSERVATIONS + slot];
    if (atomic_compare_and_swap_float(&o->occupied, 0.0f, 1.0f) == 0.0f) {
      const float4 prev = cache.field_vertices[index >> 4];
      const float4 target = kernel_data.integrator.photon_caustics_target;
      o->first = p;
      o->previous = make_float3(prev);
      o->receiver = (float3(photon->P) - make_float3(target)) / target.w;
      o->uv = make_float2(float((tag >> 10) & 1023u), float((tag >> 20) & 1023u)) / 1023.0f;
      o->object = photon->receiver_object;
      o->route = __float_as_uint(prev.w);
      o->band = (photon->pad & PHOTON_CAUSTICS_SPECTRAL) ? photon_field_band(cache, index) : ~0u;
      o->time_bin = min(uint(saturatef(photon_caustics_time(kg, cache.iteration)) * 4.0f), 3u);
      o->iteration = cache.iteration;
    }
  }
  if (!(fresh & 1u)) {
    return;
  }
  if (cache.area_guide_statistics && source_cell && source_cell <= PHOTON_AREA_GUIDE_CELLS) {
    ccl_global float4 *s =
        &cache.area_guide_statistics[photon->light * PHOTON_AREA_GUIDE_CELLS + source_cell - 1];
    atomic_add_and_fetch_float(&s->z, weight);
    atomic_add_and_fetch_float(&s->w, 1.0f);
  }
  if (!cache.guide || !cache.guide_statistics ||
      !photon_guide_light_supported(&kernel_data_fetch(lights, photon->light)))
  {
    return;
  }
  const int offset = photon->light * PHOTON_GUIDE_COMPONENTS;
  float score[PHOTON_GUIDE_COMPONENTS], largest = -FLT_MAX;
  for (int i = 0; i < PHOTON_GUIDE_COMPONENTS; ++i) {
    const ccl_global KernelPhotonGuide *g = &cache.guide[offset + i];
    score[i] = logf(g->weight.x) - 3.0f * logf(g->mean_sigma.w) -
               .5f * len_squared(p - make_float3(g->mean_sigma)) / sqr(g->mean_sigma.w);
    largest = max(largest, score[i]);
  }
  float sum = 0.0f;
  for (int i = 0; i < PHOTON_GUIDE_COMPONENTS; ++i) {
    score[i] = expf(score[i] - largest);
    sum += score[i];
  }
  for (int i = 0; i < PHOTON_GUIDE_COMPONENTS; ++i) {
    const float responsibility = score[i] / sum;
    const float w = weight * responsibility;
    ccl_global KernelPhotonGuideStatistics *s = &cache.guide_statistics[offset + i];
    atomic_add_and_fetch_float(&s->position_weight.x, w * p.x);
    atomic_add_and_fetch_float(&s->position_weight.y, w * p.y);
    atomic_add_and_fetch_float(&s->position_weight.z, w * p.z);
    atomic_add_and_fetch_float(&s->position_weight.w, w);
    atomic_add_and_fetch_float(&s->second_count.x, w * len_squared(p));
    atomic_add_and_fetch_float(&s->second_count.y, responsibility);
  }
}

CCL_NAMESPACE_END
