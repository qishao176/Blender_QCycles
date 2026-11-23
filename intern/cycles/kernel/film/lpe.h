/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: Apache-2.0 */

#pragma once

#include "kernel/film/deep_physical.h"

#include "kernel/film/write.h"
#include "kernel/film/lpe_query.h"

CCL_NAMESPACE_BEGIN

ccl_device_inline bool lpe_selection_contains(KernelGlobals kg,
                                              const int offset,
                                              const int count,
                                              const int value)
{
  if (count < 0) {
    return true;
  }
  for (int i = 0; i < count; i++) {
    if (kernel_data_fetch(lpe_selections, offset + i) == value) {
      return true;
    }
  }
  return false;
}

ccl_device uint64_t lpe_transition(KernelGlobals kg,
                                   const uint64_t active,
                                   const int object,
                                   const int shader,
                                   const int event,
                                   const bool mnee_receiver = false,
                                   const bool caustic_caster = false)
{
  uint64_t result = 0;
  for (int i = 0; i < kernel_data.film.lpe_state_count; i++) {
    const uint64_t bit = uint64_t(1) << i;
    if (!(active & bit)) {
      continue;
    }
    const ccl_global KernelLPEState *s = &kernel_data_fetch(lpe_states,
                                                            kernel_data.film.lpe_state_offset + i);
    if (s->flags & LPE_MASK_TARGET) {
      const int value = s->selector == 2 ? (shader & SHADER_MASK) : object;
      const bool selected = lpe_selection_contains(
          kg, s->selection_offset, s->selection_count, value);
      /* Once the target scatters, this camera sample's mask is finished. A
       * transparent continuation may still reveal coverage behind this hit. */
      if (event == LPE_TRANSPARENT || (!selected && (s->flags & LPE_LOOP))) {
        result |= bit;
      }
      continue;
    }
    if ((s->flags & LPE_LOOP) || (caustic_caster && (s->flags & LPE_CAUSTIC_SOURCE)) ||
        (event == LPE_TRANSPARENT && s->events != LPE_TRANSPARENT))
    {
      result |= bit;
    }
    if ((s->flags & LPE_MNEE_RECEIVER) && !mnee_receiver) {
      continue;
    }
    if (!(s->events & event)) {
      continue;
    }
    const int value = s->selector == 2 ? (shader & SHADER_MASK) : object;
    if (s->selector == 0 ||
        lpe_selection_contains(kg, s->selection_offset, s->selection_count, value))
    {
      result |= s->next;
      if (s->flags & LPE_REPEAT_MATCH)
        result |= bit;
    }
  }
  return result;
}

ccl_device_inline void lpe_path_event(KernelGlobals kg,
                                      IntegratorState state,
                                      ccl_private const ShaderData *sd,
                                      const int event)
{
  if (kernel_data.film.lpe_state_count == 0 && !lpe_query_enabled(kg)) {
    return;
  }
#ifndef __KERNEL_GPU__
  if (kg->query) {
    kg->query->camera.push_back(query_vertex(sd->object, sd->shader, event));
  }
#endif
#ifdef __KERNEL_CUDA__
  if (lpe_query_enabled(kg)) {
    INTEGRATOR_STATE_WRITE(state, path, lpe_states) = query_gpu_append(
        uint(INTEGRATOR_STATE(state, path, lpe_states)), sd->object, sd->shader, event, -1, zero_spectrum());
    return;
  }
#endif
  INTEGRATOR_STATE_WRITE(state, path, lpe_states) = lpe_transition(
      kg, INTEGRATOR_STATE(state, path, lpe_states), sd->object, sd->shader, event);
}

/* Select a scattering component without changing Cycles' transport estimator.
 * A spectral correction keeps classified LPE contributions unbiased for mixed BSDFs. */
ccl_device_inline void lpe_path_bsdf(KernelGlobals kg,
                                     IntegratorState state,
                                     ccl_private const ShaderData *sd,
                                     const int label,
                                     const Spectrum diffuse,
                                     const Spectrum glossy,
                                     const float random)
{
  if (kernel_data.film.lpe_state_count == 0 && !lpe_query_enabled(kg)) {
    return;
  }
  if (INTEGRATOR_STATE(state, path, lpe_skip_surface)) {
    INTEGRATOR_STATE_WRITE(state, path, lpe_skip_surface) = false;
    return;
  }
  if (label & LABEL_TRANSPARENT) {
    lpe_path_event(kg, state, sd, LPE_TRANSPARENT);
    return;
  }
  const Spectrum transmission = max(one_spectrum() - diffuse - glossy, zero_spectrum());
  const float pd = max(average(diffuse), 0.0f);
  const float pg = max(average(glossy), 0.0f);
  const float pt = max(average(transmission), 0.0f);
  const float sum = pd + pg + pt;
  const float r = random * sum;
  Spectrum weight;
  int event;
  if (r < pd && pd > 0.0f) {
    weight = diffuse * (sum / pd);
    event = LPE_DIFFUSE;
  }
  else if (r < pd + pg && pg > 0.0f) {
    weight = glossy * (sum / pg);
    event = LPE_GLOSSY;
  }
  else {
    weight = pt > 0.0f ? transmission * (sum / pt) : zero_spectrum();
    event = LPE_TRANSMISSION;
  }
  INTEGRATOR_STATE_WRITE(state, path, lpe_weight) *= weight;
  lpe_path_event(kg, state, sd, event);
}

ccl_device_inline void lpe_shadow_init(KernelGlobals kg,
                                       ConstIntegratorState state,
                                       IntegratorShadowState shadow,
                                       ccl_private const ShaderData *sd,
                                       const int light_object,
                                       const Spectrum diffuse,
                                       const Spectrum glossy,
                                       const int event = 0,
                                       const bool preserve_mnee = false,
                                       const int light_shader = SHADER_NONE)
{
  if (kernel_data.film.lpe_state_count == 0 && !lpe_query_enabled(kg)) {
    return;
  }
  const uint64_t active = INTEGRATOR_STATE(state, path, lpe_states);
  const bool skip = INTEGRATOR_STATE(state, path, lpe_skip_surface);
#ifndef __KERNEL_GPU__
  if (kg->query) {
    QueryPixel &query = *kg->query;
    query.shadow = query.camera;
    query.shadow_state = shadow;
    query.receiver = query_vertex(sd->object, sd->shader, 0);
    query.source = query_vertex(light_object, light_shader, 0);
    query.receiver_event = event;
    query.skip = skip;
    if (!preserve_mnee) {
      query.manifold.clear();
    }
  }
#endif
#ifdef __KERNEL_CUDA__
  if (lpe_query_enabled(kg)) {
    INTEGRATOR_STATE_WRITE(shadow, shadow_path, lpe_diffuse) = active;
    INTEGRATOR_STATE_WRITE(shadow, shadow_path, lpe_glossy) = skip ? uint64_t(-1) : uint64_t(event);
  }
  else
#endif
  if (!preserve_mnee) {
    INTEGRATOR_STATE_WRITE(shadow, shadow_path, lpe_diffuse) =
        skip ? active :
               lpe_transition(kg, active, sd->object, sd->shader, event ? event : LPE_DIFFUSE);
    INTEGRATOR_STATE_WRITE(shadow, shadow_path, lpe_glossy) =
        skip || event ? 0 : lpe_transition(kg, active, sd->object, sd->shader, LPE_GLOSSY);
    INTEGRATOR_STATE_WRITE(shadow, shadow_path, lpe_transmission) =
        skip || event ? 0 : lpe_transition(kg, active, sd->object, sd->shader, LPE_TRANSMISSION);
  }
  INTEGRATOR_STATE_WRITE(shadow, shadow_path, lpe_diffuse_weight) = skip || event ?
                                                                        one_spectrum() :
                                                                        diffuse;
  INTEGRATOR_STATE_WRITE(shadow, shadow_path, lpe_glossy_weight) = skip || event ?
                                                                       zero_spectrum() :
                                                                       glossy;
  INTEGRATOR_STATE_WRITE(shadow, shadow_path, lpe_weight) = INTEGRATOR_STATE(
      state, path, lpe_weight);
  INTEGRATOR_STATE_WRITE(shadow, shadow_path, lpe_light_object) = light_object;
  INTEGRATOR_STATE_WRITE(shadow, shadow_path, lpe_light_shader) = light_shader;
  INTEGRATOR_STATE_WRITE(shadow, shadow_path, lpe_receiver_shader) = sd->shader;
  INTEGRATOR_STATE_WRITE(shadow, shadow_path, lpe_receiver_object) = preserve_mnee ? OBJECT_NONE : sd->object;
}

ccl_device_inline bool lpe_mask_pending(KernelGlobals kg,
                                        ConstIntegratorState state,
                                        const int object,
                                        const int shader)
{
  if (!kernel_data.film.lpe_state_count)
    return false;
  const uint64_t active = INTEGRATOR_STATE(state, path, lpe_states);
  for (int i = 0; i < kernel_data.film.lpe_state_count; i++) {
    const ccl_global KernelLPEState *s = &kernel_data_fetch(lpe_states,
                                                            kernel_data.film.lpe_state_offset + i);
    if ((active & (uint64_t(1) << i)) && (s->flags & LPE_MASK_TARGET) &&
        lpe_selection_contains(kg,
                               s->selection_offset,
                               s->selection_count,
                               s->selector == 2 ? (shader & SHADER_MASK) : object))
    {
      return true;
    }
  }
  return false;
}

ccl_device void film_write_lpe_masks(KernelGlobals kg,
                                     ConstIntegratorState state,
                                     ccl_private const ShaderData *sd,
                                     const Spectrum opacity,
                                     ccl_global float *ccl_restrict render_buffer)
{
  if (!kernel_data.film.lpe_state_count)
    return;
  if (INTEGRATOR_STATE(state, path, flag) & PATH_RAY_SHADOW_CATCHER_HIT)
    return;
  const uint64_t active = INTEGRATOR_STATE(state, path, lpe_states);
  ccl_global float *buffer = film_pass_pixel_render_buffer(kg, state, render_buffer);
  for (int i = 0; i < kernel_data.film.lpe_state_count; i++) {
    const ccl_global KernelLPEState *s = &kernel_data_fetch(lpe_states,
                                                            kernel_data.film.lpe_state_offset + i);
    if (!(active & (uint64_t(1) << i)) || !(s->flags & LPE_MASK_TARGET) ||
        s->pass_offset == PASS_UNUSED ||
        !lpe_selection_contains(kg,
                                s->selection_offset,
                                s->selection_count,
                                s->selector == 2 ? (sd->shader & SHADER_MASK) : sd->object))
    {
      continue;
    }
    Spectrum coverage = INTEGRATOR_STATE(state, path, throughput) * opacity;
    if (s->flags & LPE_WEIGHTED)
      coverage *= INTEGRATOR_STATE(state, path, lpe_weight);
    film_write_pass_float(buffer + s->pass_offset, average(coverage));
  }
}

ccl_device_inline bool lpe_emitter_matches(KernelGlobals kg,
                                          const ccl_global KernelLPEState *state,
                                          const int object,
                                          const int shader)
{
  const int value = (state->flags & LPE_SOURCE_MATERIAL) ? (shader & SHADER_MASK) : object;
  return lpe_selection_contains(kg, state->light_offset, state->light_count, value);
}

/* Photon density estimates can contain thousands of small contributions per sample.
 * Accumulate those locally before touching the progressive float film, while sharing
 * exactly the same terminal, source, Union and spectral-weight semantics. */
template<bool accumulate = false, bool photon = false>
ccl_device void film_write_lpe(KernelGlobals kg,
                               const uint64_t active,
                               const Spectrum weight,
                               const int light_object,
                               const Spectrum contribution,
                               ccl_global float *ccl_restrict buffer,
                               const int light_shader = SHADER_NONE,
                               ccl_private Spectrum *state_contributions = nullptr,
                               const uint deep_pixel = 0,
                               const float deep_anchor = -1.0f,
                               const uint deep_kind = 0)
{
  for (int i = 0; i < kernel_data.film.lpe_state_count; i++) {
    if (!(active & (uint64_t(1) << i))) {
      continue;
    }
    const ccl_global KernelLPEState *s = &kernel_data_fetch(lpe_states,
                                                            kernel_data.film.lpe_state_offset + i);
    if ((s->flags & LPE_MASK_TARGET) || (!photon && (s->flags & LPE_PHOTON_SOURCE)) ||
        s->pass_offset == PASS_UNUSED ||
        !lpe_emitter_matches(kg, s, light_object, light_shader))
    {
      continue;
    }
    bool weighted = (s->flags & LPE_WEIGHTED) != 0;
    if (s->flags & LPE_UNION) {
      bool already_written = false;
      /* Different source predicates may overlap. A union writes each physical
       * contribution once, while ordinary Add still uses independent passes. */
      for (int j = 0; j < kernel_data.film.lpe_state_count; j++) {
        if (j == i || !(active & (uint64_t(1) << j)))
          continue;
        const ccl_global KernelLPEState *other = &kernel_data_fetch(
            lpe_states, kernel_data.film.lpe_state_offset + j);
        if (other->pass_offset != s->pass_offset || !(other->flags & LPE_UNION) ||
            (!photon && (other->flags & LPE_PHOTON_SOURCE)) ||
            !lpe_emitter_matches(kg, other, light_object, light_shader))
          continue;
        if (j < i) {
          already_written = true;
          break;
        }
        weighted &= (other->flags & LPE_WEIGHTED) != 0;
      }
      if (already_written)
        continue;
    }
    const Spectrum value = weighted ? contribution * weight : contribution;
    if constexpr (accumulate) {
      state_contributions[i] += value;
    }
    else {
      film_write_pass_spectrum(buffer + s->pass_offset, value);
      if (kernel_data.film.deep_physical_ptr && s->deep_channel >= 0)
        deep_physical_write(kernel_data.film.deep_physical_ptr, deep_pixel,
                            kernel_data.film.deep_lpe_channels, deep_anchor, deep_kind,
                            spectrum_to_rgb(value), s->deep_channel);
    }
  }
}

CCL_NAMESPACE_END
