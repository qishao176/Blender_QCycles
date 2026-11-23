/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: Apache-2.0 */
#pragma once

#ifndef __KERNEL_GPU__
#  include "util/lpe_query.h"
#endif

CCL_NAMESPACE_BEGIN

ccl_device_inline bool lpe_query_enabled(KernelGlobals kg)
{
#if defined(__KERNEL_CUDA__)
  return kernel_integrator_state.query_records != nullptr;
#elif defined(__KERNEL_GPU__)
  (void)kg;
  return false;
#else
  return kg->query != nullptr;
#endif
}

ccl_device_inline bool query_total_light(KernelGlobals kg)
{
#if defined(__KERNEL_CUDA__)
  return kernel_integrator_state.query_total_light;
#elif defined(__KERNEL_GPU__)
  return false;
#else
  return kg->query->total_light;
#endif
}

#ifdef __KERNEL_CUDA__
ccl_device_inline uint query_gpu_append(const uint parent, int object, int shader, int event,
                                        int pixel, const Spectrum value)
{
  if (pixel >= 0 && is_zero(value)) {
    return 0;
  }
  const uint i = atomic_fetch_and_add_uint32(kernel_integrator_state.query_count, 1);
  if (i >= kernel_integrator_state.query_capacity) {
    return 0;
  }
  const float3 rgb = spectrum_to_rgb(value);
  KernelQueryRecord &entry = kernel_integrator_state.query_records[i];
  entry = {parent, object, shader == SHADER_NONE ? -1 : shader & SHADER_MASK,
           pixel < 0 && kernel_integrator_state.query_total_light ? 0 : event, rgb.x, rgb.y, rgb.z, pixel};
  uint *mask = (uint *)(kernel_integrator_state.query_pages + (i >> 7));
  atomic_fetch_and_or_uint32(mask + ((i >> 5) & 3), uint(1) << (i & 31));
  return i + 1;
}
ccl_device_inline void query_emission(KernelGlobals kg, ConstIntegratorState state,
                                      int object, int shader, const Spectrum value)
{
  if (lpe_query_enabled(kg) && !(INTEGRATOR_STATE(state, path, flag) & PATH_RAY_SHADOW_CATCHER_HIT)) {
    query_gpu_append(uint(INTEGRATOR_STATE(state, path, lpe_states)), object, shader, 0,
                     INTEGRATOR_STATE(state, path, render_pixel_index),
                     value * INTEGRATOR_STATE(state, path, lpe_weight));
  }
}
ccl_device_inline void query_direct(KernelGlobals kg, ConstIntegratorShadowState state,
                                    const Spectrum value)
{
  if (!lpe_query_enabled(kg) || (INTEGRATOR_STATE(state, shadow_path, flag) & PATH_RAY_SHADOW_CATCHER_HIT)) {
    return;
  }
  const Spectrum wd = INTEGRATOR_STATE(state, shadow_path, lpe_diffuse_weight);
  const Spectrum wg = INTEGRATOR_STATE(state, shadow_path, lpe_glossy_weight);
  const Spectrum weights[] = {wd, wg, one_spectrum() - wd - wg};
  const int events[] = {LPE_DIFFUSE, LPE_GLOSSY, LPE_TRANSMISSION};
  const int event = int(INTEGRATOR_STATE(state, shadow_path, lpe_glossy));
  const bool total = query_total_light(kg);
  for (int i = 0; i < (total ? 1 : 3); ++i) {
    if (total || !is_zero(weights[i])) {
      uint parent = uint(INTEGRATOR_STATE(state, shadow_path, lpe_diffuse));
      if (event >= 0) {
        parent = query_gpu_append(parent, INTEGRATOR_STATE(state, shadow_path, lpe_receiver_object),
                                  INTEGRATOR_STATE(state, shadow_path, lpe_receiver_shader),
                                  event ? event : events[i], -1, zero_spectrum());
      }
      query_gpu_append(parent, INTEGRATOR_STATE(state, shadow_path, lpe_light_object),
                       INTEGRATOR_STATE(state, shadow_path, lpe_light_shader), 0,
                       INTEGRATOR_STATE(state, shadow_path, render_pixel_index),
                       value * (total ? one_spectrum() : weights[i]) * INTEGRATOR_STATE(state, shadow_path, lpe_weight));
    }
  }
}
#endif

#ifndef __KERNEL_GPU__
ccl_device_inline QueryVertex query_vertex(int object, int shader, int event)
{
  return {object, shader == SHADER_NONE ? -1 : shader & SHADER_MASK, event};
}

ccl_device_inline void query_record(KernelGlobals kg, const QueryPath &path,
                                    bool photon, const Spectrum value)
{
  const float3 rgb = spectrum_to_rgb(value);
  kg->query->record(path, photon, rgb.x, rgb.y, rgb.z);
}

ccl_device_inline void query_emission(KernelGlobals kg, ConstIntegratorState state,
                                      int object, int shader, const Spectrum value)
{
  if (kg->query && !(INTEGRATOR_STATE(state, path, flag) & PATH_RAY_SHADOW_CATCHER_HIT)) {
    QueryPath path = kg->query->camera;
    path.push_back(query_vertex(object, shader, 0));
    query_record(kg, path, false, value * INTEGRATOR_STATE(state, path, lpe_weight));
  }
}

ccl_device_inline void query_direct(KernelGlobals kg, ConstIntegratorShadowState state,
                                    const Spectrum value)
{
  QueryPixel *query = kg->query;
  if (!query || query->shadow_state != state ||
      (INTEGRATOR_STATE(state, shadow_path, flag) & PATH_RAY_SHADOW_CATCHER_HIT)) {
    return;
  }
  query->shadow_state = nullptr;
  const Spectrum wd = INTEGRATOR_STATE(state, shadow_path, lpe_diffuse_weight);
  const Spectrum wg = INTEGRATOR_STATE(state, shadow_path, lpe_glossy_weight);
  const Spectrum weights[] = {wd, wg, one_spectrum() - wd - wg};
  const int events[] = {LPE_DIFFUSE, LPE_GLOSSY, LPE_TRANSMISSION};
  const bool total = query->total_light;
  for (int i = 0; i < (total ? 1 : 3); ++i) {
    if (!total && is_zero(weights[i])) {
      continue;
    }
    QueryPath path = query->shadow;
    if (!query->skip) {
      QueryVertex receiver = query->receiver;
      receiver[2] = query->receiver_event ? query->receiver_event : events[i];
      path.push_back(receiver);
    }
    path.insert(path.end(), query->manifold.begin(), query->manifold.end());
    path.push_back(query->source);
    query_record(kg, path, false, value * (total ? one_spectrum() : weights[i]) * INTEGRATOR_STATE(state, shadow_path, lpe_weight));
  }
}
#endif
CCL_NAMESPACE_END
