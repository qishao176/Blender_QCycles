/* SPDX-FileCopyrightText: 2026 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#pragma once

#include "kernel/integrator/state.h"
#include "kernel/integrator/state_util.h"
#include "kernel/integrator/volume_stack.h"

CCL_NAMESPACE_BEGIN

/* Explicit-priority media, following Schmidt and Budge (2002). Unlike automatic
 * nesting, this ordering is independent of ray direction and geometric volume.
 * Each closed material region of an object is a homogeneous dielectric, with
 * coefficients evaluated at entry. Joining glass and water must not merge their
 * memberships. No geometry is added to the BVH. */
struct DielectricMedium {
  int object, shader, count;
  float ior, inv_abbe;
  Spectrum transmission_tint;
};

ccl_device_inline int dielectric_priority(KernelGlobals kg, const int object, const int shader)
{
  return object == OBJECT_NONE || shader == SHADER_NONE ||
                 !kernel_data_fetch(objects, object).use_nested_dielectric ?
             0 : kernel_data_fetch(shaders, shader & SHADER_MASK).nested_dielectric_priority;
}

ccl_device_inline bool dielectric_overrides(KernelGlobals kg,
                                            const int a, const int shader_a,
                                            const int b, const int shader_b)
{
  const int pa = dielectric_priority(kg, a, shader_a), pb = dielectric_priority(kg, b, shader_b);
  return pa > pb || (pa == pb &&
                    (a > b || (a == b && (shader_a & SHADER_MASK) > (shader_b & SHADER_MASK))));
}

/* Identity, not equal priority or equal IOR, defines a shared medium. Keep all
 * object memberships so the final exit from an overlapping union is retained. */
ccl_device_inline bool dielectric_same_medium(const int a, const int b)
{
  return a != SHADER_NONE && b != SHADER_NONE && (a & SHADER_MASK) == (b & SHADER_MASK);
}

/* Membership identity is independent of the optional optical union policy. */
ccl_device_inline bool dielectric_merge_medium(KernelGlobals kg, const int a, const int b)
{
  return dielectric_same_medium(a, b) &&
         (kernel_data_fetch(shaders, a & SHADER_MASK).flags & SD_NESTED_DIELECTRIC_UNION);
}

/* A material identifies the medium; an object/material pair identifies one
 * membership of it. Distinct objects sharing a material must retain both exits. */
ccl_device_inline bool dielectric_same_membership(const int object_a,
                                                 const int shader_a,
                                                 const int object_b,
                                                 const int shader_b)
{
  return object_a == object_b && dielectric_same_medium(shader_a, shader_b);
}

/* Non-nested volume traversal retains its native object-only identity. */
ccl_device_inline VolumeStack dielectric_origin_key(KernelGlobals kg,
                                                   const ccl_private ShaderData *sd)
{
  return {sd->object,
          kernel_data_fetch(objects, sd->object).use_nested_dielectric ?
              (sd->shader & SHADER_MASK) : SHADER_NONE};
}

template<bool shadow, typename State>
ccl_device_inline DielectricMedium dielectric_read(const State state, const int i)
{
  if constexpr (shadow) {
    return {INTEGRATOR_STATE_ARRAY(state, shadow_medium_stack, i, object),
            INTEGRATOR_STATE_ARRAY(state, shadow_medium_stack, i, shader),
            INTEGRATOR_STATE_ARRAY(state, shadow_medium_stack, i, count),
            1.0f,
            0.0f,
            one_spectrum()};
  }
  else {
    return {INTEGRATOR_STATE_ARRAY(state, medium_stack, i, object),
            INTEGRATOR_STATE_ARRAY(state, medium_stack, i, shader),
            INTEGRATOR_STATE_ARRAY(state, medium_stack, i, count),
            INTEGRATOR_STATE_ARRAY(state, medium_stack, i, ior),
            INTEGRATOR_STATE_ARRAY(state, medium_stack, i, inv_abbe),
            INTEGRATOR_STATE_ARRAY(state, medium_stack, i, transmission_tint)};
  }
}

template<bool shadow, typename State>
ccl_device_inline void dielectric_write(State state, const int i, const DielectricMedium m)
{
  if constexpr (shadow) {
    INTEGRATOR_STATE_ARRAY_WRITE(state, shadow_medium_stack, i, object) = m.object;
    INTEGRATOR_STATE_ARRAY_WRITE(state, shadow_medium_stack, i, shader) = m.shader;
    INTEGRATOR_STATE_ARRAY_WRITE(state, shadow_medium_stack, i, count) = m.count;
  }
  else {
    INTEGRATOR_STATE_ARRAY_WRITE(state, medium_stack, i, object) = m.object;
    INTEGRATOR_STATE_ARRAY_WRITE(state, medium_stack, i, shader) = m.shader;
    INTEGRATOR_STATE_ARRAY_WRITE(state, medium_stack, i, count) = m.count;
    INTEGRATOR_STATE_ARRAY_WRITE(state, medium_stack, i, ior) = m.ior;
    INTEGRATOR_STATE_ARRAY_WRITE(state, medium_stack, i, inv_abbe) = m.inv_abbe;
    INTEGRATOR_STATE_ARRAY_WRITE(state, medium_stack, i, transmission_tint) = m.transmission_tint;
  }
}

template<bool shadow, typename State>
ccl_device_inline DielectricMedium dielectric_top(KernelGlobals kg,
                                                  const State state,
                                                  const int exclude_object = OBJECT_NONE,
                                                  const int exclude_shader = SHADER_NONE,
                                                  const bool exiting = true)
{
  DielectricMedium top = {OBJECT_NONE, SHADER_NONE, 0, 1.0f, 0.0f, one_spectrum()};
  for (int i = 0; i < kernel_data.volume_stack_size; ++i) {
    const DielectricMedium m = dielectric_read<shadow>(state, i);
    if (m.object == OBJECT_NONE)
      break;
    if (dielectric_same_membership(m.object, m.shader, exclude_object, exclude_shader) &&
        (!dielectric_merge_medium(kg, m.shader, exclude_shader) || (exiting && m.count == 1)))
    {
      continue;
    }
    if (dielectric_overrides(kg, m.object, m.shader, top.object, top.shader)) {
      top = m;
    }
  }
  return top;
}

template<bool shadow, typename State>
ccl_device_inline bool dielectric_surface_setup(KernelGlobals kg,
                                                const State state,
                                                ccl_private ShaderData *sd)
{
  if (!(kernel_data.kernel_features & KERNEL_FEATURE_NESTED_DIELECTRICS) ||
      !dielectric_priority(kg, sd->object, sd->shader))
    return false;
  const DielectricMedium outer = dielectric_top<shadow>(
      kg, state, sd->object, sd->shader, sd->runtime_flag & SR_BACKFACING);
  sd->medium_ior = outer.ior;
  sd->medium_inv_abbe = outer.inv_abbe;
  sd->medium_transmission_tint = outer.transmission_tint;
  return outer.object != OBJECT_NONE &&
         (dielectric_merge_medium(kg, outer.shader, sd->shader) ||
          dielectric_overrides(kg, outer.object, outer.shader, sd->object, sd->shader));
}

template<bool shadow, typename State>
ccl_device_inline void dielectric_sync_volume(KernelGlobals kg, State state)
{
#ifdef __VOLUME__
  int size = 0;
  for (int i = 0; i < kernel_data.volume_stack_size; ++i) {
    const VolumeStack entry = volume_stack_read<shadow>(state, i);
    if (entry.shader == SHADER_NONE)
      break;
    if (!dielectric_priority(kg, entry.object, entry.shader))
      volume_stack_write<shadow>(state, size++, entry);
  }
  const DielectricMedium top = dielectric_top<shadow>(kg, state);
  if (top.object != OBJECT_NONE && size < kernel_data.volume_stack_size - 1 &&
      (kernel_data_fetch(shaders, top.shader & SHADER_MASK).flags & SD_HAS_VOLUME))
  {
    volume_stack_write<shadow>(state, size++, {top.object, top.shader});
  }
  volume_stack_write<shadow>(state, size, {OBJECT_NONE, SHADER_NONE});
#endif
}

template<bool shadow, typename State>
ccl_device_inline void dielectric_enter_exit(KernelGlobals kg,
                                             State state,
                                             const ccl_private ShaderData *sd,
                                             const int entry_count = 1)
{
  int size = 0, found = -1;
  for (; size < kernel_data.volume_stack_size; ++size) {
    const DielectricMedium m = dielectric_read<shadow>(state, size);
    if (m.object == OBJECT_NONE)
      break;
    if (dielectric_same_membership(m.object, m.shader, sd->object, sd->shader))
      found = size;
  }
  if (found >= 0 && dielectric_merge_medium(kg, sd->shader, sd->shader)) {
    DielectricMedium m = dielectric_read<shadow>(state, found);
    m.count += (sd->runtime_flag & SR_BACKFACING) ? -1 : entry_count;
    if (m.count > 0) {
      dielectric_write<shadow>(state, found, m);
      return;
    }
  }
  if (sd->runtime_flag & SR_BACKFACING) {
    if (found >= 0) {
      for (int i = found; i < size - 1; ++i)
        dielectric_write<shadow>(state, i, dielectric_read<shadow>(state, i + 1));
      dielectric_write<shadow>(state, size - 1, {OBJECT_NONE, SHADER_NONE, 0, 1.0f, 0.0f, one_spectrum()});
    }
  }
  else if (found < 0 && size < kernel_data.volume_stack_size - 1) {
    const Spectrum tint = sd->interior_transmission_weight > 0.0f ?
                              sd->interior_transmission_tint / sd->interior_transmission_weight :
                              one_spectrum();
    dielectric_write<shadow>(state,
                              size,
                              {sd->object, sd->shader, entry_count, sd->interior_ior, sd->interior_inv_abbe, tint});
    dielectric_write<shadow>(state, size + 1, {OBJECT_NONE, SHADER_NONE, 0, 1.0f, 0.0f, one_spectrum()});
  }
}

template<bool shadow, typename State>
ccl_device_inline void path_media_enter_exit(KernelGlobals kg,
                                             State state,
                                             const ccl_private ShaderData *sd)
{
  if ((kernel_data.kernel_features & KERNEL_FEATURE_NESTED_DIELECTRICS) &&
      dielectric_priority(kg, sd->object, sd->shader))
  {
    dielectric_enter_exit<shadow>(kg, state, sd);
    dielectric_sync_volume<shadow>(kg, state);
    return;
  }
#ifdef __VOLUME__
  volume_stack_enter_exit<shadow>(kg, state, sd);
#endif
}

CCL_NAMESPACE_END
