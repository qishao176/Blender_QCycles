/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#pragma once

#include "kernel/integrator/nested_dielectrics.h"
#include "kernel/integrator/surface_shader.h"

#include "kernel/bvh/bvh.h"

#include "kernel/geom/shader_data.h"

#include "kernel/integrator/intersect_closest.h"
#include "kernel/integrator/volume_stack.h"

CCL_NAMESPACE_BEGIN

ccl_device void integrator_volume_stack_update_for_subsurface(KernelGlobals kg,
                                                              IntegratorState state,
                                                              const float3 from_P,
                                                              const float3 to_P)
{
#ifdef __VOLUME__
  PROFILING_INIT(kg, PROFILING_INTERSECT_VOLUME_STACK);

  ShaderDataTinyStorage stack_sd_storage;
  ccl_private ShaderData *stack_sd = AS_SHADER_DATA(&stack_sd_storage);

  kernel_assert(kernel_data.integrator.use_volumes);

  Ray volume_ray ccl_optional_struct_init;
  volume_ray.P = from_P;
  volume_ray.D = safe_normalize_len(to_P - from_P, &volume_ray.tmax);
  volume_ray.tmin = 0.0f;
  volume_ray.self.object = INTEGRATOR_STATE(state, isect, object);
  volume_ray.self.prim = INTEGRATOR_STATE(state, isect, prim);
  volume_ray.self.light_object = OBJECT_NONE;
  volume_ray.self.light_prim = PRIM_NONE;
  /* Store to avoid global fetches on every intersection step. */
  const uint volume_stack_size = kernel_data.volume_stack_size;

  const uint32_t path_flag = INTEGRATOR_STATE(state, path, flag);
  const PathRayVisibility visibility = SHADOW_CATCHER_PATH_VISIBILITY(path_flag,
                                                                      PATH_RAY_VISIBILITY_ALL);

#  ifdef __VOLUME_RECORD_ALL__
  Intersection hits[2 * MAX_VOLUME_STACK_SIZE + 1];
  const uint num_hits = scene_intersect_volume(
      kg, &volume_ray, hits, 2 * volume_stack_size, visibility);
  if (num_hits > 0) {
    Intersection *isect = hits;

    qsort(hits, num_hits, sizeof(Intersection), intersections_compare);

    for (uint hit = 0; hit < num_hits; ++hit, ++isect) {
      /* Ignore self, SSS itself already enters and exits the object. */
      if (isect->object == volume_ray.self.object) {
        continue;
      }
      shader_setup_from_ray(kg, stack_sd, &volume_ray, isect);
      path_media_enter_exit<false>(kg, state, stack_sd);
    }
  }
#  else
  Intersection isect;
  int step = 0;
  while (step < 2 * volume_stack_size &&
         scene_intersect_volume(kg, &volume_ray, &isect, visibility))
  {
    /* Ignore self, SSS itself already enters and exits the object. */
    if (isect.object != volume_ray.self.object) {
      shader_setup_from_ray(kg, stack_sd, &volume_ray, &isect);
      path_media_enter_exit<false>(kg, state, stack_sd);
    }
    /* Move ray forward. */
    volume_ray.tmin = intersection_t_offset(isect.t);
    volume_ray.self.object = isect.object;
    volume_ray.self.prim = isect.prim;
    ++step;
  }
#  endif
#endif
}

/* Evaluate actual material coefficients when the ray starts inside a medium.
 * A cached texture miss restarts initialization, before any film contribution. */
ccl_device bool dielectric_probe_origin(KernelGlobals kg, IntegratorState state,
                                        const ccl_private Ray *ray,
                                        const ccl_private Intersection *isect,
                                        const int entry_count = 1)
{
  ShaderData sd;
  shader_setup_from_ray(kg, &sd, ray, isect);
  sd.medium_ior = 1.0f;
#ifdef __SPECTRAL__
  shader_setup_wavelength(kg, &sd, state);
#endif
  surface_shader_eval<KERNEL_FEATURE_NODE_MASK_SURFACE &
                      ~(KERNEL_FEATURE_NODE_AOV | KERNEL_FEATURE_NODE_RAYTRACE)>(
      kg, state, &sd, nullptr, PATH_RAY_VISIBILITY_NONE, 0);
  if (sd.runtime_flag & SR_CACHE_MISS) return false;
  sd.runtime_flag &= ~SR_BACKFACING;
  dielectric_enter_exit<false>(kg, state, &sd, entry_count);
  return true;
}

#ifdef __VOLUME__
/* For a closed oriented shell, exits minus entries along the probe ray is its
 * winding number at the origin. Joined shells can have coincident exits. Read
 * each intersected triangle once, retaining only one material evaluation per
 * active membership. Non-union media use their nearest boundary as before. */
ccl_device bool dielectric_probe_union_origin(KernelGlobals kg,
                                             IntegratorState state,
                                             const ccl_private Ray *ray,
                                             const PathRayVisibility visibility,
                                             ccl_private int *volume_size)
{
  VolumeProbePage page;
  page.after = make_int3(-1, -1, -1);
  VolumeStack key = {OBJECT_NONE, SHADER_NONE};
  Intersection nearest_exit;
  float nearest_t = FLT_MAX;
  int depth = 0;
  bool nested = false;
  do {
    page.size = 0;
    page.more = false;
    scene_intersect_volume_probe(kg, ray, visibility, &page);
    /* Including the final sentinel flushes the last material across page boundaries. */
    for (int i = 0; i < page.size + !page.more; ++i) {
      const bool done = i == page.size;
      const VolumeProbeHit hit = done ? VolumeProbeHit{} : page.hits[i];
      if (done || hit.isect.object != key.object || hit.shader != key.shader) {
        if (depth > 0) {
          if (nested) {
            if (!dielectric_probe_origin(kg, state, ray, &nearest_exit, depth)) {
              return false;
            }
          }
          else if (*volume_size < kernel_data.volume_stack_size - 1) {
            const int shader = kernel_data_fetch(tri_shader, nearest_exit.prim);
            integrator_state_write_volume_stack(state, (*volume_size)++, {key.object, shader});
          }
        }
        if (done) {
          break;
        }
        key = {hit.isect.object, hit.shader};
        nested = dielectric_priority(kg, key.object, key.shader) > 0;
        depth = 0;
        nearest_exit.t = nearest_t = FLT_MAX;
      }

      ShaderDataTinyStorage storage;
      ccl_private ShaderData *sd = AS_SHADER_DATA(&storage);
      shader_setup_from_ray(kg, sd, ray, &hit.isect);
      const bool backfacing = sd->runtime_flag & SR_BACKFACING;
      if (nested && dielectric_merge_medium(kg, key.shader, key.shader)) {
        depth += backfacing ? 1 : -1;
      }
      else if (hit.isect.t < nearest_t) {
        depth = backfacing;
        nearest_t = hit.isect.t;
      }
      if (backfacing && hit.isect.t < nearest_exit.t) {
        nearest_exit = hit.isect;
      }
    }
    if (page.more) {
      const VolumeProbeHit last = page.hits[page.size - 1];
      page.after = make_int3(last.isect.object, last.shader, last.isect.prim);
    }
  } while (page.more);
  return true;
}
#endif

ccl_device bool integrator_volume_stack_init(
    KernelGlobals kg, IntegratorState state,
    const PathRayVisibility ray_visibility = PATH_RAY_VISIBILITY_CAMERA)
{
#ifdef __VOLUME__
  if (kernel_data.kernel_features & KERNEL_FEATURE_NESTED_DIELECTRICS) {
    dielectric_write<false>(state, 0, {OBJECT_NONE, SHADER_NONE, 0, 1.0f, 0.0f, one_spectrum()});
  }
  PROFILING_INIT(kg, PROFILING_INTERSECT_VOLUME_STACK);

  ShaderDataTinyStorage stack_sd_storage;
  ccl_private ShaderData *stack_sd = AS_SHADER_DATA(&stack_sd_storage);

  Ray volume_ray ccl_optional_struct_init;
  integrator_state_read_ray(state, &volume_ray);

  /* Trace ray in random direction. Any direction works, Z up is a guess to get the
   * fewest hits. */
  volume_ray.D = make_float3(0.0f, 0.0f, 1.0f);
  volume_ray.tmin = 0.0f;
  volume_ray.tmax = FLT_MAX;
  volume_ray.self.object = OBJECT_NONE;
  volume_ray.self.prim = PRIM_NONE;
  volume_ray.self.light_object = OBJECT_NONE;
  volume_ray.self.light_prim = PRIM_NONE;

  int stack_index = 0;
  int enclosed_index = 0;

  const uint32_t path_flag = INTEGRATOR_STATE(state, path, flag);
  const PathRayVisibility visibility = SHADOW_CATCHER_PATH_VISIBILITY(path_flag,
                                                                      ray_visibility);

  /* Initialize volume stack with background volume For shadow catcher the
   * background volume is always assumed to be CG. */
  if (kernel_data.background.volume_shader != SHADER_NONE) {
    if (!(path_flag & PATH_RAY_SHADOW_CATCHER_PASS)) {
      INTEGRATOR_STATE_ARRAY_WRITE(
          state, volume_stack, stack_index, object) = kernel_data.background.object_index;
      INTEGRATOR_STATE_ARRAY_WRITE(
          state, volume_stack, stack_index, shader) = kernel_data.background.volume_shader;
      stack_index++;
    }
  }

  /* Store to avoid global fetches on every intersection step. */
  const uint volume_stack_size = kernel_data.volume_stack_size;

  if ((kernel_data.kernel_features & KERNEL_FEATURE_NESTED_DIELECTRICS) &&
      kernel_data.use_dielectric_union)
  {
    if (!dielectric_probe_union_origin(kg, state, &volume_ray, visibility, &stack_index)) {
      return false;
    }
    integrator_state_write_volume_stack(state, stack_index, {OBJECT_NONE, SHADER_NONE});
    dielectric_sync_volume<false>(kg, state);
    return true;
  }

#  ifdef __VOLUME_RECORD_ALL__
  Intersection hits[2 * MAX_VOLUME_STACK_SIZE + 1];
  const uint num_hits = scene_intersect_volume(
      kg, &volume_ray, hits, 2 * volume_stack_size, visibility);
  if (num_hits > 0) {
    VolumeStack enclosed_volumes[2 * MAX_VOLUME_STACK_SIZE];
    Intersection *isect = hits;

    qsort(hits, num_hits, sizeof(Intersection), intersections_compare);

    for (uint hit = 0; hit < num_hits; ++hit, ++isect) {
      shader_setup_from_ray(kg, stack_sd, &volume_ray, isect);
      const VolumeStack key = dielectric_origin_key(kg, stack_sd);
      if (stack_sd->runtime_flag & SR_BACKFACING) {
        bool need_add = true;
        for (int i = 0; i < enclosed_index && need_add; ++i) {
          /* If ray exited the volume and never entered to that volume
           * it means that camera is inside such a volume.
           */
          if (enclosed_volumes[i].object == key.object &&
              enclosed_volumes[i].shader == key.shader) {
            need_add = false;
          }
        }
        for (int i = 0; i < stack_index && need_add; ++i) {
          /* Don't add intersections twice. */
          const VolumeStack entry = integrator_state_read_volume_stack(state, i);
          if (entry.object == key.object &&
              (key.shader == SHADER_NONE || (entry.shader & SHADER_MASK) == key.shader)) {
            need_add = false;
            break;
          }
        }
        if (need_add && stack_index < volume_stack_size - 1) {
          if (dielectric_priority(kg, stack_sd->object, stack_sd->shader)) {
            if (!dielectric_probe_origin(kg, state, &volume_ray, isect)) return false;
          }
          else if (stack_sd->shader_flag & SD_HAS_VOLUME) {
            const VolumeStack new_entry = {stack_sd->object, stack_sd->shader};
            integrator_state_write_volume_stack(state, stack_index++, new_entry);
          }
        }
      }
      else {
        /* If ray from camera enters the volume, this volume shouldn't
         * be added to the stack on exit.
         */
        enclosed_volumes[enclosed_index++] = key;
      }
    }
  }
#  else
  /* A closest-hit march skips coincident boundaries when tmin advances past
   * their shared distance. Classify the first hit of each object/material pair
   * independently, without advancing the probe. Excluding the entire object
   * would hide other closed materials after joining glass and water meshes. */
#    if defined(__KERNEL_CUDA__) || defined(__KERNEL_OPTIX__)
  if (kernel_data.kernel_features & KERNEL_FEATURE_NESTED_DIELECTRICS) {
    VolumeStack visited[2 * MAX_VOLUME_STACK_SIZE];
    int count = 0;
    while (count < 2 * volume_stack_size && stack_index < volume_stack_size - 1) {
      Intersection isect;
      if (!scene_intersect_volume(kg, &volume_ray, &isect, visibility, visited, count)) {
        break;
      }
      shader_setup_from_ray(kg, stack_sd, &volume_ray, &isect);
      visited[count++] = dielectric_origin_key(kg, stack_sd);
      if (stack_sd->runtime_flag & SR_BACKFACING) {
        if (dielectric_priority(kg, stack_sd->object, stack_sd->shader)) {
          if (!dielectric_probe_origin(kg, state, &volume_ray, &isect)) {
            return false;
          }
        }
        else if (stack_sd->shader_flag & SD_HAS_VOLUME) {
          const VolumeStack entry = {stack_sd->object, stack_sd->shader};
          integrator_state_write_volume_stack(state, stack_index++, entry);
        }
      }
    }
  }
  else
#    endif
  {
  /* CUDA does not support definition of a variable size arrays, so use the maximum possible. */
  VolumeStack enclosed_volumes[MAX_VOLUME_STACK_SIZE];
  int step = 0;

  while (stack_index < volume_stack_size - 1 && enclosed_index < MAX_VOLUME_STACK_SIZE - 1 &&
         step < 2 * volume_stack_size)
  {
    Intersection isect;
    if (!scene_intersect_volume(kg, &volume_ray, &isect, visibility)) {
      break;
    }

    shader_setup_from_ray(kg, stack_sd, &volume_ray, &isect);
    const VolumeStack key = dielectric_origin_key(kg, stack_sd);
    if (stack_sd->runtime_flag & SR_BACKFACING) {
      /* If ray exited the volume and never entered to that volume
       * it means that camera is inside such a volume.
       */
      bool need_add = true;
      for (int i = 0; i < enclosed_index && need_add; ++i) {
        /* If ray exited the volume and never entered to that volume
         * it means that camera is inside such a volume.
         */
        if (enclosed_volumes[i].object == key.object &&
            enclosed_volumes[i].shader == key.shader) {
          need_add = false;
        }
      }
      for (int i = 0; i < stack_index && need_add; ++i) {
        /* Don't add intersections twice. */
        VolumeStack entry = integrator_state_read_volume_stack(state, i);
        if (entry.object == key.object &&
            (key.shader == SHADER_NONE || (entry.shader & SHADER_MASK) == key.shader)) {
          need_add = false;
          break;
        }
      }
      if (need_add) {
        if (dielectric_priority(kg, stack_sd->object, stack_sd->shader)) {
          if (!dielectric_probe_origin(kg, state, &volume_ray, &isect)) return false;
        }
        else if (stack_sd->shader_flag & SD_HAS_VOLUME) {
          const VolumeStack new_entry = {stack_sd->object, stack_sd->shader};
          integrator_state_write_volume_stack(state, stack_index++, new_entry);
        }
      }
    }
    else {
      /* If ray from camera enters the volume, this volume shouldn't
       * be added to the stack on exit.
       */
      enclosed_volumes[enclosed_index++] = key;
    }

    /* Move ray forward. */
    volume_ray.tmin = intersection_t_offset(isect.t);
    volume_ray.self.object = isect.object;
    volume_ray.self.prim = isect.prim;
    ++step;
  }
  }
#  endif

  /* Write terminator. */
  const VolumeStack new_entry = {OBJECT_NONE, SHADER_NONE};
  integrator_state_write_volume_stack(state, stack_index, new_entry);
  if (kernel_data.kernel_features & KERNEL_FEATURE_NESTED_DIELECTRICS)
    dielectric_sync_volume<false>(kg, state);
#endif
  return true;
}

ccl_device void integrator_intersect_volume_stack(KernelGlobals kg, IntegratorState state)
{
#ifdef __VOLUME__
  if (!integrator_volume_stack_init(kg, state)) {
    integrator_path_cache_miss(state, DEVICE_KERNEL_INTEGRATOR_INTERSECT_VOLUME_STACK);
    return;
  }

#  ifdef __SHADOW_CATCHER__
  if (INTEGRATOR_STATE(state, path, flag) & PATH_RAY_SHADOW_CATCHER_PASS) {
    /* Volume stack re-init for shadow catcher, continue with shading of hit. */
    integrator_intersect_next_kernel_after_shadow_catcher_volume<
        DEVICE_KERNEL_INTEGRATOR_INTERSECT_VOLUME_STACK>(kg, state);
  }
  else
#  endif
  {
    /* Volume stack init for camera rays, continue with intersection of camera ray. */
    integrator_path_next(state,
                         DEVICE_KERNEL_INTEGRATOR_INTERSECT_VOLUME_STACK,
                         DEVICE_KERNEL_INTEGRATOR_INTERSECT_CLOSEST);
  }
#endif
}

CCL_NAMESPACE_END
