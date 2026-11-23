/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#pragma once

#include "kernel/camera/camera.h"

#include "kernel/film/adaptive_sampling.h"
#include "kernel/film/light_passes.h"

#include "kernel/integrator/path_state.h"
#include "kernel/integrator/photon_time.h"
#include "kernel/sample/photon_noise.h"

#include "kernel/integrator/state_util.h"
#include "kernel/sample/pattern.h"

CCL_NAMESPACE_BEGIN

ccl_device_inline Spectrum integrate_camera_sample(KernelGlobals kg,
                                                   const int sample,
                                                   const int x,
                                                   const int y,
                                                   const uint rng_pixel,
                                                   ccl_private Ray *ray,
                                                   ccl_private int &r_cache_miss,
                                                   const bool photon_only)
{
  /* Filter sampling. */
  const float2 rand_filter = (sample == 0) ? make_float2(0.5f, 0.5f) :
                                             path_rng_2D(kg, rng_pixel, sample, PRNG_FILTER);

  /* Motion blur (time) and depth of field (lens) sampling. (time, lens_x, lens_y) */
  const bool use_motionblur = kernel_data.cam.shuttertime != -1.0f;
  const bool use_dof = kernel_data.cam.aperturesize > 0.0f;
  const bool use_custom_cam = kernel_data.cam.type == CAMERA_CUSTOM;
  const float3 rand_time_lens = (use_motionblur || use_dof || use_custom_cam) ?
                                    path_rng_3D(kg, rng_pixel, sample, PRNG_LENS_TIME) :
                                    zero_float3();

  /* We use x for time and y,z for lens because in practice with Sobol
   * sampling this seems to give better convergence when an object is
   * both motion blurred and out of focus, without significantly harming
   * convergence for focal blur alone.  This is a little surprising,
   * because one would expect using x,y for lens (the 2d part) would be
   * best, since x,y are the best stratified.  Since it's not entirely
   * clear why this is, this is probably worth revisiting at some point
   * to investigate further. */
  float rand_time = rand_time_lens.x;
#ifdef __PHOTON_CAUSTICS__
  if (use_motionblur && kernel_data.integrator.use_photon_caustics &&
      (photon_only ||
       !(kernel_data.film.photon_segment_stride || kernel_data.film.photon_compact)))
  {
    /* Adaptive per-pixel counts can differ from the scheduler's emission round. */
#  ifdef __KERNEL_GPU__
    rand_time = photon_caustics_time_random(kernel_integrator_state.photon_caustics_iteration);
#  else
    rand_time = photon_caustics_time_random(kg->photon_caustics.iteration);
#  endif
  }
#endif
  const float2 rand_lens = make_float2(rand_time_lens.y, rand_time_lens.z);

  /* Generate camera ray. */
  return camera_sample(kg, x, y, rand_filter, rand_time, rand_lens, ray, r_cache_miss);
}

/* Return false to indicate that this pixel is finished.
 * Used by CPU implementation to not attempt to sample pixel for multiple samples once its known
 * that the pixel did converge. */
ccl_device bool integrator_init_from_camera(KernelGlobals kg,
                                            IntegratorState state,
                                            const ccl_global KernelWorkTile *ccl_restrict tile,
                                            ccl_global float *render_buffer,
                                            const int x_,
                                            const int y_,
                                            const int scheduled_sample)
{
  PROFILING_INIT(kg, PROFILING_RAY_SETUP);

  int x, y, sample;
  /* A complemented tile sample_offset tags an independent photon pass. The scheduler keeps
   * ordinary positive offsets; only uploaded work tiles carry this internal tag. */
  const bool photon_only = (kernel_data.film.photon_segment_stride ||
                            kernel_data.film.photon_compact) &&
                           (tile ? (tile->sample_offset & 0x80000000u) != 0 :
                                   INTEGRATOR_STATE(state, path, photon_only));

  if (tile == nullptr) {
    /* Restart from miss. Reconstruct x, y, sample from state. */
    const uint pixel_index = INTEGRATOR_STATE(state, path, render_pixel_index);
    x = pixel_index % (int)kernel_data.cam.width;
    y = pixel_index / (int)kernel_data.cam.width;
    sample = INTEGRATOR_STATE(state, path, sample);
  }
  else {
    x = x_;
    y = y_;

    /* Initialize path state to give basic buffer access and allow early outputs. */
    path_state_init(state, tile, x, y);

    /* Check whether the pixel has converged and should not be sampled anymore. */
    if (!photon_only && !film_need_sample_pixel(kg, state, render_buffer)) {
      return false;
    }
    if (photon_only && kernel_data.film.use_deep_output && kernel_data.film.photon_segment_stride) {
      /* Mixed-motion Deep completes the native tile first. Match its actual
       * per-pixel coverage count, including adaptive sampling and sample subsets. */
      ccl_global float *buffer = film_pass_pixel_render_buffer(kg, state, render_buffer);
      const auto counts = (ccl_global uint *)buffer + kernel_data.film.pass_sample_count;
      if (*counts >= counts[-kernel_data.film.photon_segment_stride]) {
        return false;
      }
    }

    /* Count the sample and get an effective sample for this pixel. */
    sample = film_write_sample(kg,
                               state,
                               render_buffer,
                               scheduled_sample,
                               photon_only ? ~tile->sample_offset : tile->sample_offset);
    if (photon_only) {
      /* Match the absolute emission round, including map reuse. The count above
       * still includes every zero-contribution camera sample. */
      sample = scheduled_sample;
    }
  }

  if (tile && kernel_data.film.deep_physical_ptr) {
    deep_physical_begin(kernel_data.film.deep_physical_ptr,
                        INTEGRATOR_STATE(state, path, render_pixel_index),
                        kernel_data.film.deep_lpe_channels, photon_only);
  }

  /* Initialize random number seed for path. */
  const uint rng_pixel = path_rng_pixel_init(kg, sample, x, y);

#ifdef __PHOTON_CAUSTICS__
  const uint photon_noise_pixel = kernel_data.integrator.use_photon_caustics ?
                                      (tile ? photon_noise_pixel_index(
                                                  x, y, kernel_data.integrator.seed) :
                                              INTEGRATOR_STATE(state, path, photon_noise_pixel)) :
                                      0;
#endif

  /* Generate camera ray. */
  Ray ray;
  int cache_miss = 0;
  Spectrum T = integrate_camera_sample(kg, sample, x, y, rng_pixel, &ray, cache_miss, photon_only);
  if (cache_miss) {
#ifdef __PHOTON_CAUSTICS__
    if (kernel_data.integrator.use_photon_caustics) {
      INTEGRATOR_STATE_WRITE(state, path, photon_noise_pixel) = photon_noise_pixel;
    }
#endif
    if (tile != nullptr) {
      integrator_path_init(state, DEVICE_KERNEL_INTEGRATOR_INIT_FROM_CAMERA);
      INTEGRATOR_STATE_WRITE(state, path, sample) = sample;
      if ((kernel_data.film.photon_segment_stride || kernel_data.film.photon_compact)) {
        INTEGRATOR_STATE_WRITE(state, path, photon_only) = photon_only;
      }
    }
    integrator_path_cache_miss(state, DEVICE_KERNEL_INTEGRATOR_INIT_FROM_CAMERA);
    return true;
  }

  if (is_zero(T)) {
    if (tile == nullptr) {
      integrator_path_terminate(
          kg, state, render_buffer, DEVICE_KERNEL_INTEGRATOR_INIT_FROM_CAMERA);
    }
    return true;
  }

  /* Write camera ray to state. */
  integrator_state_write_ray(state, &ray);

  /* (Re-)initialize path state for path integration. */
  path_state_init_integrator(kg, state, sample, rng_pixel, T);

  /* Check for the volume feature should not strictly be needed, but is safer in case
   * there is a bug updating cam.is_inside_volume when a volume is removed. */
  const bool is_inside_volume = kernel_data.cam.is_inside_volume &&
                                (kernel_data.kernel_features & KERNEL_FEATURE_VOLUME);

  if (tile == nullptr) {
    integrator_path_next(state,
                         DEVICE_KERNEL_INTEGRATOR_INIT_FROM_CAMERA,
                         is_inside_volume ? DEVICE_KERNEL_INTEGRATOR_INTERSECT_VOLUME_STACK :
                                            DEVICE_KERNEL_INTEGRATOR_INTERSECT_CLOSEST);
  }
  else {
    /* Continue with intersect_closest kernel, optionally initializing volume
     * stack before that if the camera may be inside a volume. */
    if (is_inside_volume) {
      integrator_path_init(state, DEVICE_KERNEL_INTEGRATOR_INTERSECT_VOLUME_STACK);
    }
    else {
      integrator_path_init(state, DEVICE_KERNEL_INTEGRATOR_INTERSECT_CLOSEST);
    }
  }

#ifdef __PHOTON_CAUSTICS__
  if (kernel_data.integrator.use_photon_caustics) {
    INTEGRATOR_STATE_WRITE(state, path, photon_noise_pixel) = photon_noise_pixel;
  }
#endif
  if ((kernel_data.film.photon_segment_stride || kernel_data.film.photon_compact)) {
    INTEGRATOR_STATE_WRITE(state, path, photon_only) = photon_only;
  }
  return true;
}

CCL_NAMESPACE_END
