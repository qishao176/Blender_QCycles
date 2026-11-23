/* SPDX-FileCopyrightText: 2026 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#pragma once

#include "kernel/film/write.h"

CCL_NAMESPACE_BEGIN

/* Produce the ordinary film layout before display, denoising, LPE expressions and export.
 * The two raw sums are never changed by this operation. The normal pass accessor divides
 * by output_count, giving native_sum/native_count + photon_sum/photon_count.
 * A subset entirely beyond the native budget uses the photon count instead. */
ccl_device_inline void film_photon_merge(KernelGlobals kg,
                                         ccl_global float *render_buffer,
                                         const int pixel_index)
{
  const int segment = kernel_data.film.photon_segment_stride;
  if (!segment) {
    return;
  }
  ccl_global float *output = render_buffer + uint64_t(pixel_index) * kernel_data.film.pass_stride;
  const ccl_global float *native = output + segment;
  const ccl_global float *photon = native + segment;
  const uint native_count = __float_as_uint(native[kernel_data.film.pass_sample_count]);
  const uint photon_count = __float_as_uint(photon[kernel_data.film.pass_sample_count]);
  const uint output_count = native_count ? native_count : photon_count;
  const ccl_global float *metadata = native_count ? native : photon;
  const float photon_scale = photon_count ? float(output_count) / float(photon_count) : 0.0f;
  for (int i = 0; i < segment; ++i) {
    const uint mode = kernel_data_fetch(photon_caustics_merge, i);
    if (mode != 2) {
      output[i] = mode == 1 ? native[i] + photon[i] * photon_scale : metadata[i];
    }
  }
  /* Counts are integer bit patterns, so copying them through floating-point addition is
   * unsafe on devices that flush subnormals. */
  output[kernel_data.film.pass_sample_count] = __uint_as_float(output_count);
  output[kernel_data.film.pass_photon_sample_count] = __uint_as_float(photon_count);
}

CCL_NAMESPACE_END
