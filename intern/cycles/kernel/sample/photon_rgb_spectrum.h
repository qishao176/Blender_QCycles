/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: Apache-2.0 */

#pragma once

#include "util/math.h"

CCL_NAMESPACE_BEGIN

/* A nonnegative, piecewise-linear illuminant model for RGB assets without a
 * measured SPD. The peaks come from the working primaries and the native CIE
 * observer. The three bands sum to one, so equal RGB represents D65. */
ccl_device_inline float3 photon_rgb_spectral_basis(const float wavelength, const float3 peaks)
{
  const float red = saturatef((wavelength - peaks.y) / (peaks.x - peaks.y));
  const float blue = saturatef((peaks.y - wavelength) / (peaks.y - peaks.z));
  return make_float3(red, 1.0f - red - blue, blue);
}

/* Evaluate the same scalar illuminant at this wavelength for all three film
 * channels. Each observer channel is calibrated against the integral of this
 * illuminant, to preserve the input RGB exposure with the nonnegative photon
 * film response. The native wavelength sampling PDF is applied at merge. */
ccl_device_inline float3 photon_rgb_at_wavelength(const float3 rgb,
                                                  const float3 basis,
                                                  const float3 response_r,
                                                  const float3 response_g,
                                                  const float3 response_b)
{
  const float3 flux = max(rgb, zero_float3());
  if (flux.x == flux.y && flux.y == flux.z) {
    return flux;
  }
  const float3 chromaticity = flux / reduce_max(flux);
  const float emission = dot(chromaticity, basis);
  const float3 integral = make_float3(dot(chromaticity, response_r),
                                      dot(chromaticity, response_g),
                                      dot(chromaticity, response_b));
  return safe_divide(chromaticity, integral) * (emission * reduce_max(flux));
}

CCL_NAMESPACE_END
