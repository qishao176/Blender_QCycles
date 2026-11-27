/* SPDX-FileCopyrightText: 2026 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#pragma once

#include "util/math.h"

CCL_NAMESPACE_BEGIN

/* Two emitter-domain differentials plus two local BSDF sample-cell differentials.
 * Evaluable receivers terminate the light prefix. Additional rough transmissions
 * add angular covariance and recompress the four-dimensional local beam.
 * Finite emitter positions and the full rough BSDF are sampled by Monte Carlo. */
struct PhotonDifferential {
  /* Columns 0..3 describe the spatial beam. Column 4 is d/d(wavelength in nm). */
  float3 dP[5];
  float3 dD[5];
  /* Negative disables spectral transport on this path, zero waits for dispersion. */
  float spectral_half_width;
  bool valid;
  bool rough;
};

ccl_device_inline PhotonDifferential photon_differential_init(const float3 D,
                                                              const float area,
                                                              const bool parallel)
{
  PhotonDifferential differential;
  float3 U, V;
  make_orthonormals(D, &U, &V);
  const float radius = sqrtf(max(area, 0.0f) * M_1_PI_F);
  differential.dP[0] = parallel ? radius * U : zero_float3();
  differential.dP[1] = parallel ? radius * V : zero_float3();
  differential.dD[0] = parallel ? zero_float3() : radius * U;
  differential.dD[1] = parallel ? zero_float3() : radius * V;
  differential.dP[2] = differential.dP[3] = zero_float3();
  differential.dD[2] = differential.dD[3] = zero_float3();
  differential.dP[4] = differential.dD[4] = zero_float3();
  differential.spectral_half_width = 0.0f;
  differential.rough = false;
  differential.valid = area > 0.0f && isfinite_safe(area);
  return differential;
}

ccl_device_inline void photon_differential_transfer(ccl_private PhotonDifferential *beam,
                                                    const float3 D,
                                                    const float3 Ng,
                                                    const float distance)
{
  const float denominator = dot(Ng, D);
  if (!beam->valid || fabsf(denominator) < 1e-5f) {
    beam->valid = false;
    return;
  }
  const int axes = beam->spectral_half_width > 0.0f ? 5 : 4;
  for (int i = 0; i < axes; ++i) {
    const float3 H = beam->dP[i] + distance * beam->dD[i];
    beam->dP[i] = H - D * (dot(Ng, H) / denominator);
    if (!isfinite_safe(beam->dP[i])) {
      if (i == 4) {
        beam->spectral_half_width = -1.0f;
      }
      else {
        beam->valid = false;
      }
    }
  }
}

/* Differentiate the actual selected specular branch. Fresnel chooses the branch in
 * the ordinary BSDF sampler and changes flux there, never the footprint normalization. */
ccl_device_inline void photon_differential_scatter(ccl_private PhotonDifferential *beam,
                                                   const float3 D,
                                                   const float3 N,
                                                   const float3 *dN,
                                                   const float3 outgoing,
                                                   const bool transmission,
                                                   const float eta,
                                                   const float photon_ior_slope = 0.0f)
{
  if (!beam->valid) {
    return;
  }
  const float c = -dot(N, D);
  const float r = transmission ? 1.0f / eta : 1.0f;
  const float ct2 = 1.0f - r * r * (1.0f - c * c);
  if (transmission && (!(eta > 0.0f) || ct2 < 1e-8f)) {
    beam->valid = false;
    return;
  }
  const float ct = sqrtf(max(ct2, 0.0f));
  const float3 predicted = transmission ? r * D + (r * c - ct) * N : D + 2.0f * c * N;
  /* Bump normals or a nonzero microfacet offset can invalidate the geometric derivative.
   * Keep transporting the native ray and use the bounded isotropic reconstruction. */
  if (dot(predicted, outgoing) < 0.99999f) {
    beam->valid = false;
    return;
  }
  const int axes = beam->spectral_half_width > 0.0f ? 5 : 4;
  for (int i = 0; i < axes; ++i) {
    const float dc = -dot(dN[i], D) - dot(N, beam->dD[i]);
    float3 derivative = transmission ? r * beam->dD[i] + (r - r * r * c / ct) * dc * N +
                                           (r * c - ct) * dN[i] :
                                       beam->dD[i] + 2.0f * (dc * N + c * dN[i]);
    if (i == 4 && transmission) {
      const float dr = -photon_ior_slope * r * r;
      derivative += dr * (D + (c + r * (1.0f - c * c) / ct) * N);
    }
    beam->dD[i] = derivative - outgoing * dot(outgoing, derivative);
    if (!isfinite_safe(beam->dD[i])) {
      if (i == 4) {
        beam->spectral_half_width = -1.0f;
      }
      else {
        beam->valid = false;
      }
    }
  }
}

/* Add independent angular sample-cell widths. Position lies in the hit tangent
 * plane and direction in the outgoing ray tangent plane, so four columns suffice.
 * Recompression preserves position/direction cross covariance through repeated
 * rough glass interfaces; it does not grow the per-photon stored record. */
ccl_device_inline void photon_differential_add_spread(ccl_private PhotonDifferential *beam,
                                                      const float3 Ng,
                                                      const float3 D,
                                                      const float3 spread[2])
{
  float3 X, Y, U, V;
  make_orthonormals(Ng, &X, &Y);
  make_orthonormals(D, &U, &V);
  float C[4][4] = {};
  for (int k = 0; k < 6; ++k) {
    const float3 p = k < 4 ? beam->dP[k] : zero_float3();
    const float3 d = k < 4 ? beam->dD[k] : spread[k - 4];
    const float a[4] = {dot(p, X), dot(p, Y), dot(d, U), dot(d, V)};
    for (int i = 0; i < 4; ++i) {
      for (int j = 0; j <= i; ++j)
        C[i][j] += a[i] * a[j];
    }
  }
  float L[4][4] = {};
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j <= i; ++j) {
      float v = C[i][j];
      for (int k = 0; k < j; ++k)
        v -= L[i][k] * L[j][k];
      L[i][j] = i == j ? sqrtf(max(v, 0.0f)) : (L[j][j] > 0.0f ? v / L[j][j] : 0.0f);
    }
  }
  for (int i = 0; i < 4; ++i) {
    beam->dP[i] = X * L[0][i] + Y * L[1][i];
    beam->dD[i] = U * L[2][i] + V * L[3][i];
    beam->valid &= isfinite_safe(beam->dP[i]) && isfinite_safe(beam->dD[i]);
  }
  beam->rough = true;
}

/* Q is the inverse tangent covariance. K(x) = 2 sqrt(det Q)/pi * (1 - x^T Q x)
 * on x^T Q x < 1 integrates to one, including after size/anisotropy clamping. */
ccl_device_inline float4 photon_differential_footprint(const PhotonDifferential beam,
                                                       const float3 N,
                                                       const float min_radius,
                                                       const float max_radius)
{
  const float inverse_radius_squared = 1.0f / sqr(max_radius);
  const float4 fallback = make_float4(inverse_radius_squared,
                                      0.0f,
                                      inverse_radius_squared,
                                      2.0f * M_1_PI_F * inverse_radius_squared);
  if (!beam.valid) {
    return fallback;
  }
  float3 U, V;
  make_orthonormals(N, &U, &V);
  const float2 a = make_float2(dot(beam.dP[0], U), dot(beam.dP[0], V));
  const float2 b = make_float2(dot(beam.dP[1], U), dot(beam.dP[1], V));
  float xx = a.x * a.x + b.x * b.x;
  float xy = a.x * a.y + b.x * b.y;
  float yy = a.y * a.y + b.y * b.y;
  /* Independent local BSDF sample-cell axes: add their covariance, not their
   * vectors, to the emitter footprint. The full BSDF lobe is still sampled. */
  for (int i = 2; i < 4; ++i) {
    const float2 r = make_float2(dot(beam.dP[i], U), dot(beam.dP[i], V));
    xx += r.x * r.x;
    xy += r.x * r.y;
    yy += r.y * r.y;
  }
  if (beam.spectral_half_width > 0.0f && !beam.rough) {
    /* A symmetric wavelength interval has variance h^2/3. The normalized 2D
     * parabolic kernel has covariance shape/6, hence the factor of two. */
    const float2 v = beam.spectral_half_width *
                     make_float2(dot(beam.dP[4], U), dot(beam.dP[4], V));
    xx += 2.0f * v.x * v.x;
    xy += 2.0f * v.x * v.y;
    yy += 2.0f * v.y * v.y;
  }
  const float gap = hypotf(xx - yy, 2.0f * xy);
  const float lo = max(0.5f * (xx + yy - gap), 0.0f);
  const float hi = max(0.5f * (xx + yy + gap), 0.0f);
  if (!isfinite_safe(hi)) {
    return fallback;
  }
  const float upper = clamp(hi, sqr(min_radius), sqr(max_radius));
  const float lower = clamp(lo, max(sqr(min_radius), upper / 1024.0f), upper);
  const float scale = (gap > 1e-20f) ? (upper - lower) / gap : 0.0f;
  const float cxx = lower + scale * (xx - lo);
  const float cxy = scale * xy;
  const float cyy = lower + scale * (yy - lo);
  const float det = upper * lower;
  return make_float4(cyy / det, -cxy / det, cxx / det, 2.0f * M_1_PI_F / sqrtf(det));
}

CCL_NAMESPACE_END
