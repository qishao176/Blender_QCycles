/* SPDX-FileCopyrightText: 2026 Blender Foundation
 * SPDX-License-Identifier: Apache-2.0 */
#pragma once

#include "kernel/types.h"
#include "util/hash.h"
#include "util/math.h"

CCL_NAMESPACE_BEGIN

ccl_device_inline int3 photon_caustics_cell(const float3 P, const float radius)
{
  /* Hash insertion and collision filtering must quantize the identical stored
   * position identically. Fast reciprocal approximations can round an integral
   * quotient to opposite sides of a cell boundary in the two kernels, discarding
   * valid photons. Use explicit division with consistent rounding. */
#if defined(__KERNEL_CUDA__) || defined(__KERNEL_OPTIX__)
  const float3 cell = make_float3(
      __fdiv_rn(P.x, radius), __fdiv_rn(P.y, radius), __fdiv_rn(P.z, radius));
#elif defined(__KERNEL_SSE__)
  const float3 cell(_mm_div_ps(P.m128, _mm_set1_ps(radius)));
#else
  const float3 cell = P / radius;
#endif
  /* Largest float below 2^31. Leave room for signed neighbor offsets on both sides.
   * At more distant coordinates the hash is coarser; gather still tests world-space distance. */
  const float cell_limit = 2147483520.0f;
  const float3 bounded = clamp(cell, make_float3(-cell_limit), make_float3(cell_limit));
  return make_int3(int(floorf(bounded.x)), int(floorf(bounded.y)), int(floorf(bounded.z)));
}

ccl_device_inline uint photon_caustics_hash(const KernelPhotonCausticsCache cache, const int3 cell)
{
  return hash_uint3(uint(cell.x), uint(cell.y), uint(cell.z)) & (cache.hash_size - 1u);
}

CCL_NAMESPACE_END
