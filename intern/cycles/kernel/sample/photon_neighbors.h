/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: Apache-2.0 */
#pragma once
#include "util/hash.h"
#include "util/math.h"

CCL_NAMESPACE_BEGIN

/* Uniform bottom-k candidate pool followed by compatibility-proportional draws.
 * Inspired by CGNS (Junkins et al., HPG 2026), adapted to photon pilot moments.
 * This does not resample final radiance or implement ReSTIR path reuse.
 * Stable photon IDs make the candidate pool independent of hash insertion order. */
constexpr int PHOTON_NEIGHBOR_CANDIDATES = 32;
struct PhotonNeighborPool {
  uint ids[PHOTON_NEIGHBOR_CANDIDATES];
  uint keys[PHOTON_NEIGHBOR_CANDIDATES];
  uint count, size, worst;
};

ccl_device_inline bool photon_neighbor_less(const uint key_a,
                                            const uint id_a,
                                            const uint key_b,
                                            const uint id_b)
{
  return key_a < key_b || (key_a == key_b && id_a < id_b);
}

ccl_device_inline void photon_neighbor_offer(ccl_private PhotonNeighborPool *pool,
                                             const uint id,
                                             const uint seed)
{
  ++pool->count;
  const uint key = hash_uint2(id, seed);
  uint slot = pool->size;
  if (slot == PHOTON_NEIGHBOR_CANDIDATES) {
    slot = pool->worst;
    if (!photon_neighbor_less(key, id, pool->keys[slot], pool->ids[slot]))
      return;
  }
  else
    ++pool->size;
  pool->ids[slot] = id;
  pool->keys[slot] = key;
  pool->worst = 0;
  for (uint i = 1; i < pool->size; ++i) {
    const uint w = pool->worst;
    if (photon_neighbor_less(pool->keys[w], pool->ids[w], pool->keys[i], pool->ids[i]))
      pool->worst = i;
  }
}

ccl_device_inline void photon_neighbor_sort(ccl_private PhotonNeighborPool *pool)
{
  for (uint i = 1; i < pool->size; ++i) {
    const uint id = pool->ids[i];
    uint j = i;
    while (j && pool->ids[j - 1] > id) {
      pool->ids[j] = pool->ids[j - 1];
      --j;
    }
    pool->ids[j] = id;
  }
}

ccl_device_inline float photon_neighbor_score(const float distance_squared,
                                              const float radius_squared,
                                              const float normal_cosine,
                                              const float axis_cosine,
                                              const float anisotropy)
{
  const float n = saturatef(normal_cosine);
  const float direction = 1.0f - .75f * saturatef(anisotropy) * (1.0f - sqr(axis_cosine));
  /* Exploration remains positive for crossing ellipses and sharp caustic peaks. */
  return max(1.0f / 16.0f,
             expf(-sqrtf(max(distance_squared / max(radius_squared, 1e-20f), 0.0f))) *
                 sqr(sqr(sqr(n))) * direction);
}

ccl_device_inline uint photon_neighbor_draw(const ccl_private PhotonNeighborPool *pool,
                                            const ccl_private float *scores,
                                            const uint draws,
                                            const float random,
                                            ccl_private float *weight)
{
  float sum = 0;
  for (uint i = 0; i < pool->size; ++i)
    sum += scores[i];
  float target = min(random, 0.99999994f) * sum;
  uint selected = pool->size - 1;
  for (uint i = 0; i < pool->size; ++i) {
    if (target < scores[i]) {
      selected = i;
      break;
    }
    target -= scores[i];
  }
  /* Inclusion probability of the uniform pool is K/N. Each conditional draw
   * has probability score/sum. Apply this coefficient to BOTH first and second
   * moment sums, never square it when estimating the original second moment. */
  *weight = float(pool->count) / float(pool->size) * sum / (float(draws) * scores[selected]);
  return pool->ids[selected];
}

CCL_NAMESPACE_END
