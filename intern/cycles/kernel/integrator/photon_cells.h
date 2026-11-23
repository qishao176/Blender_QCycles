/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: Apache-2.0 */
#pragma once

#include "kernel/integrator/photon_spatial.h"
#include "kernel/sample/photon_cell.h"
#include "kernel/sample/photon_path.h"
#include "kernel/sample/photon_spectrum.h"
#include "util/atomic.h"

CCL_NAMESPACE_BEGIN
#ifdef __PHOTON_CAUSTICS__

ccl_device_inline uint photon_bounce_pack(const uint reflection,
                                          const uint transmission,
                                          const uint passthrough)
{
  const bool overflow = reflection == 1024 || transmission == 1024;
  const uint r = overflow ? reflection / 1024 : reflection;
  const uint t = overflow ? transmission / 1024 : transmission;
  return r | (t << PHOTON_CAUSTICS_TRANSMISSION_SHIFT) |
         (passthrough << PHOTON_CAUSTICS_PASSTHROUGH_SHIFT) |
         (overflow ? PHOTON_CAUSTICS_BOUNCE_OVERFLOW : 0u);
}

ccl_device_inline uint photon_reflection_depth(const uint packed)
{
  return (packed & PHOTON_CAUSTICS_BOUNCE_MASK) *
         ((packed & PHOTON_CAUSTICS_BOUNCE_OVERFLOW) ? 1024u : 1u);
}

ccl_device_inline uint photon_transmission_depth(const uint packed)
{
  return ((packed >> PHOTON_CAUSTICS_TRANSMISSION_SHIFT) & PHOTON_CAUSTICS_BOUNCE_MASK) *
         ((packed & PHOTON_CAUSTICS_BOUNCE_OVERFLOW) ? 1024u : 1u);
}

ccl_device_inline uint photon_passthrough_depth(const uint packed)
{
  return (packed >> PHOTON_CAUSTICS_PASSTHROUGH_SHIFT) & PHOTON_CAUSTICS_BOUNCE_MASK;
}

ccl_device_inline uint photon_event_count(const uint packed)
{
  return photon_reflection_depth(packed) + photon_transmission_depth(packed) +
         photon_passthrough_depth(packed);
}

ccl_device_inline int3 photon_cell_corner(const int3 base, const uint c)
{
  return make_int3(base.x + int(c & 1), base.y + int((c >> 1) & 1), base.z + int((c >> 2) & 1));
}

ccl_device_inline float3 photon_cell_fraction(const float3 P, const int3 base, const float width)
{
  return clamp((P - make_float3(float(base.x), float(base.y), float(base.z)) * width) / width,
               zero_float3(),
               one_float3());
}

ccl_device_inline uint photon_cell_spectrum(const KernelPhotonCausticsCache cache, const uint i)
{
  return (cache.photons[i].pad & PHOTON_CAUSTICS_SPECTRAL) ?
             1 + photon_wavelength_group(i, cache.iteration, cache.wavelength_groups) :
             0;
}

ccl_device_inline uint photon_cell_class_hash(KernelGlobals kg,
                                              const KernelPhotonCausticsCache cache,
                                              const uint i)
{
  const ccl_global KernelPhotonCaustic *p = &cache.photons[i];
  uint h = hash_uint3(uint(p->receiver_object), uint(p->light), photon_cell_spectrum(cache, i));
  if ((kernel_data.integrator.photon_reconstruction_filter & PHOTON_RECONSTRUCTION_FILTER_PATH) &&
      cache.reconstruction_metadata)
  {
    h = hash_uint2(h, cache.reconstruction_metadata[i].path_hash);
  }
  return h;
}

ccl_device_inline bool photon_cell_class_equal(KernelGlobals kg,
                                               const KernelPhotonCausticsCache cache,
                                               const uint a,
                                               const uint b)
{
  const ccl_global KernelPhotonCaustic *pa = &cache.photons[a], *pb = &cache.photons[b];
  if (pa->receiver_object != pb->receiver_object || pa->light != pb->light ||
      photon_cell_spectrum(cache, a) != photon_cell_spectrum(cache, b))
  {
    return false;
  }
  if (kernel_data.integrator.photon_reconstruction_filter & PHOTON_RECONSTRUCTION_FILTER_PATH) {
    if (!cache.reconstruction_metadata || !cache.path_indices || !cache.events ||
        cache.reconstruction_metadata[a].path_hash != cache.reconstruction_metadata[b].path_hash ||
        pa->pad != pb->pad)
    {
      return false;
    }
    const uint length = photon_event_count(pa->pad);
    if (length > cache.event_stride || cache.path_indices[a] >= cache.capacity ||
        cache.path_indices[b] >= cache.capacity)
    {
      return false;
    }
    return photon_path_events_equal(
        cache.events + size_t(cache.path_indices[a]) * cache.event_stride,
        cache.events + size_t(cache.path_indices[b]) * cache.event_stride,
        length,
        SHADER_MASK);
  }
  return true;
}

ccl_device_inline uint photon_cell_slot(KernelGlobals kg,
                                        const KernelPhotonCausticsCache cache,
                                        const uint index,
                                        const int3 node,
                                        const uint insert_key = 0)
{
  const uint mask = cache.hash_size - 1;
  const uint h = hash_uint2(hash_uint3(uint(node.x), uint(node.y), uint(node.z)),
                            photon_cell_class_hash(kg, cache, index));
  for (uint probe = 0; probe < PHOTON_CELL_PROBES; probe++) {
    const uint slot = (h + probe) & mask;
    uint key;
    if (insert_key) {
#  ifdef __KERNEL_GPU__
      key = atomicCAS(&cache.cell_statistics[slot].x, 0u, insert_key);
#  else
      key = atomic_cas_uint32(&cache.cell_statistics[slot].x, 0u, insert_key);
#  endif
      if (!key) {
        return slot;
      }
    }
    else {
      key = cache.cell_statistics[slot].x;
      if (!key) {
        return cache.hash_size;
      }
    }
    /* The key refers only to immutable, already-published photon data. No
     * partially initialized key payload or spin lock is visible to writers. */
    const uint other = (key - 1) >> 3;
    const int3 other_node = photon_cell_corner(
        photon_caustics_cell(cache.photons[other].P, cache.radius), (key - 1) & 7);
    if (node.x == other_node.x && node.y == other_node.y && node.z == other_node.z &&
        photon_cell_class_equal(kg, cache, index, other))
    {
      return slot;
    }
  }
  return cache.hash_size;
}

ccl_device void photon_cell_build(KernelGlobals kg,
                                  const KernelPhotonCausticsCache cache,
                                  const uint stored_index)
{
  if (!cache.cell_statistics || stored_index >= *cache.count) {
    return;
  }
  const uint index = cache.stored_indices[stored_index];
  const ccl_global KernelPhotonCaustic *p = &cache.photons[index];
  if (p->receiver_object < 0) {
    return;
  }
  const int3 base = photon_caustics_cell(p->P, cache.radius);
  uint weights[8];
  photon_cell_weights(photon_cell_fraction(p->P, base, cache.radius), weights);
  for (uint c = 0; c < 8; c++) {
    if (!weights[c]) {
      continue;
    }
    const uint slot = photon_cell_slot(
        kg, cache, index, photon_cell_corner(base, c), 1 + (index << 3) + c);
    if (slot == cache.hash_size) {
      /* Any overflow invalidates the map's coarse proposal, never the photons. */
      atomic_fetch_and_add_uint32(&cache.cell_statistics[cache.hash_size].x, 1);
    }
    else {
      atomic_fetch_and_add_uint32(&cache.cell_statistics[slot].y, weights[c]);
    }
  }
}

ccl_device_inline float photon_cell_occupancy(KernelGlobals kg,
                                              const KernelPhotonCausticsCache cache,
                                              const uint index)
{
  if (!cache.cell_statistics || cache.cell_statistics[cache.hash_size].x) {
    return -1;
  }
  const float3 P = cache.photons[index].P;
  const int3 base = photon_caustics_cell(P, cache.radius);
  const float3 f = photon_cell_fraction(P, base, cache.radius);
  uint own[8];
  photon_cell_weights(f, own);
  float count = 0;
  for (uint c = 0; c < 8; c++) {
    const float w = photon_cell_weight(f, c);
    if (w > 0) {
      const uint slot = photon_cell_slot(kg, cache, index, photon_cell_corner(base, c));
      if (slot != cache.hash_size) {
        count += w * max(float(cache.cell_statistics[slot].y) - float(own[c]), 0.0f) /
                 PHOTON_CELL_MASS;
      }
    }
  }
  return max(count, 0.0f);
}

#endif
CCL_NAMESPACE_END
