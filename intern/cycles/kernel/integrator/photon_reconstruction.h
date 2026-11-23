/* SPDX-FileCopyrightText: 2026 Blender Foundation
 * SPDX-License-Identifier: Apache-2.0 */
#pragma once

#include "kernel/integrator/photon_cells.h"
#include "kernel/integrator/photon_spatial.h"
#include "kernel/sample/photon_neighbors.h"
#include "kernel/sample/photon_overlap.h"
#include "kernel/sample/photon_path.h"
#include "kernel/sample/photon_reconstruction.h"
#include "kernel/sample/photon_region.h"
#include "kernel/sample/photon_spectrum.h"
#include "util/types_normal.h"

CCL_NAMESPACE_BEGIN

#ifdef __PHOTON_CAUSTICS__
/* This policy filters pilot statistics only. Photon transport, final BSDF
 * evaluation and LPE replay retain every stored contribution. A hash is only
 * an early rejection: exact ordered events resolve equal hashes/collisions. */
ccl_device_inline bool photon_reconstruction_history_compatible(
    KernelGlobals kg, const KernelPhotonCausticsCache cache, const uint a, const uint b)
{
  /* Wavelength-dependent footprints are distinct transport slices. Sharing
   * their pilot density would make spectral edges look artificially dense and
   * blur/displace dispersion. This filters statistics, never final light paths. */
  if (cache.wavelength_groups > 1 &&
      ((cache.photons[a].pad | cache.photons[b].pad) & PHOTON_CAUSTICS_SPECTRAL) &&
      photon_wavelength_group(a, cache.iteration, cache.wavelength_groups) !=
          photon_wavelength_group(b, cache.iteration, cache.wavelength_groups))
  {
    return false;
  }
  const int filter = kernel_data.integrator.photon_reconstruction_filter;
  if (filter == PHOTON_RECONSTRUCTION_FILTER_LEGACY || a == b) {
    return true;
  }
  if (!cache.reconstruction_metadata) {
    return false;
  }
  const KernelPhotonReconstructionMetadata ma = cache.reconstruction_metadata[a];
  const KernelPhotonReconstructionMetadata mb = cache.reconstruction_metadata[b];
  if (filter & PHOTON_RECONSTRUCTION_FILTER_NORMAL) {
    packed_normal na, nb;
    na.value = ma.previous_normal;
    nb.value = mb.previous_normal;
    if (ma.previous_normal != mb.previous_normal &&
        dot(na.decode(), nb.decode()) < kernel_data.integrator.photon_reconstruction_normal_cos)
    {
      return false;
    }
  }
  if (filter & PHOTON_RECONSTRUCTION_FILTER_PATH) {
    const ccl_global KernelPhotonCaustic *pa = &cache.photons[a], *pb = &cache.photons[b];
    if (ma.path_hash != mb.path_hash || pa->pad != pb->pad || !cache.events || !cache.path_indices)
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

/* Continuity changes statistical weights only, never photon energy, LPE
 * membership, or the final receiver's BSDF. Distinct last-interface events must
 * not share a local differential fit; optional full-path filtering is separate. */
ccl_device_inline float photon_reconstruction_beam_weight(KernelGlobals kg,
                                                          const KernelPhotonCausticsCache cache,
                                                          const uint a,
                                                          const uint b)
{
  if (!cache.beam_metadata || a == b) {
    return 1.0f;
  }
  const KernelPhotonBeamMetadata ma = cache.beam_metadata[a];
  const KernelPhotonBeamMetadata mb = cache.beam_metadata[b];
  if (ma.object != mb.object || ma.shader != mb.shader || ma.event != mb.event ||
      cache.photons[a].pad != cache.photons[b].pad)
  {
    return 0.0f;
  }
  packed_normal na, nb;
  na.value = ma.previous_normal;
  nb.value = mb.previous_normal;
  return photon_region_continuity(ma.previous_position,
                                  na.decode(),
                                  ma.normal_derivative,
                                  ma.valid,
                                  mb.previous_position,
                                  nb.decode(),
                                  mb.normal_derivative,
                                  mb.valid,
                                  kernel_data.integrator.photon_reconstruction_normal_cos);
}

ccl_device_inline void photon_reconstruct_overlap_neighbor(
    KernelGlobals kg,
    const KernelPhotonCausticsCache cache,
    const uint index,
    const uint other_index,
    const float power_scale,
    const float radius,
    const float maximum,
    const float strength,
    const float3 U,
    const float3 V,
    const float3 own_covariance,
    const PhotonEllipseFrame frame,
    const float selection_weight,
    ccl_private PhotonOverlapStatistics *overlap,
    ccl_private PhotonRadiusStatistics *density,
    ccl_private PhotonEllipseStatistics *ellipses)
{
  const ccl_global KernelPhotonCaustic *other = &cache.photons[other_index];
  const float3 delta = float3(other->P) - float3(cache.photons[index].P);
  packed_normal packed;
  packed.value = other->normal;
  const float3 other_normal = packed.decode();
  const float weight = photon_reconstruction_beam_weight(kg, cache, index, other_index);
  const float3 power = float3(other->power) / power_scale;
  const float4 original = cache.pilot_footprints[other_index];
  const float3 other_covariance = photon_overlap_transport(
      photon_overlap_covariance(original), other_normal, U, V);
  photon_overlap_add(overlap,
                     own_covariance,
                     other_covariance,
                     make_float2(dot(delta, U), dot(delta, V)),
                     power,
                     weight,
                     maximum,
                     selection_weight);
  if (strength > 0.0f) {
    photon_radius_add(density, delta / radius, power * weight, selection_weight);
    if (frame.peak > 0.0f) {
      photon_ellipse_add(ellipses,
                         frame,
                         original,
                         other_normal,
                         -delta,
                         average(power) * weight,
                         selection_weight);
    }
  }
  else {
    ++density->count;
  }
}

template<bool sampled>
ccl_device void photon_reconstruct_overlap(KernelGlobals kg,
                                           const KernelPhotonCausticsCache cache,
                                           const uint index,
                                           const float cell_size)
{
  if (!cache.pilot_footprints) {
    return;
  }
  ccl_global KernelPhotonCaustic *photon = &cache.photons[index];
  const float max_radius = cache.radius * kernel_data.integrator.photon_reconstruction_pilot_scale;
  float radius = max_radius;
  const float maximum = kernel_data.integrator.photon_adaptive_max_scale;
  if (maximum <= 1.0f) {
    return;
  }
  const float strength = kernel_data.integrator.use_photon_gradient_guard ?
                             kernel_data.integrator.photon_gradient_strength :
                             0.0f;
  const float3 own_power = float3(photon->power);
  const float power_scale = max(max(own_power.x, own_power.y), own_power.z);
  if (!(radius > 0.0f && power_scale > 0.0f) || !isfinite_safe(power_scale)) {
    return;
  }
  packed_normal packed;
  packed.value = photon->normal;
  const float3 normal = packed.decode();
  float3 U, V;
  make_orthonormals(normal, &U, &V);
  const float3 own_covariance = photon_overlap_covariance(cache.pilot_footprints[index]);
  const PhotonEllipseFrame frame = photon_ellipse_frame(cache.pilot_footprints[index], normal);
  const float occupancy = photon_cell_occupancy(kg, cache, index);
  const bool cell_guided = occupancy >= 0;
  if (cell_guided) {
    const float minimum = min(max_radius, max(cache.radius, frame.radius * maximum));
    radius = photon_cell_pilot_radius(occupancy,
                                      cache.radius,
                                      minimum,
                                      max_radius,
                                      float(kernel_data.integrator.photon_adaptive_neighbors));
  }
  const uint draws = kernel_data.integrator.photon_neighbor_samples;
  const uint seed = hash_uint3(index, cache.iteration, 0x639d2ef1u);
  PhotonNeighborPool pool = {};
  PhotonOverlapStatistics overlap = {};
  PhotonRadiusStatistics density = {};
  PhotonEllipseStatistics ellipses = {};
  const int3 center = photon_caustics_cell(photon->P, cell_size);
  /* One pilot scan, with at most one wider cell-guided retry; all neighbors
   * are immutable originals. No iterative reads of kernels concurrently resized by workers.
   * The pilot may omit farther overlapping kernels: this is a local bandwidth
   * heuristic, not a complete overlap integral over the photon map. */
  for (int query = 0; query < (cell_guided ? 2 : 1); query++) {
    pool = {};
    overlap = {};
    density = {};
    ellipses = {};
    float compatible_mass = 0;
    const float3 extent = make_float3(radius, radius, radius);
    const int3 low = cell_guided ? photon_caustics_cell(float3(photon->P) - extent, cell_size) :
                                   make_int3(center.x - 1, center.y - 1, center.z - 1);
    const int3 high = cell_guided ? photon_caustics_cell(float3(photon->P) + extent, cell_size) :
                                    make_int3(center.x + 1, center.y + 1, center.z + 1);
    for (int z = low.z; z <= high.z; ++z) {
      for (int y = low.y; y <= high.y; ++y) {
        for (int x = low.x; x <= high.x; ++x) {
          const int3 cell = make_int3(x, y, z);
          uint entry = cache.hash[photon_caustics_hash(cache, cell)];
          while (entry && entry <= cache.capacity) {
            const uint other_index = entry - 1;
            const ccl_global KernelPhotonCaustic *other = &cache.photons[other_index];
            entry = other->next;
            const int3 other_cell = photon_caustics_cell(other->P, cell_size);
            if (other_cell.x != cell.x || other_cell.y != cell.y || other_cell.z != cell.z ||
                other->receiver_object != photon->receiver_object || other->light != photon->light)
            {
              continue;
            }
            const float3 delta = float3(other->P) - float3(photon->P);
            packed_normal other_packed;
            other_packed.value = other->normal;
            const float3 other_normal = other_packed.decode();
            if (len_squared(delta) >= sqr(radius) || dot(normal, other_normal) < .9f ||
                fabsf(dot(delta, normal)) > .1f * cache.radius ||
                !photon_reconstruction_history_compatible(kg, cache, index, other_index))
            {
              continue;
            }
            const float weight = photon_reconstruction_beam_weight(kg, cache, index, other_index);
            const float3 power = float3(other->power) / power_scale;
            if (!(weight > 0.0f) || !isfinite_safe(power) || is_zero(power)) {
              continue;
            }
            if (cell_guided && other_index != index) {
              compatible_mass += weight * sqr(max(1.0f - len_squared(delta) / sqr(radius), 0.0f));
            }
            if (sampled && other_index != index) {
              photon_neighbor_offer(&pool, other_index, seed);
            }
            else {
              photon_reconstruct_overlap_neighbor(kg,
                                                  cache,
                                                  index,
                                                  other_index,
                                                  power_scale,
                                                  radius,
                                                  maximum,
                                                  strength,
                                                  U,
                                                  V,
                                                  own_covariance,
                                                  frame,
                                                  1.0f,
                                                  &overlap,
                                                  &density,
                                                  &ellipses);
            }
          }
        }
      }
    }
    /* Coarse counts may include incompatible normals or paths. Smoothly widen
     * once when the actual compatible evidence is sparse; no neighbor identity
     * or final photon contribution is discarded by the occupancy proposal. */
    const float refined = photon_cell_refine_radius(
        radius,
        max_radius,
        compatible_mass,
        float(kernel_data.integrator.photon_adaptive_neighbors));
    if (!cell_guided || query || !(refined > radius)) {
      break;
    }
    radius = refined;
  }
  if (sampled && pool.size) {
    photon_neighbor_sort(&pool);
    float scores[PHOTON_NEIGHBOR_CANDIDATES];
    const bool exact = pool.count <= draws;
    if (!exact) {
      for (uint i = 0; i < pool.size; ++i) {
        const uint other_index = pool.ids[i];
        packed_normal packed_other;
        packed_other.value = cache.photons[other_index].normal;
        const float3 other_normal = packed_other.decode();
        const PhotonEllipseFrame other = photon_ellipse_frame(cache.pilot_footprints[other_index],
                                                              other_normal);
        scores[i] = photon_neighbor_score(
            len_squared(float3(cache.photons[other_index].P) - float3(photon->P)),
            sqr(radius),
            dot(normal, other_normal),
            dot(frame.axis, other.axis),
            frame.anisotropy * other.anisotropy);
      }
    }
    const uint evaluations = exact ? pool.size : draws;
    for (uint i = 0; i < evaluations; ++i) {
      float selection_weight = 1.0f;
      const uint other_index = exact ?
                                   pool.ids[i] :
                                   photon_neighbor_draw(&pool,
                                                        scores,
                                                        draws,
                                                        hash_uint2_to_float(i, seed ^ 0xb28342f7u),
                                                        &selection_weight);
      photon_reconstruct_overlap_neighbor(kg,
                                          cache,
                                          index,
                                          other_index,
                                          power_scale,
                                          radius,
                                          maximum,
                                          strength,
                                          U,
                                          V,
                                          own_covariance,
                                          frame,
                                          selection_weight,
                                          &overlap,
                                          &density,
                                          &ellipses);
    }
    /* The non-linear bandwidth heuristic uses moment estimates; it is not an
     * unbiased radius estimator. The photon being resized was evaluated exactly. */
    density.count = pool.count + 1;
  }
  const float gradient = strength > 0.0f ? photon_radius_gradient(&density, false, &ellipses) :
                                           0.0f;
  const float scale = photon_overlap_scale(&overlap,
                                           float(kernel_data.integrator.photon_adaptive_neighbors),
                                           maximum,
                                           strength,
                                           gradient);
  /* A scalar for this photon, not a common scalar for the map. Scaling Q and
   * peak by the same inverse area preserves integral, aspect ratio and axes. */
  photon->footprint *= 1.0f / sqr(scale);
}

ccl_device void photon_reconstruct_region(KernelGlobals kg,
                                          const KernelPhotonCausticsCache cache,
                                          const uint index,
                                          const float cell_size)
{
  ccl_global KernelPhotonCaustic *photon = &cache.photons[index];
  const float pilot_radius = cache.radius *
                             kernel_data.integrator.photon_reconstruction_pilot_scale;
  const float3 power = float3(photon->power);
  const float power_scale = max(max(power.x, power.y), power.z);
  if (!(power_scale > 0.0f && pilot_radius > 0.0f) || !isfinite_safe(power_scale)) {
    return;
  }
  packed_normal packed;
  packed.value = photon->normal;
  const float3 normal = packed.decode();
  float3 U, V;
  make_orthonormals(normal, &U, &V);
  PhotonRegionStatistics statistics = {};
  PhotonEllipseStatistics ellipses = {};
  const PhotonEllipseFrame frame = cache.pilot_footprints ?
                                       photon_ellipse_frame(cache.pilot_footprints[index],
                                                            normal) :
                                       PhotonEllipseFrame{};
  const int3 center = photon_caustics_cell(photon->P, cell_size);
  /* cell_size includes both the support and the independent statistical radius.
   * Full neighborhoods avoid traversal-order-dependent bandwidths. This pass is
   * intentionally separate from bounded shading gather and runs once per round. */
  for (int z = -1; z <= 1; ++z) {
    for (int y = -1; y <= 1; ++y) {
      for (int x = -1; x <= 1; ++x) {
        const int3 cell = make_int3(center.x + x, center.y + y, center.z + z);
        uint entry = cache.hash[photon_caustics_hash(cache, cell)];
        while (entry && entry <= cache.capacity) {
          const uint other_index = entry - 1;
          const ccl_global KernelPhotonCaustic *other = &cache.photons[other_index];
          entry = other->next;
          const int3 other_cell = photon_caustics_cell(other->P, cell_size);
          if (other_cell.x != cell.x || other_cell.y != cell.y || other_cell.z != cell.z ||
              other->receiver_object != photon->receiver_object || other->light != photon->light)
          {
            continue;
          }
          const float3 delta = float3(other->P) - float3(photon->P);
          packed_normal other_normal;
          other_normal.value = other->normal;
          if (len_squared(delta) >= sqr(pilot_radius) ||
              dot(normal, other_normal.decode()) < .9f ||
              fabsf(dot(delta, normal)) > .1f * cache.radius ||
              !photon_reconstruction_history_compatible(kg, cache, index, other_index))
          {
            continue;
          }
          const float weight = photon_reconstruction_beam_weight(kg, cache, index, other_index);
          const float3 normalized_power = float3(other->power) / power_scale;
          if (!(weight > 0.0f) || !isfinite_safe(normalized_power)) {
            continue;
          }
          photon_region_add(&statistics,
                            make_float2(dot(delta, U), dot(delta, V)) / pilot_radius,
                            normalized_power,
                            weight);
          if (frame.peak > 0.0f) {
            photon_ellipse_add(&ellipses,
                               frame,
                               cache.pilot_footprints[other_index],
                               other_normal.decode(),
                               -delta,
                               weight * average(normalized_power));
          }
        }
      }
    }
  }
  /* Rough footprints have a larger pre-existing support. Never shrink it and
   * never extend beyond the global hash/gather envelope. */
  const float rough_scale = photon->footprint.w < 0.0f ? 4.0f : 1.0f;
  photon->footprint = photon_region_footprint(
      photon->footprint,
      &statistics,
      pilot_radius,
      cache.radius * rough_scale * kernel_data.integrator.photon_adaptive_max_scale,
      float(kernel_data.integrator.photon_adaptive_neighbors),
      kernel_data.integrator.use_photon_gradient_guard ?
          kernel_data.integrator.photon_gradient_strength :
          0.0f,
      kernel_data.integrator.photon_reconstruction_method == PHOTON_RECONSTRUCTION_ANISOTROPIC,
      photon_ellipse_evidence(&ellipses));
}

/* Analyze only completed maps, one work item per stored photon. A compact list
 * avoids serializing dense hash buckets after guiding concentrates emission.
 * It changes scheduling only: stable emission slots still identify noise and LPEs.
 * Neighbors use immutable pilot footprints; resized kernels are never read.
 * Each published slot is processed exactly once after all emission tasks finish.
 * Scaling Q and its peak together preserves each kernel's normalized integral. */
ccl_device void photon_reconstruct_map(KernelGlobals kg,
                                       const KernelPhotonCausticsCache cache,
                                       const uint stored_index,
                                       const float cell_size)
{
  if (!kernel_data.integrator.use_photon_adaptive_radius || !cache.stored_indices ||
      stored_index >= *cache.count)
  {
    return;
  }
  const float maximum = kernel_data.integrator.photon_adaptive_max_scale;
  const float target = float(kernel_data.integrator.photon_adaptive_neighbors);
  const float strength = kernel_data.integrator.use_photon_gradient_guard ?
                             kernel_data.integrator.photon_gradient_strength :
                             0.0f;
  ccl_global KernelPhotonCaustic *photon = &cache.photons[cache.stored_indices[stored_index]];
  const bool volume = photon->receiver_object < 0;
  if (!volume && kernel_data.integrator.photon_reconstruction_method ==
                     PHOTON_RECONSTRUCTION_ELLIPSE_COVERAGE)
  {
    if (kernel_data.integrator.photon_neighbor_samples > 0) {
      photon_reconstruct_overlap<true>(kg, cache, cache.stored_indices[stored_index], cell_size);
    }
    else {
      photon_reconstruct_overlap<false>(kg, cache, cache.stored_indices[stored_index], cell_size);
    }
    return;
  }
  if (!volume &&
      kernel_data.integrator.photon_reconstruction_method != PHOTON_RECONSTRUCTION_LEGACY)
  {
    photon_reconstruct_region(kg, cache, cache.stored_indices[stored_index], cell_size);
    return;
  }
  const float base_area = 2.0f * M_1_PI_F / fabsf(photon->footprint.w);
  const float base_radius = volume ? cache.radius : sqrtf(base_area);
  const float local_radius = maximum * base_radius;
  const bool independent = !volume &&
                           kernel_data.integrator.photon_reconstruction_pilot_scale > 1.0f;
  /* The independent pilot extends coverage statistics, never shrinks the local
   * neighborhood of an already wide original kernel. Gradient/ellipse evidence
   * remains local so distant caustic lobes do not become local edge estimates. */
  const float pilot_radius =
      independent ? max(local_radius,
                        cache.radius * kernel_data.integrator.photon_reconstruction_pilot_scale) :
                    local_radius;
  if (!(pilot_radius > 0.0f) || !isfinite_safe(pilot_radius)) {
    return;
  }
  const float inverse_radius = 1.0f / pilot_radius;
  const float3 own_power = float3(photon->power);
  const float power_scale = max(max(own_power.x, own_power.y), own_power.z);
  if (!(power_scale > 0.0f) || !isfinite_safe(power_scale)) {
    return;
  }
  packed_normal packed;
  packed.value = photon->normal;
  const float3 normal = volume ? zero_float3() : packed.decode();
  PhotonEllipseStatistics ellipse_statistics = {};
  const PhotonEllipseFrame ellipse_frame =
      cache.pilot_footprints && !volume ?
          photon_ellipse_frame(cache.pilot_footprints[cache.stored_indices[stored_index]],
                               normal) :
          PhotonEllipseFrame{};
  const int3 center = photon_caustics_cell(photon->P, cell_size);
  PhotonRadiusStatistics statistics = {};
  PhotonRadiusStatistics local_statistics = {};
  const float statistical_scale = independent ? pilot_radius / base_radius : maximum;
  /* Float threshold avoids overflowing an integer for tiny differential kernels. */
  const float dense_count = ceilf(
      target * (volume ? powf(statistical_scale, 3.0f) : sqr(statistical_scale)));
  /* Once coverage is sufficient, no expansion is needed. The early exit does
   * not select a bandwidth from a traversal-dependent subset: it always returns 1. */
  for (int z = -1; z <= 1 && statistics.count < dense_count; ++z) {
    for (int y = -1; y <= 1 && statistics.count < dense_count; ++y) {
      for (int x = -1; x <= 1 && statistics.count < dense_count; ++x) {
        const int3 cell = make_int3(center.x + x, center.y + y, center.z + z);
        uint entry = cache.hash[photon_caustics_hash(cache, cell)];
        while (entry && entry <= cache.capacity && statistics.count < dense_count) {
          const ccl_global KernelPhotonCaustic *other = &cache.photons[entry - 1u];
          entry = other->next;
          const int3 other_cell = photon_caustics_cell(other->P, cell_size);
          if (other_cell.x != cell.x || other_cell.y != cell.y || other_cell.z != cell.z ||
              other->receiver_object != photon->receiver_object || other->light != photon->light ||
              (volume && other->volume_signature != photon->volume_signature))
          {
            continue;
          }
          const float3 delta = float3(other->P) - float3(photon->P);
          if (len_squared(delta) >= pilot_radius * pilot_radius) {
            continue;
          }
          if (!volume) {
            packed_normal other_normal;
            other_normal.value = other->normal;
            if (dot(normal, other_normal.decode()) < .9f ||
                fabsf(dot(delta, normal)) > .1f * cache.radius)
            {
              continue;
            }
          }
          if (!photon_reconstruction_history_compatible(
                  kg, cache, cache.stored_indices[stored_index], uint(other - cache.photons)))
          {
            continue;
          }
          const float3 normalized_power = float3(other->power) / power_scale;
          if (!isfinite_safe(normalized_power) || is_zero(normalized_power)) {
            continue;
          }
          photon_radius_add(&statistics, delta * inverse_radius, normalized_power);
          const bool local = !independent || len_squared(delta) < sqr(local_radius);
          if (independent && local && strength > 0.0f) {
            photon_radius_add(&local_statistics, delta / local_radius, normalized_power);
          }
          if (local && ellipse_frame.peak > 0.0f) {
            packed_normal neighbor_normal;
            neighbor_normal.value = other->normal;
            photon_ellipse_add(&ellipse_statistics,
                               ellipse_frame,
                               cache.pilot_footprints[uint(other - cache.photons)],
                               neighbor_normal.decode(),
                               -delta,
                               average(normalized_power));
          }
        }
      }
    }
  }
  if (statistics.count < dense_count) {
    const float scale = photon_radius_scale(&statistics,
                                            target,
                                            maximum,
                                            strength,
                                            volume,
                                            &ellipse_statistics,
                                            independent ? statistical_scale : 0.0f,
                                            independent ? &local_statistics : nullptr);
    photon->footprint *= 1.0f / sqr(scale);
  }
}

#endif

CCL_NAMESPACE_END
