/* SPDX-FileCopyrightText: 2026 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#pragma once

#include "kernel/integrator/nested_dielectrics.h"

#include "kernel/bvh/bvh.h"
#include "kernel/film/lpe.h"
#include "kernel/geom/shader_data.h"
#include "kernel/geom/triangle.h"
#include "kernel/integrator/intersect_volume_stack.h"
#include "kernel/integrator/path_state.h"
#include "kernel/integrator/photon_differential.h"
#include "kernel/integrator/photon_guiding.h"
#include "kernel/integrator/photon_reconstruction.h"
#include "kernel/integrator/photon_time.h"
#include "kernel/integrator/shade_volume.h"
#include "kernel/integrator/surface_shader.h"
#include "kernel/light/sample.h"
#include "kernel/sample/photon_gather.h"
#include "kernel/sample/photon_noise.h"
#include "kernel/sample/photon_rgb_spectrum.h"
#include "kernel/shader_photon_path.h"
#include "kernel/util/caustics.h"
#include "util/math_intersect.h"
#include "util/types_normal.h"

CCL_NAMESPACE_BEGIN

#ifdef __PHOTON_CAUSTICS__
ccl_device_inline KernelPhotonCausticsCache photon_caustics_cache(KernelGlobals kg)
{
#  ifdef __KERNEL_GPU__
  KernelPhotonCausticsCache cache;
  cache.photons = kernel_integrator_state.photon_caustics;
  cache.hash = kernel_integrator_state.photon_caustics_hash;
  cache.query_hash = kernel_integrator_state.photon_caustics_query_hash;
  cache.query_next = kernel_integrator_state.photon_caustics_query_next;
  cache.count = kernel_integrator_state.photon_caustics_count;
  cache.stored_indices = kernel_integrator_state.photon_caustics_stored_indices;
  cache.pilot_footprints = kernel_integrator_state.photon_caustics_pilot_footprints;
  cache.cell_statistics = kernel_integrator_state.photon_cell_statistics;
  cache.reconstruction_metadata = kernel_integrator_state.photon_reconstruction_metadata;
  cache.beam_metadata = kernel_integrator_state.photon_beam_metadata;
  cache.cache_miss = (ccl_global uint *)&kernel_integrator_state.queue_counter->cache_miss;
  cache.capacity = kernel_integrator_state.photon_caustics_capacity;
  cache.hash_size = kernel_integrator_state.photon_caustics_hash_size;
  cache.iteration = kernel_integrator_state.photon_caustics_iteration;
  cache.wavelength_random = kernel_integrator_state.photon_caustics_wavelength_random;
  cache.wavelength_groups = kernel_integrator_state.photon_caustics_wavelength_groups;
  cache.radius = kernel_integrator_state.photon_caustics_radius;
  cache.events = kernel_integrator_state.photon_caustics_events;
  cache.path_indices = kernel_integrator_state.photon_caustics_path_indices;
  cache.guide = kernel_integrator_state.photon_guide;
  cache.guide_statistics = kernel_integrator_state.photon_guide_statistics;
  cache.guide_hits = kernel_integrator_state.photon_guide_hits;
  cache.guide_fraction = kernel_integrator_state.photon_guide_fraction;
  cache.area_guide_cdf = kernel_integrator_state.photon_area_guide_cdf;
  cache.area_guide_statistics = kernel_integrator_state.photon_area_guide_statistics;
  cache.area_guide_round_statistics = kernel_integrator_state.photon_area_guide_round_statistics;
  cache.event_stride = kernel_integrator_state.photon_caustics_event_stride;
  return cache;
#  else
  return kg->photon_caustics;
#  endif
}

ccl_device_inline uint photon_caustics_atomic_exchange(ccl_global uint *head, const uint value)
{
#  ifdef __KERNEL_GPU__
  return atomicExch(head, value);
#  else
  uint old = atomic_load_uint32(head);
  for (;;) {
    const uint previous = atomic_cas_uint32(head, old, value);
    if (previous == old) {
      return old;
    }
    old = previous;
  }
#  endif
}

ccl_device_inline uint photon_query_bucket(const KernelPhotonCausticsCache cache,
                                           const int3 cell,
                                           const int light,
                                           const bool volume)
{
  /* Hash collisions remain legal: gather checks the full cell, source and domain.
   * No aggregate radiance/mean direction is stored, so anisotropic phases and LPE survive. */
  return hash_uint3(hash_uint3(uint(cell.x), uint(cell.y), uint(cell.z)),
                    uint(light), uint(volume)) & (cache.hash_size - 1u);
}

ccl_device_inline uint photon_query_head(const KernelPhotonCausticsCache cache,
                                         const int3 cell,
                                         const int light,
                                         const bool volume)
{
  return cache.query_hash ? cache.query_hash[photon_query_bucket(cache, cell, light, volume)] :
                            cache.hash[photon_caustics_hash(cache, cell)];
}
#endif

/* Finite-radius photon density estimation of L S+ R paths. Each map remains read-only
 * during its camera-sample interval, then an independent emission round replaces it.
 * Deposits contain flux divided by EMITTED paths, not successful deposits; the camera
 * film averages the successive map estimates. Re-emission reduces photon sampling noise,
 * but does not eliminate finite-radius reconstruction bias.
 * Unsupported transparent/mixed-closure caster chains stay in ordinary path tracing.
 * This is a biased reconstruction, not an unbiased MIS or manifold technique. */
ccl_device_inline bool photon_caustics_active(KernelGlobals kg)
{
#ifdef __PHOTON_CAUSTICS__
  if (!kernel_data.integrator.use_photon_caustics) {
    return false;
  }
  /* Path-family ownership must remain active while the native estimator runs alone,
   * even before a map is built or after the photon budget is exhausted. */
  if ((kernel_data.film.photon_segment_stride || kernel_data.film.photon_compact)) {
    return true;
  }
  const KernelPhotonCausticsCache cache = photon_caustics_cache(kg);
  if (kernel_data.film.lpe_state_count &&
      (!cache.events || !cache.path_indices ||
       cache.event_stride < uint(kernel_data.integrator.max_bounce)))
  {
    return false;
  }
  return cache.photons && cache.hash && cache.count && cache.capacity != 0 &&
         cache.hash_size != 0 && cache.radius > 0.0f;
#else
  return false;
#endif
}

#ifdef __PHOTON_CAUSTICS__

/* Dither the kernel before evaluating the receiving BSDF or writing any pass.
 * Shared per-photon decisions keep Combined, Caustics, light groups and LPE equal. */
ccl_device_inline float photon_caustics_noisy_weight(KernelGlobals kg,
                                                     ConstIntegratorState state,
                                                     const KernelPhotonCausticsCache cache,
                                                     const uint photon_index,
                                                     const float weight,
                                                     const float peak,
                                                     const bool ellipse)
{
  const float strength = ellipse ? kernel_data.integrator.photon_caustics_ellipse_noise_strength :
                                   kernel_data.integrator.photon_caustics_noise_strength;
  if (!(strength > 1e-6f)) {
    return weight;
  }
  const float random = photon_noise_sample(INTEGRATOR_STATE(state, path, photon_noise_pixel),
                                           INTEGRATOR_STATE(state, path, sample),
                                           photon_index,
                                           cache.iteration,
                                           INTEGRATOR_STATE(state, path, bounce));
  return peak * photon_noise_quantize(weight / peak, strength, random);
}

enum PhotonCausticsPath : uint {
  PHOTON_CAUSTICS_RECEIVER = 1u,
  PHOTON_CAUSTICS_SPECULAR_CHAIN = 2u,
};

ccl_device_inline bool photon_caustics_light_supported(KernelGlobals kg, const int light)
{
  const ccl_global KernelLight *klight = &kernel_data_fetch(lights, light);
  if (!klight->use_caustics || !klight->visible_photon || !(klight->photon_strength > 0.0f)) {
    return false;
  }
  if (klight->type == LIGHT_BACKGROUND) {
    return kernel_data.background.photon_sun.w > 0.0f;
  }
  if (klight->type != LIGHT_SUN && klight->type != LIGHT_POINT && klight->type != LIGHT_SPOT &&
      klight->type != LIGHT_AREA)
  {
    return false;
  }
  return !(kernel_data_fetch(shaders, klight->shader_id & SHADER_MASK).flags & SD_HAS_RAYTRACE);
}

ccl_device_inline bool photon_caustics_light_linked(KernelGlobals kg,
                                                    const int receiver,
                                                    const int light)
{
  return light_link_object_match(kg,
      receiver == OBJECT_NONE ? kernel_data.background.object_index : receiver,
      kernel_data_fetch(lights, light).object_id);
}

ccl_device_inline bool photon_caustics_caster(const ccl_private ShaderData *sd)
{
  if (!(sd->object_flag & SD_OBJECT_CAUSTICS_CASTER) || !(sd->type & PRIMITIVE_TRIANGLE) ||
      ((sd->runtime_flag & (SR_BSSRDF | SR_HOLDOUT)) || (sd->shader_flag & SD_HAS_RAYTRACE)))
  {
    return false;
  }
  bool found = false;
  for (int i = 0; i < sd->num_closure; ++i) {
    const ccl_private ShaderClosure *sc = &sd->closure[i];
    if (sc->sample_weight == 0.0f) {
      continue;
    }
    if (!CLOSURE_IS_BSDF_MICROFACET(sc->type) && !CLOSURE_IS_BSDF_TRANSPARENT(sc->type) &&
        sc->type != CLOSURE_BSDF_RAY_PORTAL_ID)
    {
      return false;
    }
    found = true;
  }
  return found;
}

ccl_device_inline bool photon_caustics_receiver(const ccl_private ShaderData *sd)
{
  /* Every supported surface can receive transported photon flux. Receive Shadow
   * Caustics remains a native MNEE selection, not a requirement for photon transport. */
  if (!(sd->type & PRIMITIVE_TRIANGLE) || !(sd->runtime_flag & SR_BSDF_HAS_EVAL) ||
      ((sd->runtime_flag &
        (SR_BSSRDF | SR_HOLDOUT | SR_RAY_PORTAL | SR_BSDF_HAS_TRANSMISSION)) ||
       (sd->shader_flag & SD_HAS_RAYTRACE)))
  {
    return false;
  }
  for (int i = 0; i < sd->num_closure; ++i) {
    const ccl_private ShaderClosure *sc = &sd->closure[i];
    if (sc->sample_weight > 0.0f &&
        (CLOSURE_IS_BSDF_TRANSPARENT(sc->type) || CLOSURE_IS_BSSRDF(sc->type)))
    {
      return false;
    }
  }
  return true;
}

ccl_device_inline bool photon_caustics_label_supported(KernelGlobals kg,
                                                       const int shader,
                                                       const int label)
{
  return (label & (LABEL_SINGULAR | LABEL_GLOSSY)) &&
         material_caustics_allowed(kg, shader, label) &&
         !(label & (LABEL_TRANSPARENT | LABEL_RAY_PORTAL)) &&
         (((label & LABEL_REFLECT) && kernel_data.integrator.caustics_reflective) ||
          ((label & LABEL_TRANSMIT) && kernel_data.integrator.caustics_refractive));
}

/* Classify actual camera scattering events, not merely intersections with marked objects. */
ccl_device_inline void photon_caustics_path_bounce(KernelGlobals kg,
                                                   IntegratorState state,
                                                   const ccl_private ShaderData *sd,
                                                   const int label)
{
  if (!photon_caustics_active(kg)) {
    return;
  }
  const uint old = INTEGRATOR_STATE(state, path, photon_caustics);
  uint next = 0;
  if (label & (LABEL_RAY_PORTAL | LABEL_TRANSPARENT)) {
    /* Relocation and smooth thin-wall passthrough do not focus light, but retain
     * an existing caustic suffix when both object and material allow transmission. */
    INTEGRATOR_STATE_WRITE(state, path, photon_caustics) =
        photon_caustics_caster(sd) && kernel_data.integrator.caustics_refractive &&
                material_caustics_allowed(kg, sd->shader, LABEL_TRANSMIT) ?
            old :
            0;
    return;
  }
  if ((label & LABEL_REFLECT) && !(label & (LABEL_SINGULAR | LABEL_TRANSPARENT)) &&
      photon_caustics_receiver(sd))
  {
    next = PHOTON_CAUSTICS_RECEIVER;
  }
  if ((old & PHOTON_CAUSTICS_RECEIVER) && photon_caustics_caster(sd) &&
      photon_caustics_label_supported(kg, sd->shader, label))
  {
    next |= PHOTON_CAUSTICS_RECEIVER | PHOTON_CAUSTICS_SPECULAR_CHAIN;
  }
  INTEGRATOR_STATE_WRITE(state, path, photon_caustics) = next;
}

ccl_device_inline bool photon_caustics_direction_supported(KernelGlobals kg,
                                                           const int light,
                                                           const float3 direction)
{
  if (kernel_data_fetch(lights, light).type != LIGHT_BACKGROUND) {
    return true;
  }
  const float4 sun = kernel_data.background.photon_sun;
  return sun.w > 0.0f && precise_angle(make_float3(sun), direction) <= sun.w;
}

ccl_device_inline bool photon_caustics_cull_light(KernelGlobals kg,
                                                  ConstIntegratorState state,
                                                  const int light)
{
  return photon_caustics_active(kg) &&
         (INTEGRATOR_STATE(state, path, photon_caustics) & PHOTON_CAUSTICS_SPECULAR_CHAIN) &&
         INTEGRATOR_STATE(state, path, glossy_bounce) < kernel_data.integrator.max_glossy_bounce &&
         INTEGRATOR_STATE(state, path, transmission_bounce) <
             kernel_data.integrator.max_transmission_bounce &&
         int(INTEGRATOR_STATE(state, path, bounce)) - 1 <=
             kernel_data_fetch(lights, light).max_bounces &&
         photon_caustics_light_supported(kg, light) &&
         photon_caustics_direction_supported(kg, light, INTEGRATOR_STATE(state, ray, D));
}

/* NEE must relinquish the same rough light connection as a BSDF light hit.
 * Otherwise enabling glossy photon paths would add both estimators in full. */
ccl_device_inline bool photon_caustics_cull_direct_light(KernelGlobals kg,
                                                         ConstIntegratorState state,
                                                         const ccl_private ShaderData *sd,
                                                         const int light,
                                                         const bool transmission,
                                                         const float3 direction)
{
  const int label = LABEL_GLOSSY | (transmission ? LABEL_TRANSMIT : LABEL_REFLECT);
  return photon_caustics_active(kg) &&
         (INTEGRATOR_STATE(state, path, photon_caustics) & PHOTON_CAUSTICS_RECEIVER) &&
         photon_caustics_caster(sd) && photon_caustics_label_supported(kg, sd->shader, label) &&
         photon_caustics_light_supported(kg, light) &&
         photon_caustics_direction_supported(kg, light, direction) &&
         INTEGRATOR_STATE(state, path, glossy_bounce) + uint(!transmission) <
             uint(kernel_data.integrator.max_glossy_bounce) &&
         INTEGRATOR_STATE(state, path, transmission_bounce) + uint(transmission) <
             uint(kernel_data.integrator.max_transmission_bounce) &&
         INTEGRATOR_STATE(state, path, bounce) <= kernel_data_fetch(lights, light).max_bounces;
}

ccl_device_inline float photon_caustics_surface_radius(KernelGlobals kg,
                                                       const KernelPhotonCausticsCache cache)
{
  return cache.radius * ((kernel_data.integrator.photon_caustics_has_rough &&
                          kernel_data.integrator.photon_caustics_roughness_spread > 0.0f) ?
                             4.0f :
                             1.0f);
}

ccl_device_inline float photon_caustics_effective_scatter(KernelGlobals kg)
{
  /* Independent controls for fixed and differential footprints. The latter also
   * controls isotropic fallbacks, just like differential photon noise. */
  return kernel_data.integrator.photon_caustics_differentials ?
             kernel_data.integrator.photon_caustics_differential_scatter :
             kernel_data.integrator.photon_caustics_scatter;
}

ccl_device_inline float photon_caustics_cell_size(KernelGlobals kg,
                                                  const KernelPhotonCausticsCache cache)
{
  const float reconstruction_scale = kernel_data.integrator.use_photon_adaptive_radius ?
                                         kernel_data.integrator.photon_adaptive_max_scale :
                                         1.0f;
  const float statistical_radius =
      kernel_data.integrator.use_photon_adaptive_radius &&
              (kernel_data.integrator.photon_reconstruction_method !=
                   PHOTON_RECONSTRUCTION_LEGACY ||
               kernel_data.integrator.photon_reconstruction_pilot_scale > 1.0f) ?
          cache.radius * kernel_data.integrator.photon_reconstruction_pilot_scale :
          0.0f;
  return max(statistical_radius,
             photon_caustics_surface_radius(kg, cache) * reconstruction_scale *
                 max(1.0f, kernel_data.integrator.photon_caustics_max_radius_scale) *
                 (1.0f + photon_caustics_effective_scatter(kg)));
}

/* Differentiate the normalized interpolated mesh normal in world space. */
ccl_device_inline bool photon_caustics_use_geometry_normal(const ccl_private ShaderData *sd)
{
  return (sd->object_flag & SD_OBJECT_PHOTON_GEOMETRY_NORMAL) ||
         (sd->shader_flag & SD_PHOTON_GEOMETRY_NORMAL);
}

ccl_device_inline void photon_caustics_mesh_normal_differentials(
    KernelGlobals kg,
    const ccl_private ShaderData *sd,
    const float3 N,
    ccl_private PhotonDifferential *beam,
    float3 dN[4])
{
  for (int i = 0; i < 4; ++i)
    dN[i] = zero_float3();
  if (!beam->valid) {
    return;
  }
  if (!(sd->shader & SHADER_SMOOTH_NORMAL)) {
    beam->valid = dot(sd->Ng, N) > 0.99999f;
    return;
  }
  float3 vertices[3], normals[3];
  if (sd->type & PRIMITIVE_MOTION) {
    motion_triangle_vertices_and_normals(kg, sd, vertices, normals);
  }
  else {
    triangle_vertices_and_normals(kg, sd, vertices, normals);
  }
  if (!(sd->object_flag & SD_OBJECT_TRANSFORM_APPLIED)) {
    const Transform inverse = object_get_inverse_transform(kg, sd);
    for (int i = 0; i < 3; ++i) {
      object_position_transform(kg, sd, &vertices[i]);
      /* Normalize after interpolation, not separately per transformed vertex. */
      normals[i] = transform_direction_transposed(&inverse, normals[i]);
    }
  }
  float normal_length;
  const float3 interpolated = safe_normalize_len(
      triangle_interpolate(sd->u, sd->v, normals[0], normals[1], normals[2]), &normal_length);
  const float sign = (dot(interpolated, N) < 0.0f) ? -1.0f : 1.0f;
  if (!(normal_length > 1e-8f) || sign * dot(interpolated, N) < 0.99999f) {
    beam->valid = false;
    return;
  }
  const float3 dpdu = vertices[1] - vertices[0], dpdv = vertices[2] - vertices[0];
  const float uu = dot(dpdu, dpdu), uv = dot(dpdu, dpdv), vv = dot(dpdv, dpdv);
  const float determinant = uu * vv - uv * uv;
  if (!(determinant > 1e-12f * uu * vv)) {
    beam->valid = false;
    return;
  }
  const float3 nu = normals[1] - normals[0], nv = normals[2] - normals[0];
  const float3 dndu = sign / normal_length * (nu - interpolated * dot(interpolated, nu));
  const float3 dndv = sign / normal_length * (nv - interpolated * dot(interpolated, nv));
  for (int i = 0; i < 4; ++i) {
    const float u = dot(beam->dP[i], dpdu), v = dot(beam->dP[i], dpdv);
    dN[i] = ((vv * u - uv * v) * dndu + (uu * v - uv * u) * dndv) / determinant;
  }
}

/* Evaluate spatial derivatives of the selected closure normal through the native
 * shader, including Normal Map, Bump and displacement-to-bump. Keep time, incoming
 * direction and texture filter width fixed. Closure discontinuities, degenerate
 * triangles and numerical failures use the bounded isotropic footprint.
 * One scratch ShaderData is reused; the default geometry-normal path never calls
 * this expensive branch. Texture residency failures retry the complete round. */
ccl_device_noinline void photon_caustics_shader_normal_differentials(
    KernelGlobals kg,
    IntegratorState state,
    ccl_private ShaderData *sd,
    const int closure_index,
    const ccl_private Ray *ray,
    const ccl_private Intersection *isect,
    ccl_private PhotonDifferential *beam,
    float3 dN[4])
{
  for (int i = 0; i < 4; ++i) {
    dN[i] = zero_float3();
  }
  if (!beam->valid) {
    return;
  }
  float3 vertices[3];
  if (sd->type & PRIMITIVE_MOTION) {
    motion_triangle_vertices(kg, sd->object, sd->prim, sd->time, vertices);
  }
  else {
    triangle_vertices(kg, sd->object, sd->prim, vertices);
  }
  if (!(sd->object_flag & SD_OBJECT_TRANSFORM_APPLIED)) {
    for (int i = 0; i < 3; ++i) {
      object_position_transform(kg, sd, &vertices[i]);
    }
  }
  const float3 basis[2] = {vertices[1] - vertices[0], vertices[2] - vertices[0]};
  const float uu = dot(basis[0], basis[0]), uv = dot(basis[0], basis[1]);
  const float vv = dot(basis[1], basis[1]), determinant = uu * vv - uv * uv;
  if (!(determinant > 1e-12f * uu * vv)) {
    beam->valid = false;
    return;
  }
  const float3 N = sd->closure[closure_index].N;
  float3 gradient[2];
  ShaderData neighbor;
  for (int axis = 0; axis < 2; ++axis) {
    const float center = axis == 0 ? sd->u : sd->v;
    const float limit = max(0.0f, 1.0f - (axis == 0 ? sd->v : sd->u));
    /* Resolve derivatives within the native shading footprint without spanning
     * a triangle boundary or changing the footprint with reconstruction radius. */
    const float h = clamp(0.25f * sd->dP / sqrtf(axis == 0 ? uu : vv), 1e-5f, 1e-3f);
    const float lo = max(0.0f, center - h), hi = min(limit, center + h);
    if (!(hi - lo > 1e-7f)) {
      beam->valid = false;
      return;
    }
    float3 normals[2];
    for (int side = 0; side < 2; ++side) {
      const float coordinate = side == 0 ? lo : hi;
      Intersection shifted = *isect;
      if (axis == 0) {
        shifted.u = coordinate;
      }
      else {
        shifted.v = coordinate;
      }
      shader_setup_from_ray(kg, &neighbor, ray, &shifted);
#  ifdef __SPECTRAL__
      neighbor.rand_wavelength = sd->rand_wavelength;
      neighbor.medium_ior = sd->medium_ior;
      neighbor.medium_inv_abbe = sd->medium_inv_abbe;
      neighbor.medium_transmission_tint = sd->medium_transmission_tint;
#  endif
      neighbor.photon_path = sd->photon_path;
      neighbor.runtime_flag |= sd->runtime_flag & SR_PHOTON_ZERO_ROUGHNESS;
      neighbor.P = sd->P + (coordinate - center) * basis[axis];
      surface_shader_eval<KERNEL_FEATURE_NODE_MASK_SURFACE & ~KERNEL_FEATURE_NODE_AOV>(
          kg, state, &neighbor, nullptr, PATH_RAY_VISIBILITY_NONE, 0);
      if (neighbor.runtime_flag & SR_CACHE_MISS) {
        sd->runtime_flag |= SR_CACHE_MISS;
        beam->valid = false;
        return;
      }
      if (neighbor.num_closure != sd->num_closure ||
          neighbor.closure[closure_index].type != sd->closure[closure_index].type)
      {
        beam->valid = false;
        return;
      }
      normals[side] = neighbor.closure[closure_index].N;
      if (!isfinite_safe(normals[side]) || dot(normals[side], N) < 0.5f) {
        beam->valid = false;
        return;
      }
    }
    gradient[axis] = (normals[1] - normals[0]) / (hi - lo);
    gradient[axis] -= N * dot(N, gradient[axis]);
  }
  for (int i = 0; i < 4; ++i) {
    const float u = dot(beam->dP[i], basis[0]), v = dot(beam->dP[i], basis[1]);
    dN[i] = ((vv * u - uv * v) * gradient[0] + (uu * v - uv * u) * gradient[1]) / determinant;
    beam->valid &= isfinite_safe(dN[i]);
  }
}

/* Transport the local beam through the shader-defined portal map (P, wi) -> (P', D').
 * Probe the selected closure with the same wavelength and photon history. The
 * samples stay on the source triangle; the outgoing footprint is projected at
 * the next actual intersection, not onto an invented destination surface. */
ccl_device_noinline void photon_caustics_portal_differentials(
    KernelGlobals kg,
    IntegratorState state,
    ccl_private ShaderData *sd,
    const int closure_index,
    const ccl_private Ray *ray,
    const ccl_private Intersection *isect,
    ccl_private PhotonDifferential *beam)
{
  if (!beam->valid)
    return;
  const ccl_private RayPortalClosure *pc =
      (const ccl_private RayPortalClosure *)&sd->closure[closure_index];
  float3 vertices[3];
  if (sd->type & PRIMITIVE_MOTION)
    motion_triangle_vertices(kg, sd->object, sd->prim, sd->time, vertices);
  else
    triangle_vertices(kg, sd->object, sd->prim, vertices);
  if (!(sd->object_flag & SD_OBJECT_TRANSFORM_APPLIED))
    for (int i = 0; i < 3; ++i)
      object_position_transform(kg, sd, &vertices[i]);
  const float3 U = vertices[1] - vertices[0], V = vertices[2] - vertices[0];
  const float uu = dot(U, U), uv = dot(U, V), vv = dot(V, V);
  const float det = uu * vv - uv * uv;
  if (!(det > 1e-12f * uu * vv))
    return;
  /* Differentiate in a well-scaled surface/direction basis, then apply the
   * Jacobian to the beam. Differencing along a collapsed caustic footprint
   * loses precision after adding a portal translation to its tiny offsets. */
  float3 T, B;
  make_orthonormals(ray->D, &T, &B);
  float coefficients[4][4];
  for (int axis = 0; axis < 4; ++axis) {
    const float pu = dot(beam->dP[axis], U), pv = dot(beam->dP[axis], V);
    coefficients[axis][0] = (vv * pu - uv * pv) / det;
    coefficients[axis][1] = (uu * pv - uv * pu) / det;
    coefficients[axis][2] = dot(beam->dD[axis], T);
    coefficients[axis][3] = dot(beam->dD[axis], B);
  }
  const float3 position_basis[4] = {U, V, zero_float3(), zero_float3()};
  const float3 direction_basis[4] = {zero_float3(), zero_float3(), T, B};
  float3 position_gradient[4], direction_gradient[4];
  ShaderData neighbor;
  for (int column = 0; column < 4; ++column) {
    position_gradient[column] = position_basis[column];
    direction_gradient[column] = direction_basis[column];
    bool needed = false;
    for (int axis = 0; axis < 4; ++axis)
      needed |= coefficients[axis][column] != 0.0f;
    if (!needed)
      continue;
    float steps[2] = {1e-3f, 1e-3f};
    if (column < 2) {
      const float coordinate = column == 0 ? sd->u : sd->v;
      steps[0] = min(steps[0], max(coordinate, 0.0f));
      steps[1] = min(steps[1], max(1.0f - sd->u - sd->v, 0.0f));
    }
    if (!(steps[0] + steps[1] > 1e-8f))
      continue;
    float3 positions[2] = {pc->P, pc->P}, directions[2] = {pc->D, pc->D};
    bool valid = true;
    for (int side = 0; side < 2; ++side) {
      if (steps[side] < 1e-8f)
        continue;
      const float t = side ? steps[side] : -steps[side];
      Intersection shifted = *isect;
      if (column == 0)
        shifted.u = clamp(sd->u + t, 0.0f, 1.0f - sd->v);
      else if (column == 1)
        shifted.v = clamp(sd->v + t, 0.0f, 1.0f - sd->u);
      Ray input = *ray;
      input.D = safe_normalize(ray->D + t * direction_basis[column]);
      shader_setup_from_ray(kg, &neighbor, &input, &shifted);
      neighbor.P = sd->P + t * position_basis[column];
      neighbor.photon_path = sd->photon_path;
      neighbor.runtime_flag |= sd->runtime_flag & SR_PHOTON_ZERO_ROUGHNESS;
      neighbor.medium_ior = sd->medium_ior;
      neighbor.medium_inv_abbe = sd->medium_inv_abbe;
      neighbor.medium_transmission_tint = sd->medium_transmission_tint;
#  ifdef __SPECTRAL__
      neighbor.rand_wavelength = sd->rand_wavelength;
#  endif
      surface_shader_eval<KERNEL_FEATURE_NODE_MASK_SURFACE & ~KERNEL_FEATURE_NODE_AOV>(
          kg, state, &neighbor, nullptr, PATH_RAY_VISIBILITY_NONE, 0);
      if (neighbor.runtime_flag & SR_CACHE_MISS) {
        sd->runtime_flag |= SR_CACHE_MISS;
        return;
      }
      if (neighbor.num_closure != sd->num_closure ||
          neighbor.closure[closure_index].type != CLOSURE_BSDF_RAY_PORTAL_ID)
      {
        valid = false;
        break;
      }
      const ccl_private RayPortalClosure *sample =
          (const ccl_private RayPortalClosure *)&neighbor.closure[closure_index];
      positions[side] = sample->P;
      directions[side] = sample->D;
      valid &= isfinite_safe(sample->P) && isfinite_safe(sample->D) &&
               dot(sample->D, pc->D) > 0.5f;
    }
    if (valid) {
      const float inv_step = 1.0f / (steps[0] + steps[1]);
      const float3 output_P = (positions[1] - positions[0]) * inv_step;
      const float3 output_D = (directions[1] - directions[0]) * inv_step;
      if (isfinite_safe(output_P) && isfinite_safe(output_D)) {
        position_gradient[column] = output_P;
        direction_gradient[column] = output_D;
      }
    }
    /* A branch discontinuity has no local derivative: preserve that input basis
     * column instead of replacing the entire photon with an isotropic kernel. */
  }
  for (int axis = 0; axis < 4; ++axis) {
    float3 output_P = zero_float3(), output_D = zero_float3();
    for (int column = 0; column < 4; ++column) {
      output_P += coefficients[axis][column] * position_gradient[column];
      output_D += coefficients[axis][column] * direction_gradient[column];
    }
    beam->dP[axis] = output_P;
    beam->dD[axis] = output_D - pc->D * dot(pc->D, output_D);
  }
}

ccl_device_inline float3 photon_caustics_geometric_normal(const ccl_private ShaderData *sd)
{
  return (sd->runtime_flag & SR_BACKFACING) ? -sd->Ng : sd->Ng;
}

/* Use the same watertight triangle/self-intersection test as the camera integrator. */
ccl_device_inline float3 photon_caustics_ray_origin(KernelGlobals kg,
                                                    const ccl_private ShaderData *sd,
                                                    const float3 D)
{
  float3 vertices[3];
  if (sd->type & PRIMITIVE_MOTION) {
    motion_triangle_vertices(kg, sd->object, sd->prim, sd->time, vertices);
  }
  else {
    triangle_vertices(kg, sd->object, sd->prim, vertices);
  }
  float3 local_P = sd->P;
  float3 local_D = D;
  if (!(sd->object_flag & SD_OBJECT_TRANSFORM_APPLIED)) {
    const Transform itfm = object_get_inverse_transform(kg, sd);
    local_P = transform_point(&itfm, local_P);
    local_D = transform_direction(&itfm, local_D);
  }
  return ray_triangle_intersect_self(local_P, local_D, vertices) ? sd->P :
                                                                   ray_offset(sd->P, sd->Ng);
}

/* Rays can enter overlapping volumes in different orders. Use the exact object/shader
 * pair for a single medium and an order-independent fingerprint for compound media.
 * The negative receiver and the unused volume normal fit in the existing 64-byte record. */
struct PhotonVolumeKey {
  int receiver;
  uint signature;
};

#  ifdef __VOLUME__
ccl_device_inline PhotonVolumeKey photon_volume_key(KernelGlobals kg, ConstIntegratorState state)
{
  const VolumeStack first = integrator_state_read_volume_stack(state, 0);
  if (kernel_data.volume_stack_size == 1 ||
      INTEGRATOR_STATE_ARRAY(state, volume_stack, 1, shader) == SHADER_NONE)
  {
    return {int(~(uint(first.object) + 1u)), uint(first.shader & SHADER_MASK)};
  }
  uint objects = 0, shaders = 0;
  for (int i = 0; i < kernel_data.volume_stack_size; ++i) {
    const VolumeStack entry = integrator_state_read_volume_stack(state, i);
    if (entry.shader == SHADER_NONE) {
      break;
    }
    const uint object = uint(entry.object);
    const uint shader = uint(entry.shader & SHADER_MASK);
    objects += hash_uint2(object, shader);
    shaders += hash_uint2(shader ^ 0xd6e8feb9u, object);
  }
  /* The signature's high bit distinguishes compound media from exact shader indices. */
  return {int(objects | 0x80000000u), shaders | 0x80000000u};
}
#  endif

ccl_device_inline void photon_caustics_store(KernelGlobals kg,
                                             ConstIntegratorState state,
                                             const KernelPhotonCausticsCache cache,
                                             const ccl_private ShaderData *sd,
                                             const Spectrum power,
                                             const int light,
                                             const uint glossy,
                                             const uint transmission,
                                             const uint path_index,
                                             const PhotonDifferential beam,
                                             const KernelPhotonReconstructionMetadata metadata,
                                             const bool volume = false,
                                             const bool spectral = false,
                                             const uint passthrough = 0)
{
  /* A light path deposits at most once. Use its stable emission slot, not the
   * race-dependent compacted write order, so noise is invariant under thread
   * scheduling, LPE allocation, sample subsets and render tiles. Only published
   * hash entries are read; unused slots need no clearing or extra storage. */
  const uint index = path_index;
  if (index >= cache.capacity) {
    return;
  }
  const uint stored_index = atomic_fetch_and_add_uint32(cache.count, 1);
  if (volume && cache.count[PHOTON_PROBE_ENABLED]) {
    atomic_fetch_and_add_uint32(&cache.count[PHOTON_PROBE_VOLUME_STORED], 1);
  }
  if (cache.stored_indices) {
    cache.stored_indices[stored_index] = index;
  }
  ccl_global KernelPhotonCaustic *photon = &cache.photons[index];
  photon->P = sd->P;
  photon->power = spectrum_to_rgb(power);
  photon->direction = packed_normal(sd->wi).value;
  const packed_normal normal(volume ? make_float3(0.0f, 0.0f, 1.0f) :
                                      photon_caustics_geometric_normal(sd));
  photon->normal = normal.value;
  photon->receiver_object = sd->object;
#  ifdef __VOLUME__
  if (volume) {
    const PhotonVolumeKey medium = photon_volume_key(kg, state);
    photon->receiver_object = medium.receiver;
    photon->volume_signature = medium.signature;
  }
#  endif
  const bool widened = beam.rough && !volume;
  PhotonDifferential footprint_beam = beam;
  footprint_beam.valid &= widened || kernel_data.integrator.photon_caustics_differentials;
  photon->footprint = photon_differential_footprint(
      footprint_beam,
      normal.decode(),
      kernel_data.integrator.photon_caustics_min_radius,
      widened ? photon_caustics_surface_radius(kg, cache) : cache.radius);
  /* Negative normalization tags a rough footprint without enlarging the cache. */
  if (widened)
    photon->footprint.w = -photon->footprint.w;
  if (cache.pilot_footprints) {
    if (kernel_data.integrator.photon_reconstruction_method ==
        PHOTON_RECONSTRUCTION_ELLIPSE_COVERAGE)
    {
      /* Overlap needs ALL original surface kernels. A negative pilot peak marks
       * a fallback/rough kernel: usable for coverage, not reliable differential
       * edge evidence. This does not change the published photon's rough tag. */
      float4 pilot = volume ? zero_float4() : photon->footprint;
      if (!(kernel_data.integrator.photon_caustics_differentials && beam.valid && !beam.rough)) {
        pilot.w = -fabsf(pilot.w);
      }
      cache.pilot_footprints[index] = pilot;
    }
    else {
      cache.pilot_footprints[index] = (!volume && beam.valid && !beam.rough &&
                                       isfinite_safe(photon->footprint.w)) ?
                                          photon->footprint :
                                          zero_float4();
    }
  }
  if (cache.reconstruction_metadata) {
    cache.reconstruction_metadata[index] = metadata;
  }
  photon->light = light;
  photon->pad = photon_bounce_pack(glossy, transmission, passthrough) |
                (spectral ? PHOTON_CAUSTICS_SPECTRAL : 0u);
  if (cache.path_indices) {
    cache.path_indices[index] = path_index;
  }
  const uint bucket = photon_caustics_hash(
      cache, photon_caustics_cell(sd->P, photon_caustics_cell_size(kg, cache)));
  /* Zero is empty; readers run only after all emission tasks/threads finish. */
  photon->next = photon_caustics_atomic_exchange(&cache.hash[bucket], index + 1u);
  if (cache.query_hash) {
    const int3 cell = photon_caustics_cell(sd->P, photon_caustics_cell_size(kg, cache));
    const uint query_bucket = photon_query_bucket(cache, cell, light, volume);
    cache.query_next[index] = photon_caustics_atomic_exchange(&cache.query_hash[query_bucket],
                                                              index + 1u);
  }
}

ccl_device void integrator_photon_caustics_cells(KernelGlobals kg, const uint stored_index)
{
  photon_cell_build(kg, photon_caustics_cache(kg), stored_index);
}

/* The schedule publishes a complete map; reconstruction owns only its kernel
 * parameters and optional pilot metadata. It does not own emission or film writes. */
ccl_device void integrator_photon_caustics_reconstruct(KernelGlobals kg, const uint stored_index)
{
  const KernelPhotonCausticsCache cache = photon_caustics_cache(kg);
  photon_reconstruct_map(kg, cache, stored_index, photon_caustics_cell_size(kg, cache));
}

/* Source-first emission uses area * solid-angle measure throughout. The shader
 * is evaluated later at this exact source point/direction/time. Invalid directions
 * are zero attempts, never rejection-resampled with an unknown normalization. */
ccl_device_inline bool photon_caustics_sample_guided_area(KernelGlobals kg,
                                                          const KernelPhotonCausticsCache cache,
                                                          const ccl_global KernelLight *light,
                                                          const int light_index,
                                                          ccl_private uint *rng,
                                                          ccl_private Ray *ray,
                                                          ccl_private Spectrum *power,
                                                          ccl_private float *beam_area,
                                                          ccl_private uint *source_cell)
{
  float uv_pdf;
  const float3 source_random = lcg_step_float3(rng);
  const float2 uv = photon_area_sample(
      cache.area_guide_cdf ? cache.area_guide_cdf + light_index * PHOTON_AREA_GUIDE_CELLS :
                             nullptr,
      make_float2(source_random),
      source_cell,
      &uv_pdf);
  const float3 ru = light->area.axis_u * light->area.len_u * .5f;
  const float3 rv = light->area.axis_v * light->area.len_v * .5f;
  ray->P = float3(light->co) + (area_light_is_ellipse(&light->area) ?
                                    ellipse_sample(ru, rv, uv) :
                                    rectangle_sample(ru, rv, uv));
  float3 normal = light->area.dir;
  Transform to_world, to_center;
  if (light_motion_transform(kg, light, ray->time, &to_world, &to_center)) {
    ray->P = transform_point(&to_world, ray->P);
    normal = transform_direction(&to_world, normal);
  }
  const float invarea = fabsf(light->area.invarea);
  /* invarea is a radiance normalization and becomes 1 when Normalize is off.
   * The sampling density always uses the actual geometric emitter area. */
  const float source_area = light->area.len_u * light->area.len_v *
                            (area_light_is_ellipse(&light->area) ? M_PI_4_F : 1.0f);
  const float area_pdf = uv_pdf / source_area;
  if (!(area_pdf > 0.0f))
    return false;
  const float spread = light->area.tan_half_spread;
  if (spread == 0.0f) {
    /* Integrated delta-direction normalization matches native zero-spread Area. */
    ray->D = normal;
    *power *= invarea / area_pdf;
    *beam_area = 1.0f / area_pdf;
    return true;
  }
  const float4 target = kernel_data.integrator.photon_caustics_target;
  const float3 center = make_float3(target);
  const float3 to_target = center - ray->P;
  const float distance_squared = len_squared(to_target);
  const float bound_omc = distance_squared > sqr(target.w) ?
                              sin_sqr_to_one_minus_cos(sqr(target.w) / distance_squared) :
                              2.0f;
  const float spread_sin_squared = spread > 1e10f ? 1.0f : sqr(spread) / (1.0f + sqr(spread));
  const float spread_omc = sin_sqr_to_one_minus_cos(spread_sin_squared);
  const bool emitter_cone = spread_omc < bound_omc;
  const float3 bound_axis = safe_normalize(to_target);
  const float2 random = make_float2(lcg_step_float3(rng));
  if (emitter_cone) {
    const float2 disk = sample_uniform_disk(random) * sqrtf(spread_sin_squared);
    float3 u, v;
    make_orthonormals(normal, &u, &v);
    ray->D = disk.x * u + disk.y * v + safe_sqrtf(1.0f - len_squared(disk)) * normal;
  }
  else {
    float unused, pdf;
    ray->D = sample_uniform_cone(bound_axis, bound_omc, random, &unused, &pdf);
  }
  const float fraction = photon_guide_light_fraction(cache, light_index);
  if (fraction > 0.0f && lcg_step_float(rng) < fraction) {
    ray->D = safe_normalize(center + target.w * photon_guide_sample(cache, light_index, rng) -
                            ray->P);
  }
  const float cosine = dot(normal, ray->D);
  if (!(cosine > 0.0f))
    return false;
  const float native_pdf =
      emitter_cone ?
          (cosine >= 1.0f - spread_omc ? cosine * M_1_PI_F / spread_sin_squared : 0.0f) :
          (dot(bound_axis, ray->D) >= 1.0f - bound_omc ? M_1_2PI_F / bound_omc : 0.0f);
  const float guided_pdf =
      fraction > 0.0f ?
          photon_guide_pdf(cache, light_index, (ray->P - center) / target.w, ray->D, false) :
          0.0f;
  const float direction_pdf = (1.0f - fraction) * native_pdf + fraction * guided_pdf;
  const float attenuation = light->area.normalize_spread > 0.0f ?
                                area_light_spread_attenuation(
                                    -ray->D, normal, spread, light->area.normalize_spread) :
                                1.0f;
  if (!(direction_pdf > 0.0f && attenuation > 0.0f))
    return false;
  *power *= invarea * M_1_PI_F * cosine * attenuation / (area_pdf * direction_pdf);
  /* Finite source position remains Monte Carlo sampled, as in the native finite
   * emitter path. These two differential columns span conditional angular area;
   * the area PDF must not be inserted into a solid-angle footprint. */
  *beam_area = 1.0f / direction_pdf;
  return true;
}

/* Sample the first scattering point before invoking native finite-light sampling.
 * This preserves Point/Spot spheres and receiving-point-oriented virtual disks, including
 * Soft Falloff. Area uses its native rectangle/ellipse, one-sided emission, Spread and Normalize.
 *
 * Choose a marked object, one of its triangles, and a uniform point on that triangle.
 * q_A = 1 / (num_casters * num_triangles * world_area). Native light sampling supplies
 * q_omega at this point. Incident flux is L * abs(Ng.wi) / (q_A * q_omega); the following
 * singular BSDF sample transports this flux and must not multiply by a second cosine.
 * The caller accepts the proposal only if a full-scene ray first hits this exact triangle.
 * An occluded or ineligible candidate is a zero sample, with no PDF renormalization. */
ccl_device_inline bool photon_caustics_sample_finite_light(KernelGlobals kg,
                                                           const ccl_global KernelLight *klight,
                                                           const int light,
                                                           ccl_private uint *rng,
                                                           ccl_private Ray *ray,
                                                           ccl_private Spectrum *power,
                                                           ccl_private int *target_object,
                                                           ccl_private int *target_prim,
                                                           ccl_private float *beam_area)
{
  const int num_casters = kernel_data.integrator.photon_caustics_caster_count;
  if (num_casters == 0) {
    return false;
  }
  const int selected = min(int(lcg_step_float(rng) * num_casters), num_casters - 1);
  /* Store the primitive offset explicitly: object_prim_offset is not allocated on
   * every backend when the light tree is disabled. */
  const int2 caster = kernel_data_fetch(photon_caustics_casters, selected);
  const int num_triangles = kernel_data_fetch(objects, caster.x).mesh_volume.num_prims;
  if (num_triangles == 0) {
    return false;
  }
  *target_object = caster.x;
  *target_prim = caster.y + min(int(lcg_step_float(rng) * num_triangles), num_triangles - 1);

  float3 vertices[3];
  triangle_world_space_vertices(kg, *target_object, *target_prim, ray->time, vertices);
  const float3 edge_u = vertices[1] - vertices[0];
  const float3 edge_v = vertices[2] - vertices[0];
  float twice_area;
  const float3 normal = safe_normalize_len(cross(edge_u, edge_v), &twice_area);
  if (!(twice_area > 0.0f)) {
    return false;
  }
  const float3 random_position = lcg_step_float3(rng);
  const float sqrt_u = sqrtf(random_position.x);
  const float u = 1.0f - sqrt_u;
  const float v = random_position.y * sqrt_u;
  const float3 P = vertices[0] + u * edge_u + v * edge_v;

  LightSample ls = {};
  const float2 random_light = make_float2(lcg_step_float3(rng));
  const bool sampled = light_sample<false>(
      kg, light, ray->time, random_light, P, normal, SR_BSDF_HAS_TRANSMISSION, 0, &ls);
  if (!sampled || !(ls.pdf > 0.0f) || !(ls.t > 0.0f)) {
    return false;
  }
  const float inverse_area_pdf = float(num_casters) * float(num_triangles) * (0.5f * twice_area);
  *power *= ls.eval_fac * fabsf(dot(normal, ls.D)) * inverse_area_pdf / ls.pdf;
  ray->P = ls.P;
  /* Native sphere sampling remaps its source point onto the sphere after computing D.
   * Recompute the connection from that final point to keep the selected triangle exact. */
  ray->D = safe_normalize(P - ray->P);
  *beam_area = inverse_area_pdf * fabsf(dot(normal, ray->D));
  /* Zero-spread Area emission moves the source point while keeping direction fixed.
   * Its projected target area defines a spatial beam, not an angular beam. */
  if (!(klight->type == LIGHT_AREA && klight->area.tan_half_spread == 0.0f)) {
    *beam_area /= len_squared(P - ray->P);
  }
  return true;
}

#  ifdef __VOLUME__
ccl_device bool photon_caustics_volume_segment(KernelGlobals kg,
                                               IntegratorState state,
                                               const ccl_private Ray *ray,
                                               const uint random_seed,
                                               ccl_private ShaderData *sd,
                                               ccl_private VolumeIntegrateResult &result,
                                               const ccl_private PhotonShaderPath *photon_path)
{
  INTEGRATOR_STATE_WRITE(state, path, rng_pixel) = random_seed;
  INTEGRATOR_STATE_WRITE(state, path, rng_offset) = PRNG_BOUNCE_NUM;
  shader_setup_from_volume(sd, ray, INTEGRATOR_STATE_ARRAY(state, volume_stack, 0, object));
  sd->photon_path = photon_path;
  RNGState rng_state;
  path_state_rng_load(state, &rng_state);
  sd->lcg_state = lcg_state_init(
      rng_state.rng_pixel, rng_state.rng_offset, rng_state.sample, 0x15b4f88d);
  if (kernel_data.integrator.volume_ray_marching) {
    /* Ray marching does not build the null-scattering octree. Reuse its native
     * distance sampler, with no camera film or direct-light work in the prepass. */
    LightSample ls = {};
    const float step_size = volume_stack_step_size<false>(kg, state);
    volume_integrate_ray_marching(kg, state, ray, sd, &rng_state, nullptr, step_size, &ls, result);
    return !(sd->runtime_flag & SR_CACHE_MISS);
  }
  VolumeIntegrateState vstate = {};
  volume_integrate_state_init(kg, state, VOLUME_SAMPLE_NONE, &rng_state, ray->tmin, vstate);
  vstate.vspg = false;
  EquiangularCoefficients coeffs = {zero_float3(), {ray->tmin, ray->tmax}};
  volume_integrate_result_init(state, ray, vstate, coeffs, result);
  if (volume_is_homogeneous<false>(kg, state)) {
    volume_integrate_homogeneous(
        kg, state, ray, sd, &rng_state, nullptr, vstate, coeffs.t_range, result);
  }
  else {
    volume_integrate_heterogeneous(kg, state, ray, sd, rng_state, nullptr, vstate, result);
  }
  return !(sd->runtime_flag & SR_CACHE_MISS);
}
#  endif

/* The physical direction uses the full native BSDF. Only its shrinking local
 * sampling cell is broadened, never a second copy of the full roughness lobe.
 * This is finite-bandwidth reconstruction with a frozen microfacet derivative,
 * not an exact integration across occluders or BSDF branch discontinuities. */
ccl_device_inline void photon_caustics_scatter_differential(KernelGlobals kg,
                                                            IntegratorState state,
                                                            const ccl_private Ray *ray,
                                                            const ccl_private Intersection *isect,
                                                            ccl_private ShaderData *sd,
                                                            const ccl_private ShaderClosure *sc,
                                                            const float3 random,
                                                            const float3 outgoing,
                                                            const int label,
                                                            const float eta,
                                                            const KernelPhotonCausticsCache cache,
                                                            ccl_private PhotonDifferential *beam,
                                                            const uint photon_index)
{
  const bool rough = !(label & LABEL_SINGULAR);
  const float spread_scale = kernel_data.integrator.photon_caustics_roughness_spread;
  if (!kernel_data.integrator.photon_caustics_differentials && !rough && !beam->rough)
    return;
  if (rough && !beam->rough &&
      (!beam->valid || !kernel_data.integrator.photon_caustics_differentials))
  {
    *beam = photon_differential_init(sd->wi, M_PI_F * sqr(cache.radius), true);
    float3 U, V;
    make_orthonormals(sd->Ng, &U, &V);
    beam->dP[0] = cache.radius * U;
    beam->dP[1] = cache.radius * V;
  }
  float3 dN[4];
  if (!photon_caustics_use_geometry_normal(sd) && (sd->shader_flag & SD_HAS_BUMP)) {
    photon_caustics_shader_normal_differentials(
        kg, state, sd, int(sc - sd->closure), ray, isect, beam, dN);
  }
  else {
    photon_caustics_mesh_normal_differentials(kg, sd, sc->N, beam, dN);
  }
  if (cache.beam_metadata) {
    ccl_global KernelPhotonBeamMetadata *metadata = &cache.beam_metadata[photon_index];
    float4 derivative = zero_float4();
    packed_normal interface_normal;
    interface_normal.value = metadata->previous_normal;
    /* Fit in precisely the same tangent frame used during reconstruction.
     * Quantizing N can otherwise change the basis near its branch boundary. */
    metadata->valid = beam->valid && photon_region_normal_derivative(
                                         interface_normal.decode(), beam->dP, dN, &derivative);
    metadata->normal_derivative = derivative;
  }
  float3 H = sc->N;
  if (rough) {
    H = safe_normalize(sd->wi + outgoing * ((label & LABEL_TRANSMIT) ? eta : 1.0f));
    if (dot(H, sc->N) < 0.0f)
      H = -H;
    for (int i = 0; i < 4; ++i)
      dN[i] -= H * dot(H, dN[i]);
  }
  photon_differential_scatter(beam, -sd->wi, H, dN, outgoing, bool(label & LABEL_TRANSMIT), eta);
  if (!rough || !(spread_scale > 0.0f) || !beam->valid)
    return;
  /* A two-dimensional angular slice of a four-dimensional emitter/BSDF domain.
   * Radius falls as N^-1/4 with photons per map. More rounds still average the
   * remaining bounded-kernel bias, like the existing photon estimator. */
  const float h = min(0.25f, spread_scale / sqrtf(sqrtf(float(cache.capacity))));
  float3 angular[2] = {zero_float3(), zero_float3()};
  for (int axis = 0; axis < 2; ++axis) {
    float3 a = random, b = random;
    if (axis == 0) {
      a.x = max(0.0f, a.x - h);
      b.x = min(0.99999994f, b.x + h);
    }
    else {
      a.y = max(0.0f, a.y - h);
      b.y = min(0.99999994f, b.y + h);
    }
    Spectrum unused;
    float3 da, db;
    float pdf_a, pdf_b, ignored_eta;
    float2 ignored_roughness;
    const int la = bsdf_sample(
        kg, sd, sc, a, &unused, &da, &pdf_a, &ignored_roughness, &ignored_eta);
    const int lb = bsdf_sample(
        kg, sd, sc, b, &unused, &db, &pdf_b, &ignored_roughness, &ignored_eta);
    const int branch = LABEL_REFLECT | LABEL_TRANSMIT;
    if (!(pdf_a > 0.0f && pdf_b > 0.0f) || (la & branch) != (label & branch) ||
        (lb & branch) != (label & branch))
      continue;
    const float width = axis == 0 ? b.x - a.x : b.y - a.y;
    if (!(width > 0.0f))
      continue;
    float3 delta = (db - da) * (h / width);
    delta -= outgoing * dot(outgoing, delta);
    const float length = len(delta);
    if (isfinite_safe(delta))
      angular[axis] = delta * min(1.0f, 0.5f / max(length, 1e-8f));
  }
  photon_differential_add_spread(beam, sd->Ng, outgoing, angular);
}

/* The independent prepass uses a private CPU state or borrows one OptiX SoA state slot.
 * No camera/shadow work is queued. */
#  ifdef __VOLUME__
/* A jump can enter a medium without crossing its mesh. Count newly enclosed
 * regions while preserving a medium that surrounds both portal endpoints. */
ccl_device_noinline bool photon_caustics_portal_media_init(KernelGlobals kg,
                                                           IntegratorState state,
                                                           const ccl_private Ray *ray,
                                                           ccl_private PhotonShaderPath *path)
{
  VolumeStack previous[MAX_VOLUME_STACK_SIZE];
  int count = 0;
  if (kernel_data.integrator.use_volumes) {
    for (; count < kernel_data.volume_stack_size; ++count) {
      previous[count] = integrator_state_read_volume_stack(state, count);
      if (previous[count].shader == SHADER_NONE)
        break;
    }
  }
  integrator_state_write_ray(state, ray);
  if (!integrator_volume_stack_init(
          kg, state, PATH_RAY_VISIBILITY_ALL & ~PATH_RAY_VISIBILITY_SHADOW))
    return false;
  if (kernel_data.integrator.use_volumes) {
    for (int i = 0; i < kernel_data.volume_stack_size; ++i) {
      const VolumeStack entry = integrator_state_read_volume_stack(state, i);
      if (entry.shader == SHADER_NONE)
        break;
      bool present = false;
      for (int j = 0; j < count; ++j)
        present |= previous[j].object == entry.object && previous[j].shader == entry.shader;
      path->volume_layers += !present;
    }
  }
  return true;
}
#  endif

#  ifdef __SPECTRAL__
ccl_device_inline float3 photon_caustics_source_basis(KernelGlobals kg, const float wavelength)
{
  const float3 peaks = make_float3(kernel_data.film.photon_spectrum_r.w,
                                   kernel_data.film.photon_spectrum_g.w,
                                   kernel_data.film.photon_spectrum_b.w);
  return photon_rgb_spectral_basis(wavelength, peaks);
}

ccl_device_inline Spectrum photon_caustics_evaluate_source(KernelGlobals kg,
                                                          const Spectrum flux,
                                                          const float3 basis)
{
  return photon_rgb_at_wavelength(flux,
                                  basis,
                                  make_float3(kernel_data.film.photon_spectrum_r),
                                  make_float3(kernel_data.film.photon_spectrum_g),
                                  make_float3(kernel_data.film.photon_spectrum_b));
}
#  endif

ccl_device void integrator_photon_caustics_emit(KernelGlobals kg,
                                                IntegratorState state,
                                                const uint photon_index,
                                                const uint iteration)
{
  if (!photon_caustics_active(kg)) {
    return;
  }
  const KernelPhotonCausticsCache cache = photon_caustics_cache(kg);
  int num_lights = 0;
  for (int i = 0; i < kernel_data.integrator.num_lights; ++i) {
    num_lights += photon_caustics_light_supported(kg, i);
  }
  if (num_lights == 0) {
    return;
  }
  uint rng = hash_uint3(photon_index, iteration, 0x6a09e667u);
  int selected = min(int(lcg_step_float(&rng) * num_lights), num_lights - 1);
  int light = 0;
  for (; light < kernel_data.integrator.num_lights; ++light) {
    if (photon_caustics_light_supported(kg, light) && selected-- == 0) {
      break;
    }
  }
  const ccl_global KernelLight *klight = &kernel_data_fetch(lights, light);
  Spectrum power = one_spectrum();
  const float4 target = kernel_data.integrator.photon_caustics_target;
  const float3 center = make_float3(target);
  const float radius = target.w;
  if (!(radius > 0.0f)) {
    return;
  }
  Ray ray = {};
  ray.tmax = FLT_MAX;
  ray.time = photon_caustics_time(kg, iteration);
  ray.self.object = ray.self.light_object = OBJECT_NONE;
  ray.self.prim = ray.self.light_prim = PRIM_NONE;
  int target_object = OBJECT_NONE, target_prim = PRIM_NONE;
  float direction_pdf, beam_area = 0.0f;
  uint source_cell = PHOTON_AREA_GUIDE_CELLS;
  bool parallel_beam = false;
  const float3 random_direction = lcg_step_float3(&rng);
  if (klight->type == LIGHT_BACKGROUND) {
    const float4 sun = kernel_data.background.photon_sun;
    float cosine;
    ray.D = -sample_uniform_cone(make_float3(sun),
                                 one_minus_cos(sun.w),
                                 make_float2(random_direction), &cosine, &direction_pdf);
    const float3 random_position = lcg_step_float3(&rng);
    ray.P = center - ray.D * kernel_data.integrator.photon_caustics_sun_distance +
            radius * disk_light_sample(ray.D, make_float2(random_position));
    beam_area = M_PI_F * radius * radius;
    power *= beam_area / direction_pdf;
    parallel_beam = true;
  }
  else if (klight->type == LIGHT_SUN) {
    LightSample ls = {};
    light_sample<false>(
        kg, light, ray.time, make_float2(random_direction), center, zero_float3(), 0, 0, &ls);
    ray.D = -ls.D;
    const float3 random_position = lcg_step_float3(&rng);
    ray.P = center - ray.D * kernel_data.integrator.photon_caustics_sun_distance +
            radius * disk_light_sample(ray.D, make_float2(random_position));
    /* Uniform target disk perpendicular to the sampled sunlight direction: q_A = 1/(pi R^2).
     * Use the native finite sun-angle radiance/PDF, including its zero-angle delta limit. */
    beam_area = M_PI_F * radius * radius;
    const float fraction = photon_guide_light_fraction(cache, light);
    if (fraction > 0.0f) {
      if (lcg_step_float(&rng) < fraction) {
        const float3 point = photon_guide_sample(cache, light, &rng);
        ray.P = center + radius * (point - ray.D * dot(point, ray.D)) -
                ray.D * kernel_data.integrator.photon_caustics_sun_distance;
      }
      const float3 position = (ray.P - center) / radius;
      const float3 tangent = position - ray.D * dot(position, ray.D);
      const float native_pdf = len_squared(tangent) <= 1.0f ? 1.0f / beam_area : 0.0f;
      const float guide_pdf = photon_guide_pdf(cache, light, tangent, ray.D, true) / sqr(radius);
      const float pdf = (1.0f - fraction) * native_pdf +
                        fraction * guide_pdf;
      if (!(pdf > 0.0f)) {
        return;
      }
      beam_area = 1.0f / pdf;
    }
    power *= ls.eval_fac * beam_area / ls.pdf;
    parallel_beam = true;
  }
  else if (klight->type == LIGHT_AREA &&
           (cache.area_guide_cdf || (cache.guide && klight->area.tan_half_spread > 0.0f)))
  {
    if (!photon_caustics_sample_guided_area(
            kg, cache, klight, light, &rng, &ray, &power, &beam_area, &source_cell))
    {
      return;
    }
    parallel_beam = klight->area.tan_half_spread == 0.0f;
  }
  else if (klight->type == LIGHT_AREA || klight->spot.radius > 0.0f) {
    if (!photon_caustics_sample_finite_light(
            kg, klight, light, &rng, &ray, &power, &target_object, &target_prim, &beam_area))
    {
      return;
    }
    parallel_beam = klight->type == LIGHT_AREA && klight->area.tan_half_spread == 0.0f;
  }
  else {
    Transform motion_world, motion_center;
    const bool light_motion = light_motion_transform(
        kg, klight, ray.time, &motion_world, &motion_center);
    ray.P = light_motion ? transform_point(&motion_world, klight->co) : float3(klight->co);
    const float3 to_target = center - ray.P;
    const float distance_squared = len_squared(to_target);
    if (distance_squared > radius * radius) {
      float unused;
      const float one_minus_cos = sin_sqr_to_one_minus_cos(radius * radius / distance_squared);
      ray.D = sample_uniform_cone(normalize(to_target),
                                  one_minus_cos,
                                  make_float2(random_direction),
                                  &unused,
                                  &direction_pdf);
    }
    else {
      ray.D = sample_uniform_sphere(make_float2(random_direction));
      direction_pdf = M_1_2PI_F * 0.5f;
    }
    const float fraction = photon_guide_light_fraction(cache, light);
    if (fraction > 0.0f) {
      if (lcg_step_float(&rng) < fraction) {
        ray.D = safe_normalize(center + radius * photon_guide_sample(cache, light, &rng) - ray.P);
      }
      float native_pdf = direction_pdf;
      if (distance_squared > radius * radius &&
          dot(ray.D, normalize(to_target)) < sqrtf(1.0f - radius * radius / distance_squared))
      {
        native_pdf = 0.0f;
      }
      direction_pdf = (1.0f - fraction) * native_pdf +
                      fraction *
                          photon_guide_pdf(cache, light, (ray.P - center) / radius, ray.D, false);
      if (!(direction_pdf > 0.0f)) {
        return;
      }
    }
    beam_area = 1.0f / direction_pdf;
    power *= klight->spot.eval_fac / direction_pdf;
    if (klight->type == LIGHT_SPOT) {
      power *= spot_light_attenuation(
          &klight->spot,
          spot_light_to_local(
              kg, klight, light_motion ? transform_direction(&motion_center, ray.D) : ray.D));
    }
  }
  /* Evaluate emission at the sampled light position and outgoing direction, using the
   * native UV/normal conventions. The geometric proposal above contains no emission
   * color or strength, so textured emission and the light strength enter exactly once. */
  path_state_init_integrator(kg, state, iteration, rng, one_spectrum());
  INTEGRATOR_STATE_WRITE(state, path, render_pixel_index) = 0;
  PhotonShaderPath photon_path = {};
  photon_path.origin = ray.P;
  photon_path.emission = true;
  Spectrum emission;
  ShaderEvalResult emission_result;
  if (klight->type == LIGHT_BACKGROUND) {
    ShaderDataTinyStorage storage;
    ccl_private ShaderData *sd = AS_SHADER_DATA(&storage);
    shader_setup_from_background(kg, sd, ray.P, -ray.D, 0.0f, ray.time);
    sd->photon_path = &photon_path;
    surface_shader_eval<KERNEL_FEATURE_NODE_MASK_SURFACE_BACKGROUND>(
        kg, state, sd, nullptr, PATH_RAY_VISIBILITY_NONE, PATH_RAY_EMISSION);
    emission = surface_shader_background(sd) * rgb_to_spectrum(make_float3(
        klight->strength[0], klight->strength[1], klight->strength[2]));
    emission_result = sd->runtime_flag & SR_CACHE_MISS ? SHADER_EVAL_CACHE_MISS : SHADER_EVAL_OK;
  }
  else {
    emission_result = light_sample_shader_eval_forward(kg,
                                                       state,
                                                       light,
                                                       ray.P,
                                                       -ray.D,
                                                       klight->type == LIGHT_SUN ? FLT_MAX : 0.0f,
                                                       ray.time,
                                                       emission,
                                                       &photon_path);
  }
  if (emission_result == SHADER_EVAL_CACHE_MISS)
  {
    /* Retry the entire round with the same seed after servicing texture residency. */
    if (cache.cache_miss) {
      atomic_fetch_and_or_uint32(cache.cache_miss, 1);
    }
    return;
  }
  photon_path.emission = false;
  power *= emission * (klight->photon_strength * float(num_lights) / float(cache.capacity));
  if (cache.area_guide_round_statistics && source_cell < PHOTON_AREA_GUIDE_CELLS) {
    const float weight = average(spectrum_to_rgb(power)) * float(cache.capacity);
    if (weight > 0.0f && isfinite_safe(weight)) {
      ccl_global float2 *s =
          &cache.area_guide_round_statistics[light * PHOTON_AREA_GUIDE_CELLS + source_cell];
      atomic_add_and_fetch_float(&s->x, weight);
      atomic_add_and_fetch_float(&s->y, 1.0f);
    }
  }
  if (is_zero(power) || !isfinite_safe(power)) {
    return;
  }

  path_state_init_integrator(kg, state, iteration, rng, power);
  INTEGRATOR_STATE_WRITE(state, path, render_pixel_index) = 0;
  INTEGRATOR_STATE_WRITE(state, path, visibility) = PATH_RAY_VISIBILITY_NONE;
  INTEGRATOR_STATE_WRITE(state, path, flag) = 0;
  PhotonDifferential beam = photon_differential_init(
      ray.D, beam_area * float(num_lights) / float(cache.capacity), parallel_beam);
  /* Native shader filtering is independent of the chosen reconstruction kernel.
   * Nonzero emitter differentials let the regular Cycles Bump nodes evaluate. */
  ray.dP = len(beam.dP[0]);
  ray.dD = len(beam.dD[0]);
  if (!kernel_data.integrator.photon_caustics_differentials) {
    /* Preserve the fixed-radius branch, but leave local rough differentials usable. */
    for (int axis = 0; axis < 4; ++axis)
      beam.dP[axis] = beam.dD[axis] = zero_float3();
  }
  KernelPhotonReconstructionMetadata metadata = {};
  uint glossy = 0, transmission = 0;
#  ifdef __VOLUME__
  if (kernel_data.integrator.use_volumes ||
      (kernel_data.kernel_features & KERNEL_FEATURE_NESTED_DIELECTRICS)) {
    integrator_state_write_ray(state, &ray);
    if (!integrator_volume_stack_init(
            kg, state, PATH_RAY_VISIBILITY_ALL & ~PATH_RAY_VISIBILITY_SHADOW)) {
      atomic_fetch_and_or_uint32(cache.cache_miss, 1);
      return;
    }
  }
#  endif
#  ifdef __VOLUME__
  if (kernel_data.integrator.use_volumes) {
    for (int i = 0; i < kernel_data.volume_stack_size; ++i) {
      if (INTEGRATOR_STATE_ARRAY(state, volume_stack, i, shader) == SHADER_NONE)
        break;
      photon_path.volume_layers++;
    }
  }
#  endif
  int depth = 0;
  bool first_surface = true;
  const int step_limit = kernel_data.integrator.max_bounce + VOLUME_BOUNDS_MAX +
                         kernel_data.integrator.transparent_max_bounce;
  for (int step = 0; step < step_limit; ++step) {
#  ifdef __VOLUME__
    photon_path.in_volume = kernel_data.integrator.use_volumes &&
                            !integrator_state_volume_stack_is_empty(kg, state);
#  endif
    Intersection isect = {};
    isect.object = OBJECT_NONE;
    isect.prim = PRIM_NONE;
    /* ALL includes SHADOW_OPAQUE, which makes the software BVH return its first
     * visited hit instead of the nearest one. Photon transport needs ordered
     * interfaces, especially when entering/exiting participating media. */
    const bool hit = scene_intersect(
        kg, &ray, PATH_RAY_VISIBILITY_ALL & ~PATH_RAY_VISIBILITY_SHADOW, &isect);
#  ifdef __VOLUME__
    if (kernel_data.integrator.use_volumes && !integrator_state_volume_stack_is_empty(kg, state)) {
      Ray segment = ray;
      segment.tmax = hit ? isect.t : ray.tmax;
      ShaderData volume_sd;
      VolumeIntegrateResult result = {};
      INTEGRATOR_STATE_WRITE(state, path, throughput) = power;
      if (!photon_caustics_volume_segment(
              kg, state, &segment, lcg_step_uint(&rng), &volume_sd, result, &photon_path))
      {
        atomic_fetch_and_or_uint32(cache.cache_miss, 1);
        return;
      }
      power = result.indirect_throughput;
      if (result.indirect_scatter) {
        if (depth > 0) {
          /* Density is sigma_s * incident flux. Divide out sigma_s at the deposit
           * to reconstruct incident radiance; camera free-flight supplies sigma_s.
           * Stop at this first non-specular vertex. Further volume scattering is
           * traced by the camera prefix, avoiding multiple counts of one path. */
          volume_sd.P = ray.P + result.indirect_t * ray.D;
          volume_sd.wi = -ray.D;
          beam.valid = false;
          photon_caustics_store(kg,
                                state,
                                cache,
                                &volume_sd,
                                safe_divide_color(power, result.indirect_sigma_s),
                                light,
                                glossy,
                                transmission,
                                photon_index,
                                beam,
                                metadata,
                                true,
                                INTEGRATOR_STATE(state, path, flag) & PATH_RAY_SPECTRAL,
                                INTEGRATOR_STATE(state, path, transparent_bounce));
        }
        return;
      }
      if (is_zero(power) || !isfinite_safe(power)) {
        return;
      }
    }
#  endif
    if (!hit) {
      return;
    }
    ShaderData sd;
    shader_setup_from_ray(kg, &sd, &ray, &isect);
    sd.photon_path = &photon_path;
#  if defined(__KERNEL_OPTIX__) && OPTIX_VERSION >= 80000
    /* Opt-in only in the photon raygen. State slots and RNG use explicit photon
     * IDs, not the reordered lane. Unsupported devices legally treat this as a
     * no-op; retain the existing wavefront scheduler for camera paths. */
    if (kernel_data.integrator.use_photon_ser) {
      optixReorder(uint(sd.shader & SHADER_MASK), 4);
    }
#  endif
#  ifdef __SPECTRAL__
    sd.rand_wavelength = photon_group_wavelength_random(
        cache.wavelength_random,
        photon_wavelength_group(photon_index, cache.iteration, cache.wavelength_groups),
        cache.wavelength_groups);
#  endif
    const bool covered_boundary = dielectric_surface_setup<false>(kg, state, &sd);
    const bool caster = (sd.object_flag & SD_OBJECT_CAUSTICS_CASTER) != 0;
    if (caster && ((sd.object_flag & SD_OBJECT_PHOTON_ZERO_ROUGHNESS) ||
                   (sd.shader_flag & SD_PHOTON_ZERO_ROUGHNESS)))
    {
      sd.runtime_flag |= SR_PHOTON_ZERO_ROUGHNESS;
    }
    const bool geometry_normal = caster && photon_caustics_use_geometry_normal(&sd);
    surface_shader_eval<KERNEL_FEATURE_NODE_MASK_SURFACE & ~KERNEL_FEATURE_NODE_AOV>(
        kg, state, &sd, nullptr, PATH_RAY_VISIBILITY_NONE, 0, false, geometry_normal);
    if (sd.runtime_flag & SR_CACHE_MISS) {
      atomic_fetch_and_or_uint32(cache.cache_miss, 1);
      return;
    }
#  ifdef __VOLUME__
    if ((sd.shader_flag & SD_HAS_ONLY_VOLUME) || covered_boundary) {
      photon_shader_path_advance(&photon_path, isect.t);
      photon_path.origin = sd.P;
      if (!(sd.runtime_flag & SR_BACKFACING) && (sd.shader_flag & SD_HAS_VOLUME))
        photon_path.volume_layers++;
      path_media_enter_exit<false>(kg, state, &sd);
      for (int axis = 0; axis < 4; ++axis) {
        beam.dP[axis] += isect.t * beam.dD[axis];
      }
      ray.dP = sd.dP;
      ray.P += isect.t * ray.D;
      ray.tmin = 0.0f;
      ray.self.object = sd.object;
      ray.self.prim = sd.prim;
      continue;
    }
#  endif
    /* A splat already passed the geometry's stochastic opacity test. Its material
     * can still transmit straight through, without becoming a mesh caustic caster. */
    const bool splat = (sd.type & PRIMITIVE_GSPLAT) != 0;
    const bool first_caster_surface = first_surface && !splat;
    if ((!splat && !(sd.type & PRIMITIVE_TRIANGLE)) || (splat && !(sd.runtime_flag & SR_TRANSPARENT)) ||
        (sd.shader_flag & SD_HAS_RAYTRACE) ||
        (first_caster_surface && target_object != OBJECT_NONE &&
         (sd.object != target_object || sd.prim != target_prim)))
    {
      return;
    }
#  ifdef __SPECTRAL__
    if ((sd.runtime_flag & SR_BSDF_HAS_DISPERSION) &&
        !(INTEGRATOR_STATE(state, path, flag) & PATH_RAY_SPECTRAL))
    {
      /* Estimate the incident spectrum once, after any upstream RGB filtering. */
      const float wavelength = sample_wavelength(sd.rand_wavelength);
      const float3 basis = photon_caustics_source_basis(kg, wavelength);
      power = photon_caustics_evaluate_source(kg, power, basis);
      if (is_zero(power)) {
        return;
      }
      INTEGRATOR_STATE_WRITE(state, path, flag) |= PATH_RAY_SPECTRAL;
    }
#  endif
    if (splat) {
      for (int axis = 0; axis < 4; ++axis) {
        beam.dP[axis] += isect.t * beam.dD[axis];
      }
    }
    else {
      first_surface = false;
      photon_differential_transfer(&beam, ray.D, sd.Ng, isect.t);
    }
    if (depth > 0 && photon_caustics_receiver(&sd)) {
      photon_caustics_store(kg,
                            state,
                            cache,
                            &sd,
                            power,
                            light,
                            glossy,
                            transmission,
                            photon_index,
                            beam,
                            metadata,
                            false,
                            INTEGRATOR_STATE(state, path, flag) & PATH_RAY_SPECTRAL,
                            INTEGRATOR_STATE(state, path, transparent_bounce));
      return;
    }
    if (!splat && !photon_caustics_caster(&sd)) {
      return;
    }
    if (first_caster_surface && cache.guide_hits &&
        (photon_guide_light_supported(klight) ||
         (cache.area_guide_statistics && klight->type == LIGHT_AREA)))
    {
      cache.guide_hits[photon_index] = make_float4((sd.P - center) / radius, 0.0f);
      if (cache.area_guide_statistics && source_cell < PHOTON_AREA_GUIDE_CELLS) {
        ((ccl_global uint *)&cache.guide_hits[photon_index].w)[0] = (source_cell + 1u) << 1;
      }
    }
    /* Same Owen-scrambled sequence construction as Cycles' first-sample blue
     * noise, indexed by photon rather than camera pixel. Rounds/devices have
     * independent seeds. No claim of blue-noise spacing on arbitrary geometry. */
    const uint event_index = depth + INTEGRATOR_STATE(state, path, transparent_bounce);
    float3 rand_bsdf = sobol_burley_sample_3D(
        photon_index, PRNG_SURFACE_BSDF + event_index * PRNG_BOUNCE_NUM, iteration, 0xffffffffu);
    const ccl_private ShaderClosure *sc = surface_shader_bsdf_bssrdf_pick(&sd, &rand_bsdf);
    if (splat && !CLOSURE_IS_BSDF_TRANSPARENT(sc->type)) {
      return;
    }
    /* A straight transparent crossing preserves the receiver of light linking. Portals
     * redirect the light path and count as a transmission event. */
    const bool first_scatter = depth == 0 && photon_path.portal_depth == 0;
    if (sc->type == CLOSURE_BSDF_RAY_PORTAL_ID) {
      if (!kernel_data.integrator.caustics_refractive ||
          (first_scatter && !photon_caustics_light_linked(kg, sd.object, light)) ||
          !material_caustics_allowed(kg, sd.shader, LABEL_TRANSMIT) ||
          INTEGRATOR_STATE(state, path, transparent_bounce) + 1 >=
              kernel_data.integrator.transparent_max_bounce)
        return;
      const ccl_private RayPortalClosure *pc = (const ccl_private RayPortalClosure *)sc;
      float sum_weight = 0.0f;
      for (int i = 0; i < sd.num_closure; ++i)
        if (CLOSURE_IS_BSDF_OR_BSSRDF(sd.closure[i].type))
          sum_weight += sd.closure[i].sample_weight;
      if (!(pc->sample_weight > 0.0f) || !(sum_weight > 0.0f) || is_zero(pc->D) ||
          !isfinite_safe(pc->D) || !isfinite_safe(pc->P))
        return;
      power *= pc->weight * (sum_weight / pc->sample_weight);
      if (is_zero(power) || !isfinite_safe(power))
        return;
      if (cache.events) {
        kernel_assert(event_index < cache.event_stride);
        ccl_global KernelPhotonCausticsEvent *event =
            &cache.events[size_t(photon_index) * cache.event_stride + event_index];
        event->object = sd.object;
        event->shader = sd.shader;
        event->event = LPE_TRANSMISSION;
      }
      if (cache.reconstruction_metadata && (kernel_data.integrator.photon_reconstruction_filter &
                                            PHOTON_RECONSTRUCTION_FILTER_PATH))
        metadata.path_hash = photon_path_hash_append(
            metadata.path_hash, sd.object, sd.shader & SHADER_MASK, LPE_TRANSMISSION);
      photon_caustics_portal_differentials(
          kg, state, &sd, int(sc - sd.closure), &ray, &isect, &beam);
      if (sd.runtime_flag & SR_CACHE_MISS) {
        atomic_fetch_and_or_uint32(cache.cache_miss, 1);
        return;
      }
      photon_shader_path_advance(&photon_path, isect.t);
      const bool relocated = len_squared(sd.P - pc->P) > 1e-9f;
      ray.D = pc->D;
      ray.P = relocated ? pc->P : photon_caustics_ray_origin(kg, &sd, ray.D);
      ray.tmin = 0.0f;
      ray.tmax = FLT_MAX;
      ray.self.object = relocated ? OBJECT_NONE : sd.object;
      ray.self.prim = relocated ? PRIM_NONE : sd.prim;
      photon_path.origin = ray.P;
      photon_path.portal_depth++;
      path_state_next(kg, state, LABEL_TRANSMIT | LABEL_RAY_PORTAL, sd.runtime_flag);
#  ifdef __VOLUME__
      if (kernel_data.integrator.use_volumes ||
          (kernel_data.kernel_features & KERNEL_FEATURE_NESTED_DIELECTRICS))
      {
        if (!photon_caustics_portal_media_init(kg, state, &ray, &photon_path)) {
          atomic_fetch_and_or_uint32(cache.cache_miss, 1);
          return;
        }
      }
#  endif
      continue;
    }
    BsdfEval eval;
    float3 wo;
    float pdf, eta, avg_roughness_squared;
    float2 roughness;
    const int label = surface_shader_bsdf_sample_closure(
        kg, &sd, sc, rand_bsdf, &eval, &wo, &pdf, &roughness, &eta, avg_roughness_squared);
    if ((label & LABEL_TRANSPARENT) && pdf > 0.0f) {
      /* Native smooth Thin Wall uses a transparent closure on non-camera rays.
       * Preserve its flux and differential without inventing a focusing bounce:
       * L -> transparent -> D belongs to direct lighting, not the caustic map. */
      if ((!splat && (!kernel_data.integrator.caustics_refractive ||
                      !material_caustics_allowed(kg, sd.shader, LABEL_TRANSMIT))) ||
          INTEGRATOR_STATE(state, path, transparent_bounce) + 1 >=
              kernel_data.integrator.transparent_max_bounce)
      {
        return;
      }
      power *= bsdf_eval_sum(&eval) / pdf;
      if (is_zero(power) || !isfinite_safe(power)) {
        return;
      }
      if (cache.events) {
        kernel_assert(event_index < cache.event_stride);
        ccl_global KernelPhotonCausticsEvent *event =
            &cache.events[size_t(photon_index) * cache.event_stride + event_index];
        event->object = sd.object;
        event->shader = sd.shader;
        event->event = LPE_TRANSPARENT;
      }
      if (cache.reconstruction_metadata && (kernel_data.integrator.photon_reconstruction_filter &
                                            PHOTON_RECONSTRUCTION_FILTER_PATH))
      {
        metadata.path_hash = photon_path_hash_append(
            metadata.path_hash, sd.object, sd.shader & SHADER_MASK, LPE_TRANSPARENT);
      }
      photon_shader_path_advance(&photon_path, isect.t);
      photon_path.last_label = label;
      path_state_next(kg, state, label, sd.runtime_flag);
#  ifdef __VOLUME__
      if (!(sd.runtime_flag & SR_BACKFACING) && (sd.shader_flag & SD_HAS_VOLUME)) {
        photon_path.volume_layers++;
      }
      path_media_enter_exit<false>(kg, state, &sd);
#  endif
      ray.dP = sd.dP;
      ray.D = normalize(wo);
      ray.P = splat ? sd.P : photon_caustics_ray_origin(kg, &sd, ray.D);
      photon_path.origin = ray.P;
      ray.tmin = 0.0f;
      ray.self.object = sd.object;
      ray.self.prim = sd.prim;
      continue;
    }
    if (!(pdf > 0.0f) || !photon_caustics_label_supported(kg, sd.shader, label) ||
        (first_scatter && !photon_caustics_light_linked(kg, sd.object, light)))
    {
      return;
    }
    if (cache.events) {
      /* A light subpath is sampled in the opposite order to the camera-to-light
       * matcher. Preserve every actual scattering event for reverse replay at gather. */
      kernel_assert(event_index < cache.event_stride);
      ccl_global KernelPhotonCausticsEvent *event =
          &cache.events[size_t(photon_index) * cache.event_stride + event_index];
      event->object = sd.object;
      event->shader = sd.shader;
      event->event = (label & LABEL_TRANSMIT) ? LPE_TRANSMISSION : LPE_GLOSSY;
    }
    if (cache.beam_metadata) {
      ccl_global KernelPhotonBeamMetadata *metadata = &cache.beam_metadata[photon_index];
      metadata->previous_position = sd.P;
      metadata->previous_normal = packed_normal(sc->N).value;
      metadata->normal_derivative = zero_float4();
      metadata->valid = 0;
      metadata->object = sd.object;
      metadata->shader = sd.shader & SHADER_MASK;
      metadata->event = (label & LABEL_TRANSMIT) ? LPE_TRANSMISSION : LPE_GLOSSY;
    }
    if (cache.reconstruction_metadata) {
      if (kernel_data.integrator.photon_reconstruction_filter &
          PHOTON_RECONSTRUCTION_FILTER_NORMAL)
      {
        metadata.previous_normal = packed_normal(sc->N).value;
      }
      if (kernel_data.integrator.photon_reconstruction_filter & PHOTON_RECONSTRUCTION_FILTER_PATH)
      {
        metadata.path_hash = photon_path_hash_append(metadata.path_hash,
                                                     sd.object,
                                                     sd.shader & SHADER_MASK,
                                                     (label & LABEL_TRANSMIT) ? LPE_TRANSMISSION :
                                                                                LPE_GLOSSY);
      }
    }
    /* Native microfacet samples already return f*cos and include closure/lobe
     * probabilities in pdf. Their transmission eval/pdf is a flux weight, with
     * no additional radiance eta^2 factor here. Footprint normalization is separate. */
    power *= bsdf_eval_sum(&eval) / pdf;
    if (is_zero(power) || !isfinite_safe(power)) {
      return;
    }
    glossy += bool(label & LABEL_REFLECT);
    transmission += bool(label & LABEL_TRANSMIT);
    if (glossy >= uint(kernel_data.integrator.max_glossy_bounce) ||
        transmission >= uint(kernel_data.integrator.max_transmission_bounce))
    {
      return;
    }
    photon_caustics_scatter_differential(kg,
                                         state,
                                         &ray,
                                         &isect,
                                         &sd,
                                         sc,
                                         rand_bsdf,
                                         normalize(wo),
                                         label,
                                         eta,
                                         cache,
                                         &beam,
                                         photon_index);
    if (sd.runtime_flag & SR_CACHE_MISS) {
      atomic_fetch_and_or_uint32(cache.cache_miss, 1);
      return;
    }
    ray.dP = sd.dP;
    ray.dD = bsdf_widen_dD(kg, ray.dD, avg_roughness_squared);
    photon_shader_path_advance(&photon_path, isect.t);
    photon_path.last_label = label;
    photon_path.depth++;
    photon_path.reflection_depth += bool(label & LABEL_REFLECT);
    photon_path.glossy_depth += bool(label & LABEL_REFLECT);
    photon_path.transmission_depth += bool(label & LABEL_TRANSMIT);
    path_state_next(kg, state, label, sd.runtime_flag);
#  ifdef __VOLUME__
    if (label & LABEL_TRANSMIT) {
      if (!(sd.runtime_flag & SR_BACKFACING) && (sd.shader_flag & SD_HAS_VOLUME))
        photon_path.volume_layers++;
      path_media_enter_exit<false>(kg, state, &sd);
    }
#  endif
    if (++depth >= kernel_data.integrator.max_bounce) {
      return;
    }
    ray.D = normalize(wo);
    ray.P = photon_caustics_ray_origin(kg, &sd, ray.D);
    photon_path.origin = ray.P;
    ray.tmin = 0.0f;
    ray.self.object = sd.object;
    ray.self.prim = sd.prim;
  }
}

#  ifdef __SPECTRAL__
ccl_device_inline Spectrum photon_caustics_dispersion_norm(KernelGlobals kg)
{
  return make_float3(kernel_data.film.xyz_to_r.w, kernel_data.film.xyz_to_g.w,
                     kernel_data.film.xyz_to_b.w);
}

ccl_device_inline Spectrum photon_group_response(KernelGlobals kg,
                                                 const KernelPhotonCausticsCache cache,
                                                 const uint group)
{
  const float u = photon_group_wavelength_random(
      cache.wavelength_random, group, cache.wavelength_groups);
  return dispersion_throughput_weight(kg, u);
}
#  endif

/* The camera throughput already contains its signed wavelength response. Mask its
 * negative components for the photon merge only, before summing contributions. */
struct PhotonCausticsCameraSpectrum {
  Spectrum response_scale;
  float3 source_basis;
};

ccl_device_inline PhotonCausticsCameraSpectrum photon_caustics_camera_spectrum(
    KernelGlobals kg,
    ConstIntegratorState state,
    const KernelPhotonCausticsCache cache,
    const int camera_group)
{
#  ifdef __SPECTRAL__
  if (camera_group >= 0) {
    const Spectrum w = INTEGRATOR_STATE(state, path, throughput);
    const Spectrum mask = make_float3(w.x > 0.0f, w.y > 0.0f, w.z > 0.0f);
    const float u = photon_group_wavelength_random(
        cache.wavelength_random, uint(camera_group), cache.wavelength_groups);
    return {mask * photon_caustics_dispersion_norm(kg),
            photon_caustics_source_basis(kg, sample_wavelength(u))};
  }
#  endif
  return {one_spectrum(), zero_float3()};
}

ccl_device_inline Spectrum photon_caustics_spectral_power(
    KernelGlobals kg,
    const KernelPhotonCausticsCache cache,
    const ccl_global KernelPhotonCaustic *photon,
    const uint photon_index,
    const int camera_group,
    const PhotonCausticsCameraSpectrum camera)
{
  Spectrum power = rgb_to_spectrum(photon->power);
#  ifdef __SPECTRAL__
  if (camera_group >= 0) {
    if (photon->pad & PHOTON_CAUSTICS_SPECTRAL) {
      /* A matched light path already has a spectrum; its selection probability is 1/G. */
      power *= float(cache.wavelength_groups);
    }
    else {
      /* Only the camera crossed a dispersive interface. The light path still carries RGB. */
      power = photon_caustics_evaluate_source(kg, power, camera.source_basis);
    }
    return max(power, zero_spectrum()) * camera.response_scale;
  }
  if (photon->pad & PHOTON_CAUSTICS_SPECTRAL) {
    const uint group = photon_wavelength_group(
        photon_index, cache.iteration, cache.wavelength_groups);
    power *= photon_group_response(kg, cache, group);
    power *= photon_caustics_dispersion_norm(kg);
  }
#  endif
  return max(power, zero_spectrum());
}

ccl_device_inline int photon_caustics_camera_group(KernelGlobals kg,
                                                    ConstIntegratorState state,
                                                    const KernelPhotonCausticsCache cache)
{
#  ifdef __SPECTRAL__
  if (INTEGRATOR_STATE(state, path, flag) & PATH_RAY_SPECTRAL) {
    return int(photon_camera_path_wavelength_group(kg, state, cache.wavelength_groups));
  }
#  endif
  return -1;
}

ccl_device_inline bool photon_caustics_wavelength_matches(
    const KernelPhotonCausticsCache cache,
    const ccl_global KernelPhotonCaustic *photon,
    const int camera_group)
{
  if (camera_group < 0 || !(photon->pad & PHOTON_CAUSTICS_SPECTRAL)) {
    return true;
  }
  return photon_wavelength_merge_weight(
      true, camera_group,
      photon_wavelength_group(uint(photon - cache.photons), cache.iteration,
                              cache.wavelength_groups),
      cache.wavelength_groups) > 0.0f;
}

ccl_device_inline void photon_caustics_film_write(KernelGlobals kg,
                                                  ConstIntegratorState state,
                                                  const ccl_private ShaderData *sd,
                                                  const ccl_private BsdfEval *eval,
                                                  const int light,
                                                  ccl_global float *render_buffer,
                                                  const bool volume = false)
{
  Spectrum contribution = INTEGRATOR_STATE(state, path, throughput) * bsdf_eval_sum(eval);
  const uint bounce = INTEGRATOR_STATE(state, path, bounce);
  /* Photon density peaks are flux divided by a reconstruction area/volume, not
   * rare BSDF weights. Native per-sample clamping removes more physical energy as
   * the footprint shrinks. Keep this estimator independent of native firefly clamps. */
  contribution = ensure_finite(contribution);
  ccl_global float *buffer = film_pass_pixel_render_buffer(kg, state, render_buffer);
  uint32_t path_flag = INTEGRATOR_STATE(state, path, flag);
  PathRayVisibility visibility = INTEGRATOR_STATE(state, path, visibility);
  if (volume && bounce == 0) {
    /* This is scattering, not primary transmission for volume probability guiding. */
    path_flag &= ~PATH_RAY_VOLUME_PRIMARY_TRANSMIT;
    visibility |= PATH_RAY_VISIBILITY_VOLUME_SCATTER;
  }
  const uint camera_sample = INTEGRATOR_STATE(state, path, sample);
  const bool independent = kernel_data.film.photon_segment_stride ||
                           kernel_data.film.photon_compact;
  const uint pixel = INTEGRATOR_STATE(state, path, render_pixel_index);
  const float depth = (bounce == 0) ? camera_z_depth_at_time(kg, sd->P, sd->time) : -1.0f;
  film_write_combined_pass(
      kg, visibility, path_flag, camera_sample, contribution, buffer, depth, pixel, !independent);
#  ifdef __DEEP_OUTPUT__
  const float volume_z = INTEGRATOR_STATE(state, path, deep_volume_z);
  if (kernel_data.film.deep_photon_bins && volume_z >= 0.0f) {
    film_accumulate_deep_photon(kg, pixel, volume_z, contribution);
  }
  else {
    film_accumulate_deep_surface_rgb_path(kg, state, contribution);
  }
#  endif
#  ifdef __PASSES__
  const int lightgroup = object_lightgroup(kg, kernel_data_fetch(lights, light).object_id);
  const DeepFilmContext deep = film_deep_path_context(state);
  film_write_lightgroup_pass(kg, deep, buffer, kernel_data.film.pass_lightgroup, lightgroup, contribution);
  film_write_lightgroup_split_pass(
      kg, deep, buffer, kernel_data.film.pass_lightgroup_photon, lightgroup, contribution);
  /* The caustic contribution already includes camera throughput and material color.
   * Keep Combined, complete lightgroups and LPE, but do not also write ordinary lobes. */
  if (kernel_data.film.pass_caustics != PASS_UNUSED) {
    film_write_deep_pass(kg, deep, buffer, kernel_data.film.pass_caustics, contribution);
  }
#  endif
}

/* Replay a complete photon suffix from this camera prefix. Splitting the receiver's
 * evaluated lobes is deterministic, as in native shadow LPE writes; it does not change
 * the BSDF sampler or Combined. Scene light overrides are already in sampled photon power;
 * display edits run in the existing post-accumulation LPE program. Neither is applied again here.
 */
ccl_device_inline void photon_caustics_lpe_accum(KernelGlobals kg,
                                                 ConstIntegratorState state,
                                                 const ccl_private ShaderData *sd,
                                                 const KernelPhotonCausticsCache cache,
                                                 const uint photon_index,
                                                 const ccl_global KernelPhotonCaustic *photon,
                                                 const ccl_private BsdfEval *eval,
                                                 ccl_private Spectrum *state_contributions,
                                                 const bool volume = false)
{
#if !defined(__KERNEL_GPU__) || defined(__KERNEL_CUDA__)
  if (lpe_query_enabled(kg) && cache.events && cache.path_indices) {
    const uint id = cache.path_indices[photon_index];
    const uint count = photon_event_count(photon->pad);
    const Spectrum weight = INTEGRATOR_STATE(state, path, throughput) * INTEGRATOR_STATE(state, path, lpe_weight);
    const int events[] = {LPE_DIFFUSE, LPE_GLOSSY, LPE_TRANSMISSION};
    const bool skip_receiver = !volume && INTEGRATOR_STATE(state, path, lpe_skip_surface);
    const bool combine_lobes = volume || skip_receiver || query_total_light(kg);
    for (int i = 0; i < (combine_lobes ? 1 : 3); ++i) {
      const Spectrum lobe = combine_lobes ? bsdf_eval_sum(eval) :
                            i == 0 ? eval->diffuse : i == 1 ? eval->glossy :
                            bsdf_eval_sum(eval) - eval->diffuse - eval->glossy;
      if (is_zero(lobe)) {
        continue;
      }
#  ifdef __KERNEL_CUDA__
      uint parent = uint(INTEGRATOR_STATE(state, path, lpe_states));
      if (!skip_receiver) {
        parent = query_gpu_append(parent, sd->object, sd->shader, volume ? LPE_VOLUME : events[i],
                                  -1, zero_spectrum());
      }
      for (uint j = count; j; --j) {
        const auto &v = cache.events[size_t(id) * cache.event_stride + j - 1];
        parent = query_gpu_append(parent, v.object, v.shader, 256 | v.event, -1, zero_spectrum());
      }
      const KernelLight &source = kernel_data_fetch(lights, photon->light);
      query_gpu_append(parent, source.object_id, source.shader_id, KernelQueryRecord::photon_source,
                       INTEGRATOR_STATE(state, path, render_pixel_index), weight * lobe);
#  else
      QueryPath path = kg->query->camera;
      if (!skip_receiver) {
        path.push_back(query_vertex(sd->object, sd->shader, volume ? LPE_VOLUME : events[i]));
      }
      for (uint j = count; j; --j) {
        const auto &v = cache.events[size_t(id) * cache.event_stride + j - 1];
        path.push_back(query_vertex(v.object, v.shader, 256 | v.event));
      }
      const KernelLight &source = kernel_data_fetch(lights, photon->light);
      path.push_back(query_vertex(source.object_id, source.shader_id, 0));
      query_record(kg, path, true, weight * lobe);
#  endif
    }
    if (!kernel_data.film.lpe_state_count) {
      return;
    }
  }
#endif
  const uint64_t active = INTEGRATOR_STATE(state, path, lpe_states);
  const bool skip = !volume && INTEGRATOR_STATE(state, path, lpe_skip_surface);
  uint64_t states[3];
  states[0] = skip ?
                  active :
                  lpe_transition(
                      kg, active, sd->object, sd->shader, volume ? LPE_VOLUME : LPE_DIFFUSE, true);
  states[1] = (skip || volume) ?
                  0 :
                  lpe_transition(kg, active, sd->object, sd->shader, LPE_GLOSSY, true);
  states[2] = (skip || volume) ?
                  0 :
                  lpe_transition(kg, active, sd->object, sd->shader, LPE_TRANSMISSION, true);

  const uint path_index = cache.path_indices[photon_index];
  const uint chain_length = photon_event_count(photon->pad);
  kernel_assert(path_index < cache.capacity && chain_length <= cache.event_stride);
  for (uint i = chain_length; i > 0; --i) {
    if (!(states[0] | states[1] | states[2])) {
      /* Only LPE classification is pruned; Combined is accumulated independently. */
      return;
    }
    const ccl_global KernelPhotonCausticsEvent *event =
        &cache.events[size_t(path_index) * cache.event_stride + i - 1];
    for (int lobe = 0; lobe < 3; ++lobe) {
      if (states[lobe]) {
        states[lobe] = lpe_transition(
            kg, states[lobe], event->object, event->shader, event->event, false, true);
      }
    }
  }

  Spectrum lobes[3];
  lobes[0] = (skip || volume) ? bsdf_eval_sum(eval) : eval->diffuse;
  lobes[1] = (skip || volume) ? zero_spectrum() : eval->glossy;
  lobes[2] = (skip || volume) ? zero_spectrum() :
                                bsdf_eval_sum(eval) - eval->diffuse - eval->glossy;
  const Spectrum weight = INTEGRATOR_STATE(state, path, lpe_weight);
  const ccl_global KernelLight *klight = &kernel_data_fetch(lights, photon->light);
  for (int lobe = 0; lobe < 3; ++lobe) {
    if (states[lobe] && !is_zero(lobes[lobe])) {
  film_write_lpe<true, true>(kg,
                           states[lobe],
                           weight,
                           klight->object_id,
                           lobes[lobe],
                           nullptr,
                           klight->shader_id,
                           state_contributions);
    }
  }
}

/* Geometry-only candidate classification shared by counting and evaluation. */
ccl_device_inline bool photon_caustics_surface_candidate(
    KernelGlobals kg,
    ConstIntegratorState state,
    const ccl_private ShaderData *sd,
    const KernelPhotonCausticsCache cache,
    const int camera_group,
    const ccl_global KernelPhotonCaustic *photon,
    const int3 cell,
    const float cell_size,
    const float base_radius,
    const float3 query_P,
    const float3 geometric_normal,
    const float2 scatter_offset,
    ccl_private float &kernel_peak,
    ccl_private float &kernel_weight)
{
  const float size = kernel_data_fetch(lights, photon->light).photon_radius_scale;
  const float radius = base_radius * size;
  const bool differential = kernel_data.integrator.photon_caustics_differentials;
  const bool regional_ellipse = kernel_data.integrator.use_photon_adaptive_radius &&
                                kernel_data.integrator.photon_reconstruction_method ==
                                    PHOTON_RECONSTRUCTION_ANISOTROPIC;
  const float radius_squared = radius * radius;
  const float normalization = 2.0f * M_1_PI_F / radius_squared;
  /* Full cell comparison is essential: two of the 27 neighbors may hash to
   * the same bucket. Without it a single photon can be accumulated twice. */
  const int3 photon_cell = photon_caustics_cell(photon->P, cell_size);
  if (photon_cell.x != cell.x || photon_cell.y != cell.y || photon_cell.z != cell.z ||
      photon->receiver_object != sd->object ||
      !photon_caustics_light_supported(kg, photon->light) ||
      !photon_caustics_wavelength_matches(cache, photon, camera_group))
  {
    return false;
  }
  const bool rough_footprint = photon->footprint.w < 0.0f;
  const float3 delta = (rough_footprint ? sd->P : sd->P + (query_P - sd->P) * size) - photon->P;
  const float distance_squared = len_squared(delta);
  if (distance_squared >= cell_size * cell_size) {
    return false;
  }
  packed_normal photon_normal;
  photon_normal.value = photon->normal;
  if (dot(photon_normal.decode(), geometric_normal) < 0.9f ||
      fabsf(dot(delta, geometric_normal)) > radius * 0.1f)
  {
    return false;
  }
  const float photon_radius_squared = kernel_data.integrator.use_photon_adaptive_radius ?
                                          sqr(size) / photon->footprint.x :
                                          radius_squared;
  if (!differential && !rough_footprint && !regional_ellipse &&
      distance_squared >= photon_radius_squared)
    return false;
  kernel_peak = kernel_data.integrator.use_photon_adaptive_radius ?
                          2.0f * M_1_PI_F / photon_radius_squared :
                          normalization;
  kernel_weight = kernel_peak * (1.0f - distance_squared / photon_radius_squared);
  if (differential || rough_footprint || regional_ellipse) {
    float3 U, V;
    make_orthonormals(photon_normal.decode(), &U, &V);
    const float u = dot(delta, U), v = dot(delta, V);
    const float4 Q = photon->footprint / sqr(size);
    /* Scatter in the ellipse's normalized coordinates, preserving its small axis. */
    const float a = sqrtf(Q.x), b = Q.y / a;
    const float c = sqrtf(max(Q.z - b * b, 0.0f));
    /* Fixed footprints already jitter query_P in world space. A regional
     * anisotropic update must not apply that same scatter a second time. */
    const float2 z = make_float2(a * u + b * v, c * v) -
                     ((differential || rough_footprint) ? scatter_offset : zero_float2());
    const float q = dot(z, z);
    if (q >= 1.0f) {
      return false;
    }
    kernel_peak = fabsf(Q.w);
    kernel_weight = kernel_peak * (1.0f - q);
  }
  const uint glossy = photon_reflection_depth(photon->pad);
  const uint transmission = photon_transmission_depth(photon->pad);
  const uint bounce = INTEGRATOR_STATE(state, path, bounce);
  const uint passthrough = photon_passthrough_depth(photon->pad);
  if (passthrough && INTEGRATOR_STATE(state, path, transparent_bounce) + passthrough >=
                         uint(kernel_data.integrator.transparent_max_bounce))
  {
    return false;
  }
  if (bounce + glossy + transmission + 1 > uint(kernel_data.integrator.max_bounce) ||
      INTEGRATOR_STATE(state, path, glossy_bounce) + glossy >=
          uint(kernel_data.integrator.max_glossy_bounce) ||
      INTEGRATOR_STATE(state, path, transmission_bounce) + transmission >=
          uint(kernel_data.integrator.max_transmission_bounce) ||
      bounce + glossy + transmission >
          kernel_data_fetch(lights, photon->light).max_bounces)
  {
    return false;
  }
  packed_normal photon_direction;
  photon_direction.value = photon->direction;
  const float3 wi = photon_direction.decode();
  if (dot(sd->Ng, wi) <= 0.0f) {
    return false;
  }
  return kernel_weight > 0.0f;
}

ccl_device void integrator_photon_caustics_gather(KernelGlobals kg,
                                                  IntegratorState state,
                                                  ccl_private ShaderData *sd,
                                                  ccl_global float *render_buffer)
{
  if (((kernel_data.film.photon_segment_stride || kernel_data.film.photon_compact) &&
       !path_state_photon_only(kg, state)) ||
      !photon_caustics_active(kg) || !photon_caustics_receiver(sd))
  {
    return;
  }
  const KernelPhotonCausticsCache cache = photon_caustics_cache(kg);
  const int camera_group = photon_caustics_camera_group(kg, state, cache);
  const auto camera_response = photon_caustics_camera_spectrum(kg, state, cache, camera_group);
  const float radius = cache.radius;
  const bool differential = kernel_data.integrator.photon_caustics_differentials;
  const float cell_size = photon_caustics_cell_size(kg, cache);
  float2 scatter_offset = zero_float2();
  const float3 geometric_normal = photon_caustics_geometric_normal(sd);
  float3 query_P = sd->P;
  const float scatter = photon_caustics_effective_scatter(kg);
  if (scatter > 0.0f) {
    /* A normalized tangent-disk convolution using the scene's blue-noise/Sobol
     * sampler. Preserve its pixel indexing: hashing rng_pixel here would destroy
     * spatial blue-noise structure. Use absolute samples and a separate dimension
     * so tiling, sample subsets and LPE selections retain the same realization.
     * This changes reconstruction, never photon power or the physical hit. */
    const float2 rand = path_rng_2D(kg,
                                    INTEGRATOR_STATE(state, path, rng_pixel),
                                    INTEGRATOR_STATE(state, path, sample),
                                    0x50484f54 + 4 * INTEGRATOR_STATE(state, path, bounce));
    const float angle = M_2PI_F * rand.x;
    scatter_offset = scatter * sqrtf(rand.y) * make_float2(cosf(angle), sinf(angle));
    if (!differential)
      query_P += (scatter * radius) * disk_light_sample(geometric_normal, rand);
  }
  const int3 center = photon_caustics_cell(sd->P, cell_size);
  const uint gather_limit = kernel_data.integrator.photon_caustics_gather_max;
  uint candidates = 0;
  if (gather_limit) {
    /* This still traverses all entries; only expensive BSDF/LPE evaluation is
     * bounded. Count across all lights so the budget is per shading point. */
    for (int z = -1; z <= 1; ++z) {
      for (int y = -1; y <= 1; ++y) {
        for (int x = -1; x <= 1; ++x) {
          const int3 cell = make_int3(center.x + x, center.y + y, center.z + z);
          uint entry = cache.hash[photon_caustics_hash(cache, cell)];
          while (entry && entry <= cache.capacity) {
            const ccl_global KernelPhotonCaustic *photon = &cache.photons[entry - 1u];
            entry = photon->next;
            float peak, weight;
            candidates += photon_caustics_surface_candidate(kg, state, sd, cache, camera_group, photon, cell,
                cell_size, radius, query_P, geometric_normal, scatter_offset, peak, weight);
          }
        }
      }
    }
    if (!candidates) return;
  }
  PhotonGatherSelection selection = photon_gather_selection(candidates, gather_limit,
      hash_uint3(INTEGRATOR_STATE(state, path, rng_pixel),
                 INTEGRATOR_STATE(state, path, sample),
                 hash_uint2(cache.iteration, INTEGRATOR_STATE(state, path, bounce) ^ 0x67617468u)));
  /* Accumulate locally before film atomics to preserve small contributions. */
  for (int light = 0; light < kernel_data.integrator.num_lights; ++light) {
    if (!photon_caustics_light_supported(kg, light)) {
      continue;
    }
    BsdfEval total;
    bsdf_eval_init(&total, zero_spectrum());
    const int lpe_state_count = kernel_data.film.lpe_state_count;
    kernel_assert(lpe_state_count <= 64);
    Spectrum lpe_contributions[64];
    for (int i = 0; i < lpe_state_count; ++i) {
      lpe_contributions[i] = zero_spectrum();
    }
    /* Small photon/lobe values must not be added one by one to a large progressive
     * float film sum: colored mixed receivers lose their weaker components that way.
     * Classify into per-sample terminal sums before the film write. */
    for (int z = -1; z <= 1; ++z) {
      for (int y = -1; y <= 1; ++y) {
        for (int x = -1; x <= 1; ++x) {
          const int3 cell = make_int3(center.x + x, center.y + y, center.z + z);
          uint entry = photon_query_head(cache, cell, light, false);
          while (entry != 0) {
            if (entry > cache.capacity) {
              break;
            }
            const uint photon_index = entry - 1u;
            const ccl_global KernelPhotonCaustic *photon = &cache.photons[photon_index];
            entry = cache.query_next ? cache.query_next[photon_index] : photon->next;
            if (photon->light != light) {
              continue;
            }
            float kernel_peak, kernel_weight;
            if (!photon_caustics_surface_candidate(kg, state, sd, cache, camera_group, photon, cell, cell_size,
                    radius, query_P, geometric_normal, scatter_offset, kernel_peak, kernel_weight) ||
                !photon_gather_select(&selection)) {
              continue;
            }
            const uint glossy = photon_reflection_depth(photon->pad);
            packed_normal photon_direction;
            photon_direction.value = photon->direction;
            const float3 wi = photon_direction.decode();
            BsdfEval eval;
            bsdf_eval_init(&eval, zero_spectrum());
            for (int i = 0; i < sd->num_closure; ++i) {
              const ccl_private ShaderClosure *sc = &sd->closure[i];
              if (!CLOSURE_IS_BSDF(sc->type)) {
                continue;
              }
              const bool diffuse = CLOSURE_IS_BSDF_DIFFUSE(sc->type);
              if ((diffuse && INTEGRATOR_STATE(state, path, diffuse_bounce) + 1 >=
                                  uint(kernel_data.integrator.max_diffuse_bounce)) ||
                  (!diffuse && INTEGRATOR_STATE(state, path, glossy_bounce) + glossy + 1 >=
                                   uint(kernel_data.integrator.max_glossy_bounce)))
              {
                continue;
              }
              float pdf;
              const float cos_in = fabsf(dot(sc->N, wi));
              if (cos_in > 1e-7f) {
                /* Flux density already contains the geometric incidence cosine. Native
                 * BSDF evaluation returns f*cos, so recover f before multiplying flux. */
                const Spectrum f = bsdf_eval(kg, sd, sc, wi, &pdf) * sc->weight / cos_in;
                bsdf_eval_accum(&eval, sc, wi, f);
              }
            }
            if (!bsdf_eval_is_zero(&eval)) {
              photon_guide_record(kg, state, cache, photon_index, photon);
            }
            kernel_weight = photon_caustics_noisy_weight(
                kg, state, cache, photon_index, kernel_weight, kernel_peak, differential);
            if (!(kernel_weight > 0.0f)) {
              continue;
            }
            /* Epanechnikov disk kernel integrates to one on a flat receiving surface. */
            const Spectrum power = photon_caustics_spectral_power(
                kg, cache, photon, photon_index, camera_group, camera_response);
            bsdf_eval_mul(&eval, power * (kernel_weight * selection.weight));
            if (!bsdf_eval_is_zero(&eval)) {
              total.diffuse += eval.diffuse;
              total.glossy += eval.glossy;
              total.sum += eval.sum;
              if (lpe_state_count || lpe_query_enabled(kg)) {
                photon_caustics_lpe_accum(
                    kg, state, sd, cache, photon_index, photon, &eval, lpe_contributions);
              }
            }
          }
        }
      }
    }
    if (!bsdf_eval_is_zero(&total)) {
      photon_caustics_film_write(kg, state, sd, &total, light, render_buffer);
      const Spectrum throughput = INTEGRATOR_STATE(state, path, throughput);
      ccl_global float *buffer = film_pass_pixel_render_buffer(kg, state, render_buffer);
      for (int i = 0; i < lpe_state_count; ++i) {
        if (!is_zero(lpe_contributions[i])) {
          /* Source/Union/weighted decisions were made for each physical path before
           * summation. Different terminal states may legitimately share one pass. */
          const int offset =
              kernel_data_fetch(lpe_states, kernel_data.film.lpe_state_offset + i).pass_offset;
          const int channel = kernel_data_fetch(lpe_states,
              kernel_data.film.lpe_state_offset + i).deep_channel;
          if (kernel_data.film.deep_physical_ptr && channel >= 0)
            deep_physical_write(kernel_data.film.deep_physical_ptr,
                                INTEGRATOR_STATE(state, path, render_pixel_index),
                                kernel_data.film.deep_lpe_channels,
                                INTEGRATOR_STATE(state, path, deep_volume_z),
                                INTEGRATOR_STATE(state, path, deep_surface_sample_idx),
                                spectrum_to_rgb(ensure_finite(throughput * lpe_contributions[i])),
                                channel);
          kernel_assert(offset != PASS_UNUSED);
          film_write_pass_spectrum(buffer + offset,
                                   ensure_finite(throughput * lpe_contributions[i]));
        }
      }
    }
  }
}

#  ifdef __VOLUME__
ccl_device void photon_caustics_volume_gather(KernelGlobals kg,
                                              ConstIntegratorState state,
                                              const ccl_private ShaderData *sd,
                                              const ccl_private ShaderVolumePhases *phases,
                                              ccl_global float *render_buffer)
{
  if (((kernel_data.film.photon_segment_stride || kernel_data.film.photon_compact) &&
       !path_state_photon_only(kg, state)) ||
      !photon_caustics_active(kg) || phases->num_closure == 0 ||
      INTEGRATOR_STATE(state, path, volume_bounce) + 1 >=
          uint(kernel_data.integrator.max_volume_bounce))
  {
    return;
  }
  const KernelPhotonCausticsCache cache = photon_caustics_cache(kg);
  const int camera_group = photon_caustics_camera_group(kg, state, cache);
  const auto camera_response = photon_caustics_camera_spectrum(kg, state, cache, camera_group);
  const float cell_size = photon_caustics_cell_size(kg, cache);
  const float3 query_P = sd->P;
  const int3 center = photon_caustics_cell(query_P, cell_size);
  /* Accumulate locally before film atomics to preserve small contributions. */
  const PhotonVolumeKey medium = photon_volume_key(kg, state);
  for (int light = 0; light < kernel_data.integrator.num_lights; ++light) {
    if (!photon_caustics_light_supported(kg, light)) {
      continue;
    }
    const float size = kernel_data_fetch(lights, light).photon_radius_scale;
    const float radius = cache.radius * size;
    const float radius_squared = radius * radius;
    /* Integral over the unit ball of (1-r^2) is 8*pi/15. */
    const float normalization = (15.0f / 8.0f) * M_1_PI_F / (radius_squared * radius);
    BsdfEval total;
    bsdf_eval_init(&total, zero_spectrum());
    const int lpe_state_count = kernel_data.film.lpe_state_count;
    kernel_assert(lpe_state_count <= 64);
    Spectrum lpe_contributions[64];
    for (int i = 0; i < lpe_state_count; ++i) {
      lpe_contributions[i] = zero_spectrum();
    }
    /* Small photon/lobe values must not be added one by one to a large progressive
     * float film sum: colored mixed receivers lose their weaker components that way.
     * Classify into per-sample terminal sums before the film write. */
    for (int z = -1; z <= 1; ++z) {
      for (int y = -1; y <= 1; ++y) {
        for (int x = -1; x <= 1; ++x) {
          const int3 cell = make_int3(center.x + x, center.y + y, center.z + z);
          uint entry = photon_query_head(cache, cell, light, true);
          while (entry != 0) {
            if (entry > cache.capacity) {
              break;
            }
            const uint photon_index = entry - 1u;
            const ccl_global KernelPhotonCaustic *photon = &cache.photons[photon_index];
            entry = cache.query_next ? cache.query_next[photon_index] : photon->next;
            /* Full cell comparison is essential: two of the 27 neighbors may hash to
             * the same bucket. Without it a single photon can be accumulated twice. */
            const int3 photon_cell = photon_caustics_cell(photon->P, cell_size);
            if (photon_cell.x != cell.x || photon_cell.y != cell.y || photon_cell.z != cell.z ||
                photon->receiver_object != medium.receiver ||
                photon->volume_signature != medium.signature || photon->light != light ||
                !photon_caustics_wavelength_matches(cache, photon, camera_group))
            {
              continue;
            }
            const float3 delta = query_P - photon->P;
            const float distance_squared = len_squared(delta);
            if (distance_squared >= cell_size * cell_size) {
              continue;
            }
            const float photon_radius_squared = kernel_data.integrator.use_photon_adaptive_radius ?
                                                    sqr(size) / photon->footprint.x :
                                                    radius_squared;
            if (distance_squared >= photon_radius_squared) {
              continue;
            }
            const float photon_normalization = kernel_data.integrator.use_photon_adaptive_radius ?
                                                   (15.0f / 8.0f) * M_1_PI_F /
                                                       (photon_radius_squared *
                                                        sqrtf(photon_radius_squared)) :
                                                   normalization;
            const uint glossy = photon_reflection_depth(photon->pad);
            const uint transmission = photon_transmission_depth(photon->pad);
            const uint bounce = INTEGRATOR_STATE(state, path, bounce);
            const uint passthrough = photon_passthrough_depth(photon->pad);
            if (passthrough && INTEGRATOR_STATE(state, path, transparent_bounce) + passthrough >=
                                   uint(kernel_data.integrator.transparent_max_bounce))
            {
              continue;
            }
            if (bounce + glossy + transmission + 1 > uint(kernel_data.integrator.max_bounce) ||
                INTEGRATOR_STATE(state, path, glossy_bounce) + glossy >=
                    uint(kernel_data.integrator.max_glossy_bounce) ||
                INTEGRATOR_STATE(state, path, transmission_bounce) + transmission >=
                    uint(kernel_data.integrator.max_transmission_bounce) ||
                bounce + glossy + transmission >
                    kernel_data_fetch(lights, photon->light).max_bounces)
            {
              continue;
            }
            packed_normal photon_direction;
            photon_direction.value = photon->direction;
            const float3 wi = photon_direction.decode();
            BsdfEval eval;
            bsdf_eval_init(&eval, zero_spectrum());
            /* The camera's sampled free flight contains sigma_s. The stored power
             * has sigma_s divided out, so evaluate only the normalized phase here. */
            _volume_shader_phase_eval_mis(sd, phases, wi, nullptr, &eval, 0.0f, 0.0f);
            if (!bsdf_eval_is_zero(&eval)) {
              photon_guide_record(kg, state, cache, photon_index, photon);
            }
            const float kernel_weight = photon_caustics_noisy_weight(
                kg,
                state,
                cache,
                photon_index,
                photon_normalization * (1.0f - distance_squared / photon_radius_squared),
                photon_normalization,
                false);
            if (!(kernel_weight > 0.0f)) {
              continue;
            }

            const Spectrum power = photon_caustics_spectral_power(
                kg, cache, photon, photon_index, camera_group, camera_response);
            bsdf_eval_mul(&eval, power * kernel_weight);
            if (!bsdf_eval_is_zero(&eval)) {
              total.diffuse += eval.diffuse;
              total.glossy += eval.glossy;
              total.sum += eval.sum;
              if (lpe_state_count || lpe_query_enabled(kg)) {
                photon_caustics_lpe_accum(
                    kg, state, sd, cache, photon_index, photon, &eval, lpe_contributions, true);
              }
            }
          }
        }
      }
    }
    if (!bsdf_eval_is_zero(&total)) {
      photon_caustics_film_write(kg, state, sd, &total, light, render_buffer, true);
      const Spectrum throughput = INTEGRATOR_STATE(state, path, throughput);
      ccl_global float *buffer = film_pass_pixel_render_buffer(kg, state, render_buffer);
      for (int i = 0; i < lpe_state_count; ++i) {
        if (!is_zero(lpe_contributions[i])) {
          /* Source/Union/weighted decisions were made for each physical path before
           * summation. Different terminal states may legitimately share one pass. */
          const int offset =
              kernel_data_fetch(lpe_states, kernel_data.film.lpe_state_offset + i).pass_offset;
          const int channel = kernel_data_fetch(lpe_states,
              kernel_data.film.lpe_state_offset + i).deep_channel;
          if (kernel_data.film.deep_physical_ptr && channel >= 0)
            deep_physical_write(kernel_data.film.deep_physical_ptr,
                                INTEGRATOR_STATE(state, path, render_pixel_index),
                                kernel_data.film.deep_lpe_channels,
                                INTEGRATOR_STATE(state, path, deep_volume_z),
                                INTEGRATOR_STATE(state, path, deep_surface_sample_idx),
                                spectrum_to_rgb(ensure_finite(throughput * lpe_contributions[i])),
                                channel);
          kernel_assert(offset != PASS_UNUSED);
          film_write_pass_spectrum(buffer + offset,
                                   ensure_finite(throughput * lpe_contributions[i]));
        }
      }
    }
  }
}

#  endif

#endif /* __PHOTON_CAUSTICS__ */

CCL_NAMESPACE_END
