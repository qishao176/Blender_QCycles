/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: Apache-2.0 */

#pragma once

#include "kernel/bvh/bvh.h"
#include "kernel/geom/shader_data.h"
#include "kernel/integrator/path_state.h"
#include "kernel/integrator/surface_shader.h"
#include "kernel/integrator/vcm_types.h"
#include "kernel/light/sample.h"
#include "kernel/integrator/photon_time.h"
#include "kernel/shader_photon_path.h"
#include "kernel/sample/lcg.h"
#include "kernel/sample/photon_gaussian.h"
#include "kernel/sample/photon_spectrum.h"

CCL_NAMESPACE_BEGIN

/* A secondary ray on the shared edge of two triangles can hit the neighbour at t=0.
 * Offset the traversal origin to the outgoing side, also for transmission. The stored
 * vertex stays at the true intersection for area densities, visibility and merging. */
ccl_device_inline float3 vcm_scattered_origin(const ccl_private ShaderData *sd, const float3 D)
{
  return ray_offset(sd->P, dot(sd->Ng, D) < 0.0f ? -sd->Ng : sd->Ng);
}

/* Keep delta probability mass separate from continuous densities. For the mutually
 * reciprocal simple dielectric, Fresnel chooses the same discrete branch in reverse. */
ccl_device_inline bool vcm_scatter(KernelGlobals kg,
                                   ccl_private ShaderData *sd,
                                   const float3 random,
                                   ccl_global VCMVertex &v,
                                   ccl_private Spectrum *flux,
                                   const bool importance)
{
  const ccl_private ShaderClosure *sc = &sd->closure[0];
  float3 wo;
  v.eta = 1.0f;
  const float in_s = dot(sc->N, sd->wi), in_g = dot(sd->Ng, sd->wi);
  if (!(in_s > 0.0f && in_g > 0.0f)) {
    *flux = zero_spectrum();
    v.label = LABEL_NONE;
    v.pdf_forward = v.pdf_reverse = 0.0f;
    v.outgoing = zero_float3();
    return true;
  }
  if (sc->type == CLOSURE_BSDF_DIFFUSE_ID) {
    float pdf;
    sample_cos_hemisphere(sc->N, make_float2(random), &wo, &pdf);
    v.label = LABEL_DIFFUSE | LABEL_REFLECT;
    v.pdf_forward = pdf;
    v.pdf_reverse = pdf_cos_hemisphere(sc->N, sd->wi);
    *flux *= sc->weight;
  }
  else {
    const bool glass = sc->type == CLOSURE_BSDF_MICROFACET_GGX_GLASS_ID ||
                       sc->type == CLOSURE_BSDF_MICROFACET_BECKMANN_GLASS_ID;
    if (!glass && sc->type != CLOSURE_BSDF_MICROFACET_GGX_ID &&
        sc->type != CLOSURE_BSDF_MICROFACET_BECKMANN_ID) {
      return false;
    }
    const ccl_private MicrofacetBsdf *bsdf = (const ccl_private MicrofacetBsdf *)sc;
    if (bsdf->alpha_x != 0.0f || bsdf->alpha_y != 0.0f || bsdf->energy_scale != 1.0f) {
      return false;
    }
    if (glass && bsdf->fresnel_type != MicrofacetFresnel::DIELECTRIC) {
      if (bsdf->fresnel_type != MicrofacetFresnel::GENERALIZED_SCHLICK) {
        return false;
      }
      const ccl_private FresnelGeneralizedSchlick *f =
          (const ccl_private FresnelGeneralizedSchlick *)bsdf->fresnel;
      if (f->thin_film.thickness != 0.0f || f->exponent >= 0.0f ||
          !is_zero(f->tint.reflectance - f->tint.transmittance) ||
          !is_zero(f->f90 - one_spectrum()) ||
          reduce_max(fabs(spectrum_to_rgb(f->f0) - make_float3(F0_from_ior(bsdf->ior)))) > 1e-5f) {
        return false;
      }
    }
    const float cosine = dot(sc->N, sd->wi);
    float cos_t = 0.0f;
    const FresnelCoeff f = microfacet_fresnel(kg, bsdf, cosine, &cos_t);
    if (iszero(f)) {
      *flux = zero_spectrum();
      v.outgoing = zero_float3();
      v.pdf_forward = v.pdf_reverse = 0.0f;
      v.label = LABEL_NONE;
      return true;
    }
    const float reflect_probability = average(f.reflectance) / average(f.sum());
    const bool reflect = reflect_probability == 1.0f || random.z < reflect_probability;
    const float mass = reflect ? reflect_probability : 1.0f - reflect_probability;
    v.eta = reflect ? 1.0f : bsdf->ior;
    wo = reflect ? 2.0f * cosine * sc->N - sd->wi :
                   refract_angle(sd->wi, sc->N, cos_t, 1.0f / bsdf->ior);
    v.label = LABEL_SINGULAR | (reflect ? LABEL_REFLECT : LABEL_TRANSMIT);
    v.pdf_forward = v.pdf_reverse = mass;
    *flux *= sc->weight * (reflect ? f.reflectance : f.transmittance) / mass;
  }
  v.outgoing = wo;
  const float sign = (v.label & LABEL_TRANSMIT) ? -1.0f : 1.0f;
  const float out_g = sign * dot(sd->Ng, wo), out_s = sign * dot(sc->N, wo);
  if (!(out_g > 0.0f && out_s > 0.0f)) {
    /* A diffuse arrival still represents incident flux, even if this continuation dies. */
    *flux = zero_spectrum();
    v.pdf_forward = v.pdf_reverse = 0.0f;
  }
  else if (importance && sc->N != sd->Ng) {
    /* Adjoint shading-normal factor for light transport. Camera transport already
     * integrates against the shading cosine. Geometric area is shared by all strategies. */
    *flux *= (in_s * out_g) / (in_g * out_s);
  }
  return true;
}

/* Multiple hero wavelengths in each round. Eye and light use the same jitter within
 * a stratum, including when they touch the dispersive interface at different depths. */
ccl_device_inline float vcm_wavelength(const VCMBatch batch, const uint index)
{
  return photon_group_wavelength_random(photon_caustics_wavelength_random(batch.round, batch.seed),
                                        index % PHOTON_WAVELENGTH_GROUPS, PHOTON_WAVELENGTH_GROUPS);
}

ccl_device_inline float vcm_shutter_time(KernelGlobals kg, const VCMBatch &batch)
{
  return photon_caustics_time(kg, hash_uint2(batch.seed, batch.round));
}

ccl_device_inline bool vcm_triangle(int type)
{
  return type == PRIMITIVE_TRIANGLE || type == PRIMITIVE_MOTION_TRIANGLE;
}

ccl_device_inline bool vcm_point_source(const ccl_global KernelLight *light)
{
  return (light->type == LIGHT_POINT || light->type == LIGHT_SPOT) && light->spot.radius == 0.0f;
}

ccl_device_inline float vcm_point_intensity(KernelGlobals kg,
                                            const ccl_global KernelLight *light,
                                            const float3 direction)
{
  return light->spot.eval_fac * (light->type == LIGHT_SPOT ?
      spot_light_attenuation(&light->spot, spot_light_to_local(kg, light, direction)) : 1.0f);
}

/* Advance only after the actual light-side scattering event. At the next vertex the
 * shader adds the final segment from origin, just like the photon caustics integrator. */
ccl_device_inline void vcm_photon_bounce(PhotonShaderPath &path, const float3 P, const int label)
{
  photon_shader_path_advance(&path, distance(path.origin, P));
  path.origin = P;
  path.depth++;
  path.last_label = label;
  if (label & LABEL_REFLECT) {
    path.reflection_depth++;
    if (label & LABEL_DIFFUSE) {
      path.diffuse_depth++;
    }
    else {
      path.glossy_depth++;
    }
  }
  else if (label & LABEL_TRANSMIT) {
    path.transmission_depth++;
  }
}

ccl_device_inline bool vcm_eval_surface(KernelGlobals kg, IntegratorState state,
                                        ccl_private ShaderData *sd, const Ray *ray,
                                        const Intersection *hit, const float wavelength,
                                        const PhotonShaderPath *photon = nullptr)
{
  if (!vcm_triangle(hit->type)) {
    return false;
  }
  shader_setup_from_ray(kg, sd, ray, hit);
  sd->photon_path = photon;
  if (sd->shader_flag & (SD_HAS_EMISSION | SD_HAS_VOLUME)) {
    return false;
  }
#ifdef __SPECTRAL__
  sd->rand_wavelength = wavelength;
#endif
  surface_shader_eval<KERNEL_FEATURE_NODE_MASK_SURFACE & ~KERNEL_FEATURE_NODE_AOV>(
      kg, state, sd, nullptr,
      photon ? PATH_RAY_VISIBILITY_NONE : INTEGRATOR_STATE(state, path, visibility),
      photon ? 0 : INTEGRATOR_STATE(state, path, flag));
  if (sd->runtime_flag & (SR_CACHE_MISS | SR_BSSRDF | SR_RAY_PORTAL | SR_HOLDOUT)) {
    return false;
  }
  for (int i = 0; i < sd->num_closure; ++i) {
    if (!CLOSURE_IS_BSDF(sd->closure[i].type) || sd->closure[i].type == CLOSURE_BSDF_TRANSPARENT_ID) {
      return false;
    }
  }
  return true;
}

ccl_device_inline int vcm_native_bounce(KernelGlobals kg, ccl_private ShaderData *sd,
                                        float3 random, const bool importance, float3 *wo,
                                        Spectrum *beta, float *sample_pdf = nullptr)
{
  if (!sd->num_closure) {
    return LABEL_NONE;
  }
  const ShaderClosure *sc = surface_shader_bsdf_bssrdf_pick(sd, &random);
  float pdf, eta, roughness;
  float2 sampled_roughness;
  BsdfEval eval;
  const int label = surface_shader_bsdf_sample_closure(kg, sd, sc, random, &eval, wo, &pdf,
                                                      &sampled_roughness, &eta, roughness);
  if (!(pdf > 0.0f)) {
    return LABEL_NONE;
  }
  if (sample_pdf) {
    *sample_pdf = pdf;
  }
  *beta *= bsdf_eval_sum(&eval) / pdf;
  if (importance) {
    const float denom = dot(sd->Ng, sd->wi) * dot(sc->N, *wo);
    if (denom == 0.0f) {
      return LABEL_NONE;
    }
    *beta *= fabsf(dot(sc->N, sd->wi) * dot(sd->Ng, *wo) / denom);
  }
  else if (label & LABEL_TRANSMIT) {
    *beta /= sqr(eta);
  }
  return label;
}

ccl_device void vcm_trace_specular_prefix(KernelGlobals kg, IntegratorState state,
                                          const VCMBatch batch, VCMLightPath &path,
                                          VCMVertex *vertices, uint *rng, Ray ray, Spectrum power)
{
  path_state_init_integrator(kg, state, batch.round, *rng, power);
  INTEGRATOR_STATE_WRITE(state, path, flag) = 0;
  INTEGRATOR_STATE_WRITE(state, path, render_pixel_index) = 0;
  ray.tmax = FLT_MAX;
  ray.self.object = ray.self.light_object = OBJECT_NONE;
  ray.self.prim = ray.self.light_prim = PRIM_NONE;
  PhotonShaderPath photon = {};
  photon.origin = ray.P;
  bool spectral = false;
  for (uint k = 0; k < batch.max_depth && !is_zero(power); ++k) {
    Intersection hit = {};
    if (!scene_intersect(kg, &ray, PATH_RAY_VISIBILITY_ALL & ~PATH_RAY_VISIBILITY_SHADOW, &hit)) {
      path.status = VCM_PATH_ESCAPE;
      return;
    }
    ShaderData sd;
    if (!vcm_eval_surface(kg, state, &sd, &ray, &hit, vcm_wavelength(batch, path.index), &photon)) {
      if (vcm_triangle(hit.type) && (sd.runtime_flag & SR_CACHE_MISS)) {
        path.status = VCM_PATH_CACHE_MISS;
      }
      return;
    }
    spectral |= (sd.runtime_flag & SR_BSDF_HAS_DISPERSION) != 0;
    VCMVertex &v = vertices[k];
    v = {};
    v.position = sd.P;
    v.normal = sd.Ng;
    v.incoming = sd.wi;
    v.shading_normal = sd.N;
    v.flux = spectrum_to_rgb(power);
    v.object = sd.object;
    v.prim = sd.prim;
    v.shader = sd.shader & SHADER_MASK;
    v.u = hit.u;
    v.v = hit.v;
    float3 wo = zero_float3();
    const int label = vcm_native_bounce(kg, &sd, lcg_step_float3(rng), true, &wo, &power);
    v.label = (label & LABEL_SINGULAR) | (k && (sd.runtime_flag & SR_BSDF_HAS_EVAL) ?
                                          VCM_VERTEX_RECEIVER : 0) |
              (spectral ? VCM_VERTEX_SPECTRAL : 0);
    if (hit.type == PRIMITIVE_MOTION_TRIANGLE) {
      v.label |= VCM_VERTEX_MOTION_TRIANGLE;
    }
    v.outgoing = wo;
    /* A path without a post-specular receiver still counts in the emission budget.
     * The CPU only needs its source for next-event estimation. */
    if (v.label & VCM_VERTEX_RECEIVER) {
      path.count = k + 1;
    }
    if (!(label & LABEL_SINGULAR)) {
      path.status = VCM_PATH_ABSORBED;
      return;
    }
    vcm_photon_bounce(photon, sd.P, label);
    ray.D = wo;
    ray.P = vcm_scattered_origin(&sd, wo);
    ray.self.object = sd.object;
    ray.self.prim = sd.prim;
    path_state_next(kg, state, label, sd.runtime_flag);
    INTEGRATOR_STATE_WRITE(state, path, throughput) = power;
  }
  path.status = VCM_PATH_DEPTH_LIMIT;
}

ccl_device bool vcm_emitted_power(KernelGlobals kg, IntegratorState state, const VCMBatch batch,
                                  VCMLightPath &path, const Ray &ray, Spectrum &power)
{
  const auto &lamp = kernel_data_fetch(lights, path.light);
  if (light_sample_shader_eval_nee_constant(kg, lamp.shader_id, path.light, true, power)) {
    return true;
  }
  path_state_init_integrator(kg, state, batch.round, batch.seed, one_spectrum());
  INTEGRATOR_STATE_WRITE(state, path, render_pixel_index) = 0;
  PhotonShaderPath photon = {};
  photon.origin = ray.P;
  photon.emission = true;
  if (light_sample_shader_eval_forward(kg, state, path.light, ray.P, -ray.D, 0.0f, ray.time,
                                       power, &photon) == SHADER_EVAL_CACHE_MISS) {
    path.status = VCM_PATH_CACHE_MISS;
    return false;
  }
  return true;
}

ccl_device void vcm_emit_point(KernelGlobals kg, IntegratorState state, const VCMBatch batch,
                               ccl_global VCMLightPath &path, ccl_global VCMVertex *vertices,
                               const ccl_global KernelLight *light, uint *rng)
{
  Ray ray = {};
  ray.time = vcm_shutter_time(kg, batch);
  ray.P = light->co;
  const float2 uv = make_float2(lcg_step_float3(rng));
  float pdf = M_1_2PI_F * 0.5f;
  if (light->type == LIGHT_SPOT && light->spot.cos_half_larger_spread < 1.0f) {
    float cos_theta;
    ray.D = sample_uniform_cone(light->spot.dir, 1.0f-light->spot.cos_half_larger_spread,
                                uv, &cos_theta, &pdf);
  }
  else {
    ray.D = sample_uniform_sphere(uv);
  }
  if (!light->use_rigid_motion && batch.guide_light == path.light && batch.guide_fraction > 0.0f) {
    const float3 center = batch.guide_target;
    if (lcg_step_float(rng) < batch.guide_fraction) {
      const float3 xi = lcg_step_float3(rng);
      const float4 random = make_float4(xi, lcg_step_float(rng));
      ray.D = safe_normalize(center + batch.guide_sigma * photon_gaussian_sample(random));
      if (light->type == LIGHT_SPOT && light->spot.cos_half_larger_spread < 1.0f &&
          dot(light->spot.dir, ray.D) < light->spot.cos_half_larger_spread) {
        pdf = 0.0f;
      }
    }
    pdf = (1.0f - batch.guide_fraction) * pdf + batch.guide_fraction *
          photon_gaussian_direction_pdf(center, batch.guide_sigma, ray.D);
  }
  const float intensity = vcm_point_intensity(kg, light, ray.D);
  Transform world, center;
  if (light_motion_transform(kg, light, ray.time, &world, &center)) {
    ray.P = transform_point(&world, ray.P);
    ray.D = normalize(transform_direction(&world, ray.D));
  }
  path.source = ray.P;
  path.direction = ray.D;
  path.area_pdf = 1.0f; /* Discrete position mass. */
  path.direction_pdf = pdf;
  Spectrum power;
  if (!vcm_emitted_power(kg, state, batch, path, ray, power)) {
    return;
  }
  power *= intensity / (pdf * path.light_pdf);
  path.flux = spectrum_to_rgb(power);
  vcm_trace_specular_prefix(kg, state, batch, path, vertices, rng, ray, power);
}

ccl_device void integrator_vcm_emit(KernelGlobals kg,
                                    IntegratorState state,
                                    const VCMBatch batch,
                                    const uint lane,
                                    ccl_global VCMLightPath *paths,
                                    ccl_global VCMVertex *vertices)
{
  ccl_global VCMLightPath &path = paths[lane];
  path = {};
  path.index = batch.first + lane;
  path.round = batch.round;
  path.status = VCM_PATH_UNSUPPORTED;
  path.light = -1;
  uint rng = hash_uint3(batch.seed, batch.round, path.index);
  const int num_lights = kernel_data.integrator.num_lights - kernel_data.integrator.num_background_lights;
  if (!num_lights) {
    path.status = VCM_PATH_ESCAPE;
    return;
  }
  int selected = min(int(lcg_step_float(&rng) * num_lights), num_lights - 1);
  for (int i = 0; i < kernel_data.integrator.num_lights; ++i) {
    if (kernel_data_fetch(lights, i).type != LIGHT_BACKGROUND && selected-- == 0) {
      path.light = i;
      break;
    }
  }
  const ccl_global KernelLight *light = &kernel_data_fetch(lights, path.light);
  path.light_pdf = 1.0f / num_lights;
  if (vcm_point_source(light)) {
    vcm_emit_point(kg, state, batch, path, vertices + size_t(lane)*batch.max_depth, light, &rng);
    return;
  }
  if (light->type != LIGHT_AREA || light->area.tan_half_spread != FLT_MAX) {
    return;
  }
  const float3 u = float3(light->area.axis_u) * light->area.len_u;
  const float3 v = float3(light->area.axis_v) * light->area.len_v;
  const bool ellipse = area_light_is_ellipse(&light->area);
  const float area = len(cross(u, v)) * (ellipse ? M_PI_4_F : 1.0f);
  if (!(area > 0.0f)) {
    return;
  }
  const float2 uv = make_float2(lcg_step_float3(&rng));
  Ray ray = {};
  ray.P = float3(light->co) + (ellipse ? ellipse_sample(0.5f * u, 0.5f * v, uv) :
                                         rectangle_sample(0.5f * u, 0.5f * v, uv));
  ray.tmax = FLT_MAX;
  ray.time = vcm_shutter_time(kg, batch);
  ray.self.object = ray.self.light_object = OBJECT_NONE;
  ray.self.prim = ray.self.light_prim = PRIM_NONE;
  path.light_pdf = 1.0f / float(num_lights);
  path.area_pdf = 1.0f / area;
  float dir_pdf;
  sample_cos_hemisphere(light->area.dir, make_float2(lcg_step_float3(&rng)), &ray.D, &dir_pdf);
  Transform world, center;
  if (light_motion_transform(kg, light, ray.time, &world, &center)) {
    ray.P = transform_point(&world, ray.P);
    ray.D = normalize(transform_direction(&world, ray.D));
  }
  path.direction_pdf = dir_pdf;
  path.source = ray.P;
  path.direction = ray.D;
  if (!(dir_pdf > 0.0f)) {
    path.status = VCM_PATH_ABSORBED;
    return;
  }

  Spectrum flux;
  if (!vcm_emitted_power(kg, state, batch, path, ray, flux)) {
    return;
  }
  /* The Lambert emission cosine cancels the cosine factor in direction_pdf. */
  flux *= fabsf(light->area.invarea) / (path.light_pdf * path.area_pdf);
  path.flux = spectrum_to_rgb(flux);
  if (batch.native_bsdf) {
    vcm_trace_specular_prefix(kg, state, batch, path, vertices + size_t(lane) * batch.max_depth,
                              &rng, ray, flux);
    return;
  }
  path_state_init_integrator(kg, state, batch.round, rng, flux);
  INTEGRATOR_STATE_WRITE(state, path, flag) = 0;
  INTEGRATOR_STATE_WRITE(state, path, render_pixel_index) = 0;
  PhotonShaderPath photon = {};
  photon.origin = ray.P;
  for (uint depth = 0; depth < batch.max_depth; ++depth) {
    Intersection hit = {};
    if (!scene_intersect(kg, &ray, PATH_RAY_VISIBILITY_ALL & ~PATH_RAY_VISIBILITY_SHADOW, &hit)) {
      path.status = VCM_PATH_ESCAPE;
      return;
    }
    if (!vcm_triangle(hit.type)) {
      return;
    }
    ShaderData sd;
    shader_setup_from_ray(kg, &sd, &ray, &hit);
    sd.photon_path = &photon;
    if (sd.shader_flag & (SD_REQUIRES_WAVELENGTH | SD_HAS_EMISSION | SD_HAS_VOLUME))
    {
      return;
    }
    surface_shader_eval<KERNEL_FEATURE_NODE_MASK_SURFACE & ~KERNEL_FEATURE_NODE_AOV>(
        kg, state, &sd, nullptr, PATH_RAY_VISIBILITY_NONE, 0);
    if (sd.runtime_flag & SR_CACHE_MISS) {
      path.status = VCM_PATH_CACHE_MISS;
      return;
    }
    if (!sd.num_closure) {
      path.status = VCM_PATH_ABSORBED;
      return;
    }
    if (sd.num_closure != 1) {
      return;
    }
    ccl_global VCMVertex &vertex = vertices[size_t(lane) * batch.max_depth + depth];
    vertex.position = sd.P;
    vertex.normal = sd.Ng;
    vertex.shading_normal = sd.closure[0].N;
    vertex.incoming = sd.wi;
    vertex.flux = spectrum_to_rgb(flux);
    vertex.u = hit.u;
    vertex.v = hit.v;
    vertex.object = sd.object;
    vertex.prim = sd.prim;
    vertex.shader = sd.shader & SHADER_MASK;
    if (!vcm_scatter(kg, &sd, lcg_step_float3(&rng), vertex, &flux, true)) {
      return;
    }
    if (hit.type == PRIMITIVE_MOTION_TRIANGLE) {
      vertex.label |= VCM_VERTEX_MOTION_TRIANGLE;
    }
    ++path.count;
    if (is_zero(flux) || vertex.pdf_forward <= 0.0f) {
      path.status = VCM_PATH_ABSORBED;
      return;
    }
    vcm_photon_bounce(photon, sd.P, vertex.label);
    ray.D = vertex.outgoing;
    ray.P = vcm_scattered_origin(&sd, ray.D);
    ray.self.object = sd.object;
    ray.self.prim = sd.prim;
    path_state_next(kg, state, vertex.label, sd.runtime_flag);
    INTEGRATOR_STATE_WRITE(state, path, throughput) = flux;
  }
  path.status = VCM_PATH_DEPTH_LIMIT;
}

CCL_NAMESPACE_END
