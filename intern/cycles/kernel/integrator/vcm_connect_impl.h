/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: Apache-2.0 */

#pragma once

/* Shared by the CPU reference, hybrid replay and OptiX device estimator. The including
 * backend supplies Result (film sink) and MergeRound (light table and spatial query). */
#include "kernel/camera/camera.h"
#include "kernel/integrator/vcm_emit.h"

CCL_NAMESPACE_BEGIN
namespace vcm {

ccl_device bool vcm_finite(double x)
{
#ifdef __KERNEL_GPU__
  return isfinite(x);
#else
  return std::isfinite(x);
#endif
}

using RGB = VCMColor;
constexpr double pi = 3.14159265358979323846;

ccl_device RGB rgb(const float3 value)
{
  return {double(value.x), double(value.y), double(value.z)};
}

ccl_device RGB product(const RGB &a, const RGB &b, const double scale)
{
  return {a[0] * b[0] * scale, a[1] * b[1] * scale, a[2] * b[2] * scale};
}

using Node = VCMConnectionNode;

struct Edge {
  double length_squared, from_cos, to_cos, from_shading, to_shading;
};

ccl_device Edge geometry(const Node &a, const Node &b)
{
  const double x = double(b.p.x) - double(a.p.x);
  const double y = double(b.p.y) - double(a.p.y);
  const double z = double(b.p.z) - double(a.p.z);
  const double rr = x * x + y * y + z * z;
  if (!(rr > 0.0)) {
    return {};
  }
  const double inv_d = 1.0 / ::sqrt(rr);
  return {rr,
          (x * double(a.n.x) + y * double(a.n.y) + z * double(a.n.z)) * inv_d,
          -(x * double(b.n.x) + y * double(b.n.y) + z * double(b.n.z)) * inv_d,
          (x * double(a.ns.x) + y * double(a.ns.y) + z * double(a.ns.z)) * inv_d,
          -(x * double(b.ns.x) + y * double(b.ns.y) + z * double(b.ns.z)) * inv_d};
}

struct Camera {
  Node endpoint;
  Transform world_to_camera;
  float2 plane_origin, plane_extent;
  float time;
  double plane_area;
  double merge_eta = 0.0;
};

ccl_device Camera pinhole(KernelGlobals kg, const VCMBatch &batch)
{
  const KernelCamera &cam = kernel_data.cam;
  Camera eye = {};
  eye.time = vcm_shutter_time(kg, batch);
  Transform to_world = cam.cameratoworld;
  if (cam.num_motion_steps) {
    transform_motion_array_interpolate(&to_world, kernel_data_array(camera_motion),
                                       cam.num_motion_steps, eye.time, motion_use_linear_quaternion());
  }
  eye.world_to_camera = cam.num_motion_steps ? transform_inverse(to_world) : cam.worldtocamera;
  eye.endpoint.p = transform_point(&to_world, zero_float3());
  eye.endpoint.n = normalize(transform_direction(&to_world, make_float3(0, 0, 1)));
  eye.endpoint.ns = eye.endpoint.n;
  auto plane_point = [&](const float x, const float y) {
    const float3 raster = make_float3(x, y, 0);
    float3 p = transform_perspective(&cam.rastertocamera, raster);
    if (cam.have_perspective_motion) {
      const ProjectionTransform other = eye.time < 0.5f ? cam.perspective_pre : cam.perspective_post;
      p = interp(p, transform_perspective(&other, raster), fabsf(2.0f * eye.time - 1.0f));
    }
    return p / p.z;
  };
  const float3 origin = plane_point(0, 0);
  eye.plane_origin = make_float2(origin);
  eye.plane_extent = make_float2(plane_point(float(cam.width), float(cam.height)) - origin);
  eye.plane_area = fabsf(eye.plane_extent.x * eye.plane_extent.y);
  return eye;
}

ccl_device double camera_pdf(const double cos_camera, const double plane_area)
{
  return cos_camera > 0.0 ? 1.0 / (plane_area * cos_camera * cos_camera * cos_camera) : 0.0;
}

ccl_device float2 vcm_raster(const KernelCamera &cam, const Camera &eye, const float3 local)
{
  return make_float2((local.x / local.z - eye.plane_origin.x) * cam.width / eye.plane_extent.x,
                     (local.y / local.z - eye.plane_origin.y) * cam.height / eye.plane_extent.y);
}

ccl_device Ray initial_ray(float time)
{
  Ray ray = {};
  ray.time = time;
  ray.tmax = FLT_MAX;
  ray.self.object = ray.self.light_object = OBJECT_NONE;
  ray.self.prim = ray.self.light_prim = PRIM_NONE;
  return ray;
}

/* Intersect the open segment. Use original endpoints for all PDF and geometry calculations. */
ccl_device bool visible(KernelGlobals kg, const Node &a, const Node &b, const bool to_camera, const Camera &eye)
{
  Ray ray = initial_ray(eye.time);
  const float3 p = a.prim == PRIM_NONE ? a.p : ray_offset(a.p, a.n);
  float3 q = b.prim == PRIM_NONE ? b.p : ray_offset(b.p, b.n);
  if (to_camera) {
    const float3 local = transform_point(&eye.world_to_camera, p);
    q += (p - q) * (kernel_data.cam.nearclip / local.z);
  }
  ray.P = p;
  ray.D = safe_normalize_len(q - p, &ray.tmax);
  ray.self.object = a.object;
  ray.self.prim = a.prim;
  ray.self.light_object = b.object;
  ray.self.light_prim = b.prim;
  return ray.tmax > 0 && !scene_intersect_shadow(kg, &ray, PATH_RAY_VISIBILITY_SHADOW);
}

ccl_device bool replay_surface(KernelGlobals kg,
                               IntegratorState state,
                               ShaderData &sd,
                               const Ray &ray,
                               const Intersection &hit,
                               const PhotonShaderPath *photon = nullptr)
{
  if (!vcm_triangle(hit.type)) {
    return false;
  }
  sd.runtime_flag = 0;
  shader_setup_from_ray(kg, &sd, &ray, &hit);
  sd.photon_path = photon;
  if (sd.shader_flag & (SD_REQUIRES_WAVELENGTH | SD_HAS_VOLUME | SD_HAS_EMISSION))
  {
    return false;
  }
  surface_shader_eval<KERNEL_FEATURE_NODE_MASK_SURFACE & ~KERNEL_FEATURE_NODE_AOV>(
      kg, state, &sd, nullptr,
      photon ? PATH_RAY_VISIBILITY_NONE : INTEGRATOR_STATE(state, path, visibility),
      photon ? 0 : INTEGRATOR_STATE(state, path, flag));
  return !(sd.runtime_flag & SR_CACHE_MISS) && sd.num_closure <= 1;
}

ccl_device bool vcm_area_intersect(KernelGlobals kg, const KernelLight &lamp, const Ray &ray, float *t)
{
  Transform world, center;
  Ray local = ray;
  if (light_motion_transform(kg, &lamp, ray.time, &world, &center)) {
    local.P = transform_point(&center, ray.P);
    local.D = transform_direction(&center, ray.D);
  }
  return area_light_intersect(&lamp, &local, t);
}

ccl_device bool light_endpoint(
    KernelGlobals kg, int index, float3 position, Node &node, double &pdf, float time,
    const Spectrum *sampled_power = nullptr)
{
  const KernelLight &light = kernel_data_fetch(lights, index);
  Spectrum power;
  if (light.type != LIGHT_AREA || light.area.tan_half_spread != FLT_MAX ||
      (!sampled_power && !light_sample_shader_eval_nee_constant(kg, light.shader_id, index, true, power)))
  {
    return false;
  }
  const double area = double(len(cross(float3(light.area.axis_u) * light.area.len_u,
                                       float3(light.area.axis_v) * light.area.len_v))) *
                      (area_light_is_ellipse(&light.area) ? pi / 4.0 : 1.0);
  if (!(area > 0.0)) {
    return false;
  }
  pdf = 1.0 / (area *
               (kernel_data.integrator.num_lights - kernel_data.integrator.num_background_lights));
  node.source_light = index;
  node.p = position;
  Transform world, center;
  node.n = node.ns = light_motion_transform(kg, &light, time, &world, &center) ?
                         normalize(transform_direction(&world, light.area.dir)) : float3(light.area.dir);
  node.beta = rgb(spectrum_to_rgb(sampled_power ? *sampled_power : power));
  for (int i = 0; i < 3; ++i) {
    node.beta[i] *= double(fabsf(light.area.invarea)) / pi;
  }
  return true;
}

/* Both strategies read the path in light-to-camera order. VM can use any photon's
 * stored prefix directly; never materialize or reverse a path per neighbor. */
struct NodeList {
  Node *nodes;
  int count;
  ccl_device_inline_method int size() const
  {
    return count;
  }
  ccl_device_inline_method Node &operator[](int i) const
  {
    return nodes[i];
  }
  ccl_device_inline_method Node &back() const
  {
    return nodes[count - 1];
  }
  ccl_device_inline_method void push_back(const Node &n)
  {
    nodes[count++] = n;
  }
};

struct JoinedPath {
  const Node *source;
  const VCMVertex *prefix;
  int light_count;
  NodeList eye;
  ccl_device_inline_method int size() const
  {
    return light_count + eye.count;
  }
  ccl_device_inline_method Node operator[](int i) const
  {
    if (i >= light_count) {
      return eye.nodes[size() - 1 - i];
    }
    if (!prefix || !i) {
      return source[i];
    }
    const VCMVertex &v = prefix[i - 1];
    Node n = {};
    n.p = v.position;
    n.n = v.normal;
    n.ns = v.shading_normal;
    n.delta = (v.label & LABEL_SINGULAR) != 0;
    return n;
  }
};

ccl_device_inline JoinedPath joined(const NodeList &light, int count, const NodeList &eye)
{
  return {light.nodes, nullptr, count, eye};
}

/* Ordered techniques: cut s at 2*s, merge vertex j at 2*j+1. A merge samples
 * vertex j from BOTH sides. Its extra density factor is the disk area times the
 * number of emission attempts. Only compare strategies that traverse each delta
 * in the same direction; select that family from the full path, independently of
 * the sampled strategy. This also admits a merge flanked by two singular events. */
ccl_device double strategy_weight(const JoinedPath &path,
                                  const int selected,
                                  const double source_pdf,
                                  const double plane_area,
                                  const double eta,
                                  Result &result)
{
  const int n = int(path.size());
  if (n == 2) {
    return selected == 0 ? 1.0 : 0.0;
  }
  auto legal = [&](int pos) {
    const int i = pos / 2;
    if (pos & 1) {
      /* The grid starts after the first light-side hit. All strategies must use
       * this same path-space rule, including BSDF hits, NEE and camera splats. */
      return eta > 0.0 && i > 1 && i < n - 1 && !path[i].delta;
    }
    return i == 0 || (!path[i - 1].delta && !path[i].delta);
  };
  int last = 2 * (n - 1);
  while (!legal(last)) {
    --last;
  }
  int first = 0;
  for (int i = 1; 2 * i + 1 < last; ++i) {
    if (path[i].delta) {
      first = 2 * i + 2;
    }
  }
  while (!legal(first)) {
    ++first;
  }
  if (selected < first || selected > last || !legal(selected)) {
    result.count_delta();
    return 0.0;
  }
  /* Remove common prefix and suffix densities before evaluating any ratios: this
   * cancels all shared singular conditional masses without converting them to area. */
  double logs[2 * (64 + 1) + 1];
  const int num = last - first + 1;
  for (int pos = first; pos <= last; ++pos) {
    logs[pos - first] = !legal(pos) ? -INFINITY : (pos & 1) ? ::log(eta) : 0.0;
  }
  /* Forward and reverse cumulative products in log space. Adding a factor at its
   * own cut and walking outwards also handles an exact zero without a 0/0 ratio. */
#ifndef __KERNEL_GPU__
  std::vector<double> reference;
  if (result.check_weights) {
    reference.assign(logs, logs + num);
  }
#endif
  const double zero = -INFINITY;
  double prefix = 0.0, suffix = 0.0;
  for (int k = 0; k <= last - first; ++k) {
    const int left = first + k, right = last - k;
    if (left > first && (left & 1)) {
      const int i = left / 2;
      double pdf = source_pdf;
      if (i) {
        const Edge e = geometry(path[i - 1], path[i]);
        pdf = e.length_squared > 0 && e.from_cos > 0 && e.from_shading > 0 ?
                  e.from_shading * ::fabs(e.to_cos) / (pi * e.length_squared) :
                  0;
      }
      const double factor = pdf > 0 ? ::log(pdf) : zero;
      prefix += factor;
#ifndef __KERNEL_GPU__
      if (result.check_weights) {
        for (int p = left; p <= last; ++p) {
          reference[p - first] += factor;
        }
      }
#endif
    }
    if (right < last && (right & 1)) {
      const int i = right / 2;
      const Edge e = geometry(path[i + 1], path[i]);
      const double dir = i == n - 2 ? camera_pdf(e.from_cos, plane_area) :
                                      (e.from_cos > 0 ? ::fmax(e.from_shading, 0.0) / pi : 0.0);
      const double pdf = e.length_squared > 0 ? dir * ::fabs(e.to_cos) / e.length_squared : 0;
      const double factor = pdf > 0 ? ::log(pdf) : zero;
      suffix += factor;
#ifndef __KERNEL_GPU__
      if (result.check_weights) {
        for (int p = first; p <= right; ++p) {
          reference[p - first] += factor;
        }
      }
#endif
    }
    logs[left - first] += prefix;
    logs[right - first] += suffix;
  }
  double peak = logs[0];
  for (int k = 1; k < num; ++k) {
    peak = ::fmax(peak, logs[k]);
  }
  if (!vcm_finite(peak)) {
    return NAN;
  }
  double sum = 0.0;
  for (int k = 0; k < num; ++k) {
    logs[k] = ::exp(logs[k] - peak);
    sum += logs[k];
  }
#ifndef __KERNEL_GPU__
  double check = 0.0;
  for (int k = 0; k < num; ++k) {
    check += logs[k] / sum;
  }
  result.max_weight_error = std::max(result.max_weight_error, ::fabs(check - 1.0));
  if (result.check_weights) {
    const double ref_peak = *std::max_element(reference.begin(), reference.end());
    double ref_sum = 0;
    for (double &v : reference) {
      v = ::exp(v - ref_peak);
      ref_sum += v;
    }
    for (size_t i = 0; i < size_t(num); ++i) {
      const double error = ::fabs(logs[i] / sum - reference[i] / ref_sum);
      if (!(error <= 1e-12)) {
        return NAN;
      }
      result.max_weight_error = std::max(result.max_weight_error, error);
    }
  }
#endif
  return logs[selected - first] / sum;
}

ccl_device bool record(const JoinedPath &path,
                       int split,
                       double pdf,
                       const Camera &eye,
                       size_t pixel,
                       VCMTechnique technique,
                       const RGB &value,
                       Result &result)
{
  const double weight = strategy_weight(
      path, 2 * split, pdf, eye.plane_area, eye.merge_eta, result);
  if (!vcm_finite(weight)) {
    result.set_failure("invalid connection strategy density");
    return false;
  }
  if (weight > 0.0) {
    RGB weighted = value;
    for (int i = 0; i < 3; ++i) {
      weighted[i] *= weight;
      if (!vcm_finite(weighted[i])) {
        result.set_failure("non-finite connection contribution");
        return false;
      }
    }
    result.add(pixel, technique, weighted);
    result.add_source(pixel, path.source->source_light, weighted);
    if (technique == VCM_CAMERA_HIT && path.size() == 2) {
      result.add_primary(pixel, false, weighted);
    }
  }
  return true;
}

/* Gather incoming power on the camera receiver. Keep the photon prefix for alternative
 * densities, and evaluate the diffuse receiver exactly once at the contracted vertex. */
ccl_device bool merge_surface(KernelGlobals kg,
                              const MergeRound &round,
                              const NodeList &eyes,
                              const Camera &eye,
                              const size_t pixel,
                              const uint max_depth,
                              Result &result)
{
  const Node &c = eyes.back();
  const RGB receiver = product(c.beta, c.albedo, (1.0 / pi) / round.eta);
  RGB sum = {};
  const bool ok = round.visit(c.p, c.n, [&](uint64_t id, const packed_float3 &flux) {
    const uint index = uint(id >> 32), depth = uint(id);
    if (depth + eyes.size() - 1 > max_depth) {
      return true;
    }
    const VCMVertex *prefix = round.prefix(index);
    const VCMVertex &hit = prefix[depth];
    if (c.object != hit.object || dot(c.n, float3(hit.normal)) <= 0.0f) {
      return true;
    }
    /* Photon density is per geometric area. Convert its incident projected
     * flux to the receiver's shading frame before evaluating the diffuse BSDF. */
    const float3 incoming = hit.incoming;
    const double projected_s = dot(c.ns, incoming), projected_g = dot(c.n, incoming);
    if (!(projected_s > 0 && projected_g > 0)) {
      return true;
    }
    const VCMLightPath &light = round.paths[index];
    Node source = {};
    double pdf;
    if (!light_endpoint(kg, light.light, light.source, source, pdf, eye.time)) {
      return false;
    }
    const JoinedPath path = {&source, prefix, int(depth + 1), eyes};
    const double weight = strategy_weight(
        path, 2 * int(depth + 1) + 1, pdf, eye.plane_area, round.eta, result);
    if (!vcm_finite(weight)) {
      return false;
    }
    const RGB term = product(receiver, rgb(flux), weight * (projected_s / projected_g));
    result.add_source(pixel, source.source_light, term);
    for (int i = 0; i < 3; ++i) {
      sum[i] += term[i];
    }
    return true;
  });
  if (!ok || !vcm_finite(sum[0]) || !vcm_finite(sum[1]) || !vcm_finite(sum[2])) {
    return false;
  }
  if (sum[0] || sum[1] || sum[2]) {
    result.add(pixel, VCM_SURFACE_MERGE, sum);
  }
  return true;
}

/* Only the light family is partitioned. At every camera vertex NEE estimates paths
 * whose first light-side scattering is continuous. A point source followed by a
 * singular chain has zero camera-hit probability; its first continuous receiver is
 * estimated by VM, and, when camera-visible, also by a light-to-camera connection. */
ccl_device Spectrum vcm_eval_direction(KernelGlobals kg,
                                       IntegratorState state,
                                       ShaderData &sd,
                                       const float3 direction)
{
  BsdfEval eval;
  float roughness;
  surface_shader_bsdf_eval(kg, state, &sd, direction, &eval, 0, roughness);
  return bsdf_eval_sum(&eval);
}

ccl_device Spectrum vcm_spectral_weight(KernelGlobals kg,
                                        const bool spectral,
                                        const float wavelength)
{
#ifdef __SPECTRAL__
  if (spectral) {
    return dispersion_throughput_weight(kg, wavelength);
  }
#endif
  return one_spectrum();
}

ccl_device bool point_record(Result &result,
                             const size_t pixel,
                             const VCMTechnique technique,
                             const Spectrum value,
                             const int source_light = -2, const bool primary = false)
{
  const float3 color = spectrum_to_rgb(value);
  if (!isfinite_safe(color)) {
    result.set_failure("non-finite point-source contribution");
    return false;
  }
  if (!is_zero(value)) {
    result.add(pixel, technique, rgb(color));
    result.add_source(pixel, source_light, rgb(color));
    if (primary) {
      result.add_primary(pixel, source_light == -1, rgb(color));
    }
  }
  return true;
}

ccl_device bool record_world(KernelGlobals kg,
                             IntegratorState state,
                             const Ray &ray,
                             const Spectrum throughput,
                             const size_t pixel,
                             Result &result)
{
  const int shader = kernel_data.background.surface_shader;
  const auto visibility = INTEGRATOR_STATE(state, path, visibility);
  const auto flags = INTEGRATOR_STATE(state, path, flag);
  if (!is_light_shader_visible_to_path(shader, visibility, flags)) {
    return true;
  }
  Spectrum value;
  if (!surface_shader_constant_emission(kg, shader, &value)) {
    ShaderDataTinyStorage storage;
    ShaderData *sd = AS_SHADER_DATA(&storage);
    shader_setup_from_background(kg, sd, ray.P, ray.D, ray.dD, ray.time);
    surface_shader_eval<KERNEL_FEATURE_NODE_MASK_SURFACE_BACKGROUND & ~KERNEL_FEATURE_NODE_AOV>(
        kg, state, sd, nullptr, visibility, flags | PATH_RAY_EMISSION);
    if (sd->runtime_flag & SR_CACHE_MISS) {
      result.cache_miss = true;
      result.set_failure("background shader texture unavailable");
      return false;
    }
    value = surface_shader_background(sd);
  }
  return point_record(result, pixel, VCM_CAMERA_HIT, throughput * value, -1,
                      INTEGRATOR_STATE(state, path, bounce) == 0);
}

ccl_device void connect_native_surface(KernelGlobals kg,
                                     IntegratorState state,
                                     const VCMBatch &batch,
                                     const VCMLightPath &light,
                                     const VCMVertex *vertices,
                                     const MergeRound *round,
                                     Result &result)
{
  const KernelCamera &cam = kernel_data.cam;
  const Camera eye = pinhole(kg, batch);
  uint rng = hash_uint3(batch.seed ^ 0x79dedea3u, batch.round, light.index);
  const float3 jitter = lcg_step_float3(&rng);
  const float x = fminf(jitter.x * cam.width, nextafterf(float(cam.width), 0.0f));
  const float y = fminf(jitter.y * cam.height, nextafterf(float(cam.height), 0.0f));
  const size_t pixel = size_t(y) * size_t(cam.width) + size_t(x);
  const uint eye_group = hash_uint2(light.index, batch.round ^ batch.seed) % 8;
  const float wavelength = vcm_wavelength(batch, eye_group);
  path_state_init_integrator(kg, state, batch.round, rng, one_spectrum());
  INTEGRATOR_STATE_WRITE(state, path, render_pixel_index) = 0;
  bool spectral = false;

  /* Evaluate the receiver from the camera side for both strategies. The stored incident
   * flux is per geometric area, while the Cycles BSDF evaluation includes its cosine. */
  for (uint k = 1; k < light.count && k + 1 <= batch.max_depth; ++k) {
    const VCMVertex &v = vertices[k];
    if (!(v.label & VCM_VERTEX_RECEIVER)) {
      continue;
    }
    Node node = {};
    node.p = v.position;
    node.n = v.normal;
    node.object = v.object;
    node.prim = v.prim;
    const float3 local = transform_point(&eye.world_to_camera, node.p);
    if (local.z <= cam.nearclip || local.z - cam.nearclip >= cam.cliplength) {
      continue;
    }
    const float2 raster = vcm_raster(cam, eye, local);
    if (!(raster.x >= 0 && raster.x < cam.width && raster.y >= 0 && raster.y < cam.height)) {
      continue;
    }
    Ray incident = initial_ray(eye.time);
    float distance;
    incident.D = safe_normalize_len(node.p - eye.endpoint.p, &distance);
    incident.P = node.p;
    const double projection = ::fabs(dot(node.n, incident.D));
    const double cosine = dot(incident.D, eye.endpoint.n);
    const double incident_cos = ::fabs(dot(node.n, float3(v.incoming)));
    if (!(cosine > 0 && projection > 0 && incident_cos > 0 && distance > 0)) {
      continue;
    }
    if (dot(node.n, incident.D) > 0) {
      node.n = -node.n;
    }
    result.count_shadow();
    if (!visible(kg, node, eye.endpoint, true, eye)) {
      result.count_blocked();
      continue;
    }
    const float lambda = vcm_wavelength(batch, light.index);
    const Intersection hit = {0.0f, v.u, v.v, v.prim, v.object,
                              (v.label & VCM_VERTEX_MOTION_TRIANGLE) ? PRIMITIVE_MOTION_TRIANGLE : PRIMITIVE_TRIANGLE};
    ShaderData sd;
    sd.runtime_flag = 0;
    if (!vcm_eval_surface(kg, state, &sd, &incident, &hit, lambda)) {
      result.cache_miss = vcm_triangle(hit.type) && (sd.runtime_flag & SR_CACHE_MISS);
      result.set_failure("unsupported light-to-camera receiver");
      return;
    }
    const float scale = float((round ? 0.5 : 1.0) * camera_pdf(cosine, eye.plane_area) *
                              projection / (distance * double(distance) * incident_cos));
    const Spectrum value = rgb_to_spectrum(float3(v.flux)) *
                           vcm_eval_direction(kg, state, sd, v.incoming) * scale *
                           vcm_spectral_weight(kg, v.label & VCM_VERTEX_SPECTRAL, lambda);
    if (!point_record(result,
                      size_t(raster.y) * size_t(cam.width) + size_t(raster.x),
                      VCM_CAMERA_SPLAT,
                      value, light.light))
    {
      return;
    }
  }

  Ray ray = initial_ray(eye.time);
  camera_sample_perspective(kg, make_float2(x, y), zero_float2(), &ray);
  Spectrum beta = one_spectrum();
  bool has_continuous_event = false;
  bool last_singular = false;
  float last_bsdf_pdf = 0.0f;
  float3 previous_position = ray.P;
  for (uint depth = 0; depth <= batch.max_depth; ++depth) {
    Intersection hit = {};
    const bool found = scene_intersect(kg, &ray, path_state_ray_visibility(state), &hit);
    if (batch.native_bsdf && !(last_singular && has_continuous_event)) {
      Ray segment = ray;
      if (found) {
        segment.tmax = hit.t;
      }
      for (int i = 0; i < kernel_data.integrator.num_lights; ++i) {
        const KernelLight &lamp = kernel_data_fetch(lights, i);
        float t;
        if (lamp.type != LIGHT_AREA ||
            (depth == 0 && (lamp.shader_id & SHADER_EXCLUDE_CAMERA)) ||
            !vcm_area_intersect(kg, lamp, segment, &t)) {
          continue;
        }
        Node source = {};
        double p_area;
        Spectrum power;
        if (light_sample_shader_eval_forward(kg, state, i, ray.P, ray.D, t, ray.time, power) ==
            SHADER_EVAL_CACHE_MISS) {
          result.cache_miss = true;
          result.set_failure("area light texture cache miss");
          return;
        }
        if (!light_endpoint(kg, i, ray.P + t * ray.D, source, p_area, eye.time, &power)) {
          return;
        }
        const float dist2 = len_squared(source.p - previous_position);
        const float cos_light = dot(source.n, -ray.D);
        const float light_pdf = float(p_area) * dist2 / cos_light;
        const float weight = !depth || last_singular ? 1.0f :
                             power_heuristic(last_bsdf_pdf, light_pdf);
        const Spectrum radiance = rgb_to_spectrum(make_float3(
            float(source.beta[0]), float(source.beta[1]), float(source.beta[2])));
        if (!point_record(result, pixel, VCM_CAMERA_HIT, beta * radiance * weight *
                          vcm_spectral_weight(kg, spectral, wavelength), i, depth == 0)) {
          return;
        }
      }
    }
    if (!found) {
      if (!record_world(
              kg, state, ray, beta * vcm_spectral_weight(kg, spectral, wavelength), pixel, result))
      {
        return;
      }
      break;
    }
    if (depth == batch.max_depth) {
      break;
    }
    ShaderData sd;
    sd.runtime_flag = 0;
    if (!vcm_eval_surface(kg, state, &sd, &ray, &hit, wavelength)) {
      result.cache_miss = vcm_triangle(hit.type) && (sd.runtime_flag & SR_CACHE_MISS);
      result.set_failure("unsupported point-source camera surface");
      result.object = hit.object;
      result.prim = hit.prim;
      return;
    }
    spectral |= (sd.runtime_flag & SR_BSDF_HAS_DISPERSION) != 0;
    if (sd.runtime_flag & SR_BSDF_HAS_EVAL) {
      if (light.light >= 0) {
        const KernelLight &lamp = kernel_data_fetch(lights, light.light);
        float distance;
        const bool area = lamp.type == LIGHT_AREA;
        const float3 position = light.source;
        const float3 wi = safe_normalize_len(position - sd.P, &distance);
        BsdfEval bsdf_eval;
        float roughness;
        const float bsdf_pdf = surface_shader_bsdf_eval(kg, state, &sd, wi, &bsdf_eval, 0, roughness);
        const Spectrum eval = bsdf_eval_sum(&bsdf_eval);
        const float3 outgoing = light_motion_direction_to_center(kg, &lamp, ray.time, -wi);
        const float cos_light = area ? fmaxf(dot(lamp.area.dir, outgoing), 0.0f) : 1.0f;
        const float light_pdf = area && cos_light > 0 ?
            light.light_pdf * light.area_pdf * distance * distance / cos_light : 0.0f;
        const float intensity = area ?
            (light_pdf > 0 ? fabsf(lamp.area.invarea) * M_1_PI_F *
                             power_heuristic(light_pdf, bsdf_pdf) / light_pdf : 0.0f) :
            (distance > 0 ? vcm_point_intensity(kg, &lamp, outgoing) /
                            (distance * distance * light.light_pdf) : 0.0f);
        if (intensity > 0 && !is_zero(eval)) {
          Node c = {}, l = {};
          c.p = sd.P;
          c.n = dot(sd.Ng, wi) >= 0 ? sd.Ng : -sd.Ng;
          c.prim = hit.prim;
          c.object = hit.object;
          l.p = position;
          result.count_shadow();
          if (visible(kg, c, l, false, eye)) {
            Spectrum power;
            if (light_sample_shader_eval_forward(kg, state, light.light, position, wi, 0.0f,
                                                 ray.time, power) == SHADER_EVAL_CACHE_MISS) {
              result.cache_miss = true;
              result.set_failure("sampled light texture cache miss");
              return;
            }
            if (!point_record(result,
                              pixel,
                              VCM_NEXT_EVENT,
                              beta * eval * power * vcm_spectral_weight(kg, spectral, wavelength) *
                                  intensity, light.light))
            {
              return;
            }
          }
          else {
            result.count_blocked();
          }
        }
      }
      if (round) {
        Spectrum sum = zero_spectrum();
        const bool ok = round->visit(
            sd.P,
            sd.Ng,
            [&](uint64_t id, const packed_float3 &flux) {
              const uint index = uint(id >> 32), ld = uint(id);
              if (ld + depth + 1 > batch.max_depth) {
                return true;
              }
              const VCMVertex &v = round->prefix(index)[ld];
              if (v.object != hit.object || ::fabs(dot(sd.Ng, float3(v.normal))) < 0.5f) {
                return true;
              }
              const bool light_spectral = (v.label & VCM_VERTEX_SPECTRAL) != 0;
              const uint group = index % 8;
              if (spectral && light_spectral && group != eye_group) {
                return true;
              }
              const float3 wi = v.incoming;
              const float cos_in = fabsf(dot(sd.Ng, wi));
              if (cos_in > 0) {
                const Spectrum term = rgb_to_spectrum(float3(flux)) * vcm_eval_direction(kg, state, sd, wi) *
                       vcm_spectral_weight(kg,
                                           spectral || light_spectral,
                                           spectral ? wavelength : vcm_wavelength(batch, group)) *
                       ((spectral && light_spectral ? 8.0f : 1.0f) / cos_in);
                sum += term;
                if (result.route_sources) {
                  const int source_light = round->paths[index].light;
                  result.add_source(pixel, source_light, rgb(spectrum_to_rgb(
                      beta * term * float((depth ? 1.0 : 0.5) / round->eta))));
                }
              }
              return true;
            },
            true);
        if (!ok || !point_record(result,
                                 pixel,
                                 VCM_SURFACE_MERGE,
                                 beta * sum * float((depth ? 1.0 : 0.5) / round->eta)))
        {
          return;
        }
      }
    }
    float3 wo;
    const int label = vcm_native_bounce(
        kg, &sd, lcg_step_float3(&rng), false, &wo, &beta, &last_bsdf_pdf);
    previous_position = sd.P;
    last_singular = (label & LABEL_SINGULAR) != 0;
    has_continuous_event |= !last_singular;
    if (!label || is_zero(beta)) {
      break;
    }
    ray.P = vcm_scattered_origin(&sd, wo);
    ray.D = wo;
    ray.tmin = 0;
    ray.tmax = FLT_MAX;
    ray.self.object = hit.object;
    ray.self.prim = hit.prim;
    path_state_next(kg, state, label, sd.runtime_flag);
    INTEGRATOR_STATE_WRITE(state, path, throughput) = beta;
  }
  result.set_failure(nullptr);
}

ccl_device void connect_path(KernelGlobals kg,
                             IntegratorState state,
                             Node *scratch,
                             const VCMBatch &batch,
                             const VCMLightPath &light,
                             const VCMVertex *vertices,
                             const MergeRound *merging,
                             Result &result)
{
  if (batch.native_bsdf || light.light < 0 || vcm_point_source(&kernel_data_fetch(lights, light.light))) {
    connect_native_surface(kg, state, batch, light, vertices, merging, result);
    return;
  }
  const KernelCamera &cam = kernel_data.cam;
  Camera eye = pinhole(kg, batch);
  eye.merge_eta = merging ? merging->eta : 0.0;
  uint rng = hash_uint3(batch.seed ^ 0x79dedea3u, light.round, light.index);
  const float3 random = lcg_step_float3(&rng);
  const float x = fminf(random.x * cam.width, nextafterf(float(cam.width), 0.0f));
  const float y = fminf(random.y * cam.height, nextafterf(float(cam.height), 0.0f));
  const size_t pixel = size_t(y) * size_t(cam.width) + size_t(x);
  Ray ray = initial_ray(eye.time);
  camera_sample_perspective(kg, make_float2(x, y), zero_float2(), &ray);
  path_state_init_integrator(kg, state, batch.round, rng, one_spectrum());
  INTEGRATOR_STATE_WRITE(state, path, render_pixel_index) = 0;
  NodeList lights = {scratch, 1}, eyes = {scratch + batch.max_depth + 1, 1};
  lights[0] = {};
  eyes[0] = eye.endpoint;
  double lamp_pdf;
  if (!light_endpoint(kg, light.light, light.source, lights[0], lamp_pdf, eye.time)) {
    return;
  }
  PhotonShaderPath photon = {};
  photon.origin = light.source;
  for (uint k = 0; k < light.count; ++k) {
    const VCMVertex &v = vertices[k];
    if (!v.label) {
      break;
    }
    Node l = {};
    l.p = v.position;
    l.n = v.normal;
    l.ns = v.shading_normal;
    l.object = v.object;
    l.prim = v.prim;
    l.beta = rgb(v.flux);
    l.delta = (v.label & LABEL_SINGULAR) != 0;
    if (!l.delta) {
      Ray incident = initial_ray(eye.time);
      incident.P = l.p;
      incident.D = -float3(v.incoming);
      const Intersection hit = {0.0f, v.u, v.v, v.prim, v.object,
                              (v.label & VCM_VERTEX_MOTION_TRIANGLE) ? PRIMITIVE_MOTION_TRIANGLE : PRIMITIVE_TRIANGLE};
      ShaderData sd;
      sd.runtime_flag = 0;
      if (!replay_surface(kg, state, sd, incident, hit, &photon) || sd.num_closure != 1 ||
          sd.closure[0].type != CLOSURE_BSDF_DIFFUSE_ID || (sd.shader & SHADER_MASK) != v.shader)
      {
        result.cache_miss = vcm_triangle(hit.type) && (sd.runtime_flag & SR_CACHE_MISS);
        result.set_failure("unsupported diffuse light vertex");
        return;
      }
      l.albedo = rgb(spectrum_to_rgb(sd.closure[0].weight));
      const float3 incoming = v.incoming;
      l.incident_projection = double(dot(l.ns, incoming)) / double(dot(l.n, incoming));
    }
    vcm_photon_bounce(photon, l.p, v.label);
    lights.push_back(l);
    if (l.delta) {
      continue;
    }
    const float3 local = transform_point(&eye.world_to_camera, l.p);
    if (!(local.z > cam.nearclip && local.z - cam.nearclip < cam.cliplength)) {
      continue;
    }
    const float2 raster = vcm_raster(cam, eye, local);
    if (!(raster.x >= 0 && raster.x < cam.width && raster.y >= 0 && raster.y < cam.height)) {
      continue;
    }
    const Edge e = geometry(l, eye.endpoint);
    if (e.from_cos <= 0 || e.from_shading <= 0 || e.to_cos <= 0) {
      continue;
    }
    result.count_shadow();
    if (!visible(kg, l, eye.endpoint, true, eye)) {
      result.count_blocked();
      continue;
    }
    const double response = camera_pdf(e.to_cos, eye.plane_area) * e.from_cos *
                            l.incident_projection / (pi * e.length_squared);
    const size_t target = size_t(raster.y) * size_t(cam.width) + size_t(raster.x);
    if (!record(joined(lights, lights.size(), eyes),
                int(lights.size()),
                lamp_pdf,
                eye,
                target,
                VCM_CAMERA_SPLAT,
                product(l.beta, l.albedo, response),
                result))
    {
      return;
    }
  }

  Spectrum throughput = one_spectrum();
  for (uint depth = 0; depth <= batch.max_depth; ++depth) {
    Intersection hit = {};
    const bool found = scene_intersect(kg, &ray, path_state_ray_visibility(state), &hit);
    Ray segment = ray;
    if (found) {
      segment.tmax = hit.t;
    }
    /* Analytic lamps contribute on a camera hit but do not occlude other scene lights. */
    for (int i = 0; i < kernel_data.integrator.num_lights; ++i) {
      float t;
      const KernelLight &lamp = kernel_data_fetch(lights, i);
      if (lamp.type == LIGHT_BACKGROUND) {
        continue;
      }
      if (lamp.type != LIGHT_AREA) {
        return;
      }
      if (depth == 0 && (lamp.shader_id & SHADER_EXCLUDE_CAMERA)) {
        continue;
      }
      if (vcm_area_intersect(kg, lamp, segment, &t)) {
        Node lamp_node = {};
        NodeList source = {&lamp_node, 1};
        double pdf;
        if (!light_endpoint(kg, i, ray.P + ray.D * t, source[0], pdf, eye.time) ||
            !record(joined(source, 1, eyes),
                    0,
                    pdf,
                    eye,
                    pixel,
                    VCM_CAMERA_HIT,
                    product(rgb(spectrum_to_rgb(throughput)), source[0].beta, 1.0),
                    result))
        {
          return;
        }
      }
    }
    if (!found) {
      if (!record_world(kg, state, ray, throughput, pixel, result)) {
        return;
      }
      break;
    }
    if (depth == batch.max_depth) {
      break;
    }
    ShaderData sd;
    sd.runtime_flag = 0;
    if (!replay_surface(kg, state, sd, ray, hit)) {
      result.cache_miss = vcm_triangle(hit.type) && (sd.runtime_flag & SR_CACHE_MISS);
      result.set_failure("unsupported camera surface");
      result.object = hit.object;
      result.prim = hit.prim;
      return;
    }
    if (!sd.num_closure || !(dot(sd.wi, sd.closure[0].N) > 0 && dot(sd.wi, sd.Ng) > 0)) {
      break;
    }
    Node c = {};
    c.p = sd.P;
    c.n = sd.Ng;
    c.ns = sd.closure[0].N;
    c.prim = sd.prim;
    c.object = sd.object;
    c.beta = rgb(spectrum_to_rgb(throughput));
    c.delta = sd.closure[0].type != CLOSURE_BSDF_DIFFUSE_ID;
    if (!c.delta) {
      c.albedo = rgb(spectrum_to_rgb(sd.closure[0].weight));
    }
    eyes.push_back(c);
    if (merging && !c.delta &&
        !merge_surface(kg, *merging, eyes, eye, pixel, batch.max_depth, result))
    {
      result.set_failure("invalid merge strategy or receiver");
      return;
    }
    for (size_t s = 1; !c.delta && s <= lights.size() && s + depth <= batch.max_depth; ++s) {
      const Node &l = lights[s - 1];
      if (l.delta) {
        continue;
      }
      const Edge e = geometry(c, l);
      if (e.from_cos <= 0.0 || e.to_cos <= 0.0 || e.from_shading <= 0.0 || e.to_shading <= 0.0) {
        continue;
      }
      result.count_shadow();
      if (!visible(kg, c, l, false, eye)) {
        result.count_blocked();
        continue;
      }
      const double g = e.from_shading * e.to_cos * l.incident_projection / e.length_squared;
      const RGB c_bsdf = product(c.beta, c.albedo, 1.0 / pi);
      const RGB l_bsdf = s == 1 ? l.beta : product(l.beta, l.albedo, 1.0 / pi);
      if (!record(joined(lights, s, eyes),
                  int(s),
                  lamp_pdf,
                  eye,
                  pixel,
                  s == 1 ? VCM_NEXT_EVENT : VCM_PAIRED,
                  product(c_bsdf, l_bsdf, s == 1 ? g / lamp_pdf : g),
                  result))
      {
        return;
      }
    }
    VCMVertex event = {};
    if (!vcm_scatter(kg, &sd, lcg_step_float3(&rng), event, &throughput, false)) {
      result.set_failure("unsupported camera closure");
      result.object = hit.object;
      result.prim = hit.prim;
      return;
    }
    if (is_zero(throughput) || event.pdf_forward == 0.0f) {
      break;
    }
    /* Light subpaths use importance transport. The camera traverses refraction in radiance mode.
     */
    if (event.label & LABEL_TRANSMIT) {
      throughput /= sqr(event.eta);
    }
    ray.D = event.outgoing;
    ray.P = vcm_scattered_origin(&sd, ray.D);
    ray.tmin = 0.0f;
    ray.tmax = FLT_MAX;
    ray.self.object = c.object;
    ray.self.prim = c.prim;
    path_state_next(kg, state, event.label, sd.runtime_flag);
    INTEGRATOR_STATE_WRITE(state, path, throughput) = throughput;
  }
  result.set_failure(nullptr);
}

}  // namespace vcm
CCL_NAMESPACE_END
