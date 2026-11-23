/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: Apache-2.0 */

// clang-format off
#include "kernel/device/optix/compat.h"
#include "kernel/device/optix/globals.h"
#include "kernel/device/gpu/image.h"
#include "kernel/geom/primitive.h"
#include "kernel/integrator/vcm_emit.h"

#include "kernel/camera/camera.h"
#include "kernel/integrator/state_util.h"
// clang-format on

namespace vcm {

struct Result {
  float *film;
  size_t pixels;
  const int *route_sources;
  bool failure = true;
  ccl_device_inline_method void set_failure(const char *message) { failure = message != nullptr; }
  int object = -1, prim = -1;
  bool cache_miss = false;
  ccl_device_inline_method void count_shadow() {}
  ccl_device_inline_method void count_blocked() {}
  ccl_device_inline_method void count_delta() {}
  ccl_device_inline_method void add_primary(size_t pixel, bool world, const VCMColor &rgb)
  {
    add_source(pixel, kernel_data.integrator.num_lights + int(world), rgb);
  }
  ccl_device_inline_method void add_source(size_t pixel, int light, const VCMColor &rgb)
  {
    if (route_sources && light >= -1) {
      const int source = light + 1;
      for (int i = route_sources[source]; i < route_sources[source + 1]; ++i) {
        add(pixel + size_t(1 + route_sources[i]) * pixels, VCM_CAMERA_HIT, rgb);
      }
    }
  }
  ccl_device_inline_method void add(size_t pixel, VCMTechnique, const VCMColor &rgb)
  {
    for (int c = 0; c < 3; ++c) {
      /* The film is global device storage. Qualify the reduction explicitly in PTX. */
      const auto dst = __cvta_generic_to_global(film + pixel * 3 + c);
      asm volatile("red.global.add.f32 [%0], %1;" :: "l"(dst), "f"(float(rgb[c])) : "memory");
    }
  }
};

ccl_device bool grid_cell(const float3 p, const double radius, int3 &c)
{
  const double x = floor(double(p.x) / radius), y = floor(double(p.y) / radius),
               z = floor(double(p.z) / radius);
  if (!(fabs(x) < 2147483646.0 && fabs(y) < 2147483646.0 && fabs(z) < 2147483646.0)) {
    return false;
  }
  c = make_int3(int(x), int(y), int(z));
  return true;
}

/* OptiX PTX may target a virtual architecture older than SM 6.0. */
ccl_device void add_moment(double *p, double value)
{
#if __CUDA_ARCH__ >= 600
  atomicAdd(p, value);
#else
  auto *bits = reinterpret_cast<unsigned long long *>(p);
  unsigned long long previous = atomicCAS(bits, 0ull, 0ull), expected;
  do {
    expected = previous;
    previous = atomicCAS(
        bits, expected, __double_as_longlong(__longlong_as_double(expected) + value));
  } while (previous != expected);
#endif
}

ccl_device uint bucket(const int3 cell, const uint mask)
{
  return hash_uint3(uint(cell.x), uint(cell.y), uint(cell.z)) & mask;
}

struct MergeRound {
  VCMLightPath *paths;
  VCMVertex *vertices;
  const VCMLaunch &launch;
  double eta;
  ccl_device_inline_method const VCMVertex *prefix(uint index) const
  {
    return vertices + size_t(index) * launch.batch.max_depth;
  }
  template<typename Visitor>
  ccl_device_inline_method bool visit(const float3 p,
                                      const float3 n,
                                      const Visitor &visitor,
                                      bool both_sides = false) const
  {
    int3 center;
    if (!grid_cell(p, launch.radius, center)) {
      return false;
    }
    for (int z = -1; z <= 1; ++z) {
      for (int y = -1; y <= 1; ++y) {
        for (int x = -1; x <= 1; ++x) {
          const int3 cell = center + make_int3(x, y, z);
          uint link = launch.heads[bucket(cell, launch.bucket_mask)];
          while (link) {
            const uint slot = link - 1;
            link = launch.links[slot];
            const VCMVertex &v = vertices[slot];
            const double dx = double(v.position.x) - p.x, dy = double(v.position.y) - p.y,
                         dz = double(v.position.z) - p.z;
            if (dx * dx + dy * dy + dz * dz > launch.radius * launch.radius ||
                (!both_sides && dot(n, float3(v.incoming)) <= 0))
            {
              continue;
            }
            /* Hash collisions, including within the 27 queried buckets, must not
             * count the same photon twice. */
            int3 actual;
            if (!grid_cell(v.position, launch.radius, actual) || actual != cell) {
              continue;
            }
            const uint path = slot / launch.batch.max_depth;
            const uint depth = slot % launch.batch.max_depth;
            if (!visitor((uint64_t(path) << 32) | depth, v.flux)) {
              return false;
            }
          }
        }
      }
    }
    return true;
  }
};

}  // namespace vcm

#include "kernel/integrator/vcm_connect_impl.h"

extern "C" __global__ void __raygen__kernel_optix_vcm_emit()
{
  const VCMLaunch &launch = *reinterpret_cast<const VCMLaunch *>(kernel_params.path_index_array);
  const uint lane = optixGetLaunchIndex().x;
  if (launch.accumulate) {
    const size_t i = size_t(3) * lane;
    for (int c = 0; c < 3; ++c) {
      const float value = launch.film[i + c];
      launch.accumulate[i + c] += value;
      if (launch.adaptive_sum && lane < size_t(kernel_data.cam.width) * size_t(kernel_data.cam.height)) {
        launch.adaptive_sum[i + c] += 2.0f * value;
      }
    }
    return;
  }
  if (!launch.heads) {
    integrator_vcm_emit(nullptr, lane, launch.batch, lane, launch.paths, launch.vertices);
    return;
  }
  const uint index = launch.batch.first + lane;
  const uint first_slot = index * launch.batch.max_depth;
  VCMVertex *vertices = launch.vertices + first_slot;
  VCMLightPath &path = launch.paths[index];
  if (launch.connect) {
    vcm::Result result = {launch.film, size_t(kernel_data.cam.width) * size_t(kernel_data.cam.height), launch.source_routes};
    const vcm::MergeRound round = {launch.paths, launch.vertices, launch, launch.eta};
    VCMConnectionNode *scratch = launch.scratch ?
                                     launch.scratch +
                                         size_t(lane) * 2 * (launch.batch.max_depth + 1) :
                                     nullptr;
    vcm::connect_path(nullptr, lane, scratch, launch.batch, path, vertices, &round, result);
    if (result.failure) {
      atomicOr(launch.status, result.cache_miss ? 1u : 2u);
    }
    return;
  }
  integrator_vcm_emit(nullptr,
                      lane,
                      launch.batch,
                      lane,
                      launch.paths + launch.batch.first,
                      launch.vertices + size_t(launch.batch.first) * launch.batch.max_depth);
  if (path.status < VCM_PATH_ESCAPE || path.status > VCM_PATH_ABSORBED) {
    atomicOr(launch.status, path.status == VCM_PATH_CACHE_MISS ? 1u : 2u);
    return;
  }
  bool trained = false;
  for (uint k = 1; k < path.count; ++k) {
    const VCMVertex &v = vertices[k];
    if (!(v.label & (LABEL_DIFFUSE | VCM_VERTEX_RECEIVER))) {
      continue;
    }
    int3 cell;
    if (!vcm::grid_cell(v.position, launch.radius, cell)) {
      atomicOr(launch.status, 4u);
      return;
    }
    const uint slot = first_slot + k;
    launch.links[slot] = atomicExch(&launch.heads[vcm::bucket(cell, launch.bucket_mask)],
                                    slot + 1);
    if (!trained && (v.label & VCM_VERTEX_RECEIVER)) {
      const double w = average(float3(v.flux));
      const float3 p = float3(vertices[0].position) - float3(path.source);
      VCMGuideMoments &m = launch.moments[path.light];
      vcm::add_moment(&m.weight, w);
      vcm::add_moment(&m.squared_weight, w * w);
      vcm::add_moment(&m.second, w * (double(p.x) * p.x + double(p.y) * p.y + double(p.z) * p.z));
      vcm::add_moment(&m.first[0], w * p.x);
      vcm::add_moment(&m.first[1], w * p.y);
      vcm::add_moment(&m.first[2], w * p.z);
      trained = true;
    }
  }
}
