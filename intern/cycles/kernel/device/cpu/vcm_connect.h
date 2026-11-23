/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: Apache-2.0 */

#pragma once

#include "kernel/integrator/vcm_types.h"
#include <array>
#include <vector>

CCL_NAMESPACE_BEGIN

struct ThreadKernelGlobalsCPU;
struct VCMMergeRound;

struct VCMFilmSample {
  size_t pixel;
  VCMTechnique technique;
  std::array<double, 3> rgb;
  int source_light = -2;
};

struct VCMConnectionResult {
  std::vector<VCMFilmSample> samples;
  uint shadow = 0, blocked = 0, singular_skipped = 0;
  double max_weight_error = 0.0;
  bool check_weights = false;
  bool cache_miss = false;
  void set_failure(const char *message) { failure = message; }
  void count_shadow()
  {
    ++shadow;
  }
  void count_blocked()
  {
    ++blocked;
  }
  void count_delta()
  {
    ++singular_skipped;
  }
  void add(size_t pixel, VCMTechnique type, const VCMColor &v)
  {
    samples.push_back({pixel, type, {v[0], v[1], v[2]}});
  }
  int num_lights = 0;
  bool route_sources = false;
  void add_primary(size_t pixel, bool world, const VCMColor &v)
  {
    add_source(pixel, num_lights + int(world), v);
  }
  void add_source(size_t pixel, int source_light, const VCMColor &v)
  {
    if (route_sources && source_light >= -1) {
      samples.push_back({pixel, VCM_SURFACE_MERGE, {v[0], v[1], v[2]}, source_light});
    }
  }
  const char *failure = "unsupported emitter or direction";
  int object = -1, prim = -1;
};

/* One independent camera path paired with one global emission attempt, including escapes. */
void vcm_connect_cpu(const ThreadKernelGlobalsCPU *kg,
                     const VCMBatch &batch,
                     const VCMLightPath &light,
                     const VCMVertex *vertices,
                     const VCMMergeRound *merging,
                     VCMConnectionResult &result);

CCL_NAMESPACE_END
