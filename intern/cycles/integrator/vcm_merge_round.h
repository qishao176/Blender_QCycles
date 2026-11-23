/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: Apache-2.0 */

#pragma once

#include "integrator/vcm_surface_map.h"

CCL_NAMESPACE_BEGIN

/* Published to CPU camera workers only after the last light batch of this round.
 * Keep only realized vertices; path ids index the deterministic global emission order. */
struct VCMMergeRound {
  VCMSurfaceMap surface;
  double eta;
  std::vector<VCMLightPath> paths;
  std::vector<size_t> offsets;
  std::vector<VCMVertex> vertices;

  VCMMergeRound(const double radius, const uint count)
      : surface(radius, count), eta(3.14159265358979323846 * radius * radius * count)
  {
    paths.reserve(count);
    offsets.reserve(count);
  }

  bool append(const VCMLightPath &path, const VCMVertex *source)
  {
    paths.push_back(path);
    offsets.push_back(vertices.size());
    vertices.insert(vertices.end(), source, source + path.count);
    /* Direct arrivals are already available to connection strategies. Keep the
     * vertex in the path for VC and deeper merges, but omit it from the VM grid. */
    for (uint i = 1; i < path.count; ++i) {
      if (!surface.insert(path.index, i, source[i])) {
        return false;
      }
    }
    return true;
  }

  template<typename Visitor>
  bool visit(const float3 p, const float3 n, const Visitor &visitor, bool both_sides = false) const
  {
    return surface.visit({p.x, p.y, p.z}, {n.x, n.y, n.z}, visitor, both_sides);
  }

  const VCMVertex *prefix(const uint index) const
  {
    return paths[index].count ? &vertices[offsets[index]] : nullptr;
  }
};

CCL_NAMESPACE_END
