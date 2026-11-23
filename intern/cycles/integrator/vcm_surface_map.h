/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: Apache-2.0 */

#pragma once

#include "kernel/integrator/vcm_types.h"
#include "kernel/types.h"

#include <array>
#include <cmath>
#include <unordered_map>
#include <unordered_set>

#include "util/hash.h"
#include <vector>

CCL_NAMESPACE_BEGIN

/* Surface VM reference for one emission round. The emission count includes escapes.
 * flux is incident power at the surface, before scattering at that vertex. */
class VCMSurfaceMap {
  using Cell = std::array<int64_t, 3>;
  struct Photon {
    packed_float3 position, incoming, flux;
    uint64_t vertex_id;
  };

  const double radius_;
  const uint emitted_;
  struct CellHash {
    size_t operator()(const Cell &c) const
    {
      const auto fold = [](int64_t i) {
        const uint64_t u = uint64_t(i);
        return uint(u) ^ uint(u >> 32) * 0x9e3779b9u;
      };
      return hash_uint3(fold(c[0]), fold(c[1]), fold(c[2]));
    }
  };
  std::unordered_map<Cell, std::vector<Photon>, CellHash> cells_;
  /* Point-source receivers only. Mark all coarse query blocks reached by the disk,
   * so even an isolated caustic survives the fast empty-region test. */
  std::unordered_set<Cell, CellHash> receiver_blocks_;

  bool cell_at(const double x, const double y, const double z, Cell &cell) const
  {
    const double p[3] = {x, y, z};
    for (int i = 0; i < 3; ++i) {
      const double index = std::floor(p[i] / radius_);
      /* Keep every unit offset exactly representable, including adjacent-cell queries. */
      if (!(std::abs(index) < 0x1p52)) {
        return false;
      }
      cell[i] = int64_t(index);
    }
    return true;
  }

 public:
  VCMSurfaceMap(const double radius, const uint emitted) : radius_(radius), emitted_(emitted) {}

  bool insert(const uint path, const uint depth, const VCMVertex &v)
  {
    if (!(v.label & (LABEL_DIFFUSE | VCM_VERTEX_RECEIVER))) {
      return true;
    }
    Cell cell;
    if (!cell_at(v.position.x, v.position.y, v.position.z, cell)) {
      return false;
    }
    auto &bucket = cells_[cell];
    if (bucket.empty() && (v.label & VCM_VERTEX_RECEIVER)) {
      for (int64_t z = (cell[2]-1) >> 3; z <= (cell[2]+1) >> 3; ++z) {
        for (int64_t y = (cell[1]-1) >> 3; y <= (cell[1]+1) >> 3; ++y) {
          for (int64_t x = (cell[0]-1) >> 3; x <= (cell[0]+1) >> 3; ++x) {
            receiver_blocks_.insert({x, y, z});
          }
        }
      }
    }
    bucket.push_back({v.position, v.incoming, v.flux, (uint64_t(path) << 32) | depth});
    return true;
  }

  template<typename Visitor>
  bool visit(const std::array<double, 3> &p,
             const std::array<double, 3> &normal,
             const Visitor &visitor,
             const bool both_sides = false) const
  {
    Cell center;
    if (!cell_at(p[0], p[1], p[2], center)) {
      return false;
    }
    if (cells_.empty() || (both_sides &&
        !receiver_blocks_.contains({center[0] >> 3, center[1] >> 3, center[2] >> 3}))) {
      return true;
    }
    for (int z = -1; z <= 1; ++z) {
      for (int y = -1; y <= 1; ++y) {
        for (int x = -1; x <= 1; ++x) {
          auto bucket = cells_.find({center[0] + x, center[1] + y, center[2] + z});
          if (bucket == cells_.end()) {
            continue;
          }
          for (const Photon &v : bucket->second) {
            const double dx = double(v.position.x) - p[0], dy = double(v.position.y) - p[1],
                         dz = double(v.position.z) - p[2];
            if (dx * dx + dy * dy + dz * dz > radius_ * radius_ ||
                (!both_sides && normal[0] * v.incoming.x + normal[1] * v.incoming.y +
                                   normal[2] * v.incoming.z <= 0))
            {
              continue;
            }
            if (!visitor(v.vertex_id, v.flux)) {
              return false;
            }
          }
        }
      }
    }
    return true;
  }

  /* Unit Lambert probe, normalized over every emitted path, including escapes. */
  bool radiance(const std::array<double, 3> &p,
                const std::array<double, 3> &normal,
                std::array<double, 3> &rgb,
                uint64_t &hits,
                std::vector<uint64_t> *ids) const
  {
    hits = 0;
    rgb = {};
    if (!visit(p, normal, [&](uint64_t id, const packed_float3 &flux) {
          ++hits;
          if (ids) {
            ids->push_back(id);
          }
          rgb[0] += flux.x;
          rgb[1] += flux.y;
          rgb[2] += flux.z;
          return true;
        }))
    {
      return false;
    }
    constexpr double pi = 3.14159265358979323846;
    for (double &c : rgb) {
      c /= double(emitted_) * pi * pi * radius_ * radius_;
    }
    return true;
  }
};

CCL_NAMESPACE_END
