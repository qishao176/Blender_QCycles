/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: Apache-2.0 */
#pragma once

#include "util/defines.h"

#include <array>
#include <atomic>
#include <memory>
#include <bitset>
#include <map>
#include <mutex>
#include <string>
#include <tuple>
#include <vector>

CCL_NAMESPACE_BEGIN

/* CPU diagnostic worker only. Device indices are resolved after scene synchronization.
 * Event bit 8 identifies a caustic caster. The last vertex is the emitter. */
using QueryVertex = std::array<int, 3>;
using QueryPath = std::vector<QueryVertex>;
using QueryRGB = std::array<double, 3>;
using QueryPoint = std::array<int, 2>;
using QueryImage = std::map<QueryPoint, QueryRGB>;
using QueryImages = std::map<std::pair<QueryPath, bool>, QueryImage>;

struct QueryPixel {
  QueryPath camera, shadow, manifold;
  const void *shadow_state = nullptr;
  QueryVertex receiver{}, source{};
  int receiver_event = 0;
  bool skip = false;
  std::map<std::pair<QueryPath, bool>, int> slots;
  /* Each page keeps 128 simultaneously occupied routes. Flush a full page so a rare
   * late sample still contributes. CPU/GPU production LPE state has its own ABI. */
  std::bitset<128> occupied;
  static_assert(sizeof(occupied) == 16);
  std::array<QueryRGB, 128> values;
  std::map<std::pair<QueryPath, bool>, QueryRGB> overflow;
  QueryPoint point;

  const bool total_light;
  QueryPixel(int x, int y);
  void record(const QueryPath &path, bool photon, float r, float g, float b);
  void flush();
};

struct QueryResults {
  QueryImages images;
  std::map<int, std::string> objects, materials;
  int world = -1;
};
struct QuerySlice {
  int frame_w, frame_h, x, y, w, h, sample, count;
  bool total_light = false;
  bool last_slice = false;
  uint64_t stream = 0, revision = 0;
};
struct QuerySliceResult {
  QueryResults data;
  std::string error;
  size_t bytes = 0;
  uint64_t photon_builds = 0, photon_reuses = 0;
};
struct QueryTicket {
  QuerySlice slice;
  QuerySliceResult result;
  std::atomic_bool finished{false}, cancelled{false};
  uint64_t revision = 0;
};
bool lpe_query_active();
void lpe_query_begin(bool total_light = false);
void lpe_query_set_paused(bool pause);
void lpe_query_wait_if_paused();
QueryResults lpe_query_take(bool finish);
void lpe_query_names(std::map<int, std::string> objects, std::map<int, std::string> materials, int world);

CCL_NAMESPACE_END
