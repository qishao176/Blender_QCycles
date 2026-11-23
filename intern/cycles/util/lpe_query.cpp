/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: Apache-2.0 */
#include "util/lpe_query.h"

#include <atomic>
#include <cmath>
#include <condition_variable>
#include <utility>

CCL_NAMESPACE_BEGIN
namespace {
std::atomic<bool> active{false};
std::atomic<bool> group_events{false};
std::atomic<bool> paused{false};
std::mutex pause_mutex;
std::condition_variable pause_condition;
std::mutex result_mutex;
QueryResults results;
}  // namespace

bool lpe_query_active()
{
  return active.load(std::memory_order_relaxed);
}

void lpe_query_set_paused(bool pause)
{
  {
    std::lock_guard lock(pause_mutex);
    paused.store(pause);
  }
  if (!pause) {
    pause_condition.notify_all();
  }
}

void lpe_query_wait_if_paused()
{
  if (paused.load(std::memory_order_relaxed)) {
    std::unique_lock lock(pause_mutex);
    pause_condition.wait(lock, [] { return !paused.load(); });
  }
}

void lpe_query_begin(const bool total_light)
{
  group_events.store(total_light);
  lpe_query_set_paused(false);
  std::lock_guard lock(result_mutex);
  results = {};
  active.store(true);
}

QueryResults lpe_query_take(bool finish)
{
  std::lock_guard lock(result_mutex);
  if (finish) {
    active.store(false);
    return std::exchange(results, QueryResults{});
  }
  QueryResults batch;
  batch.objects = results.objects;
  batch.materials = results.materials;
  batch.world = results.world;
  batch.images.swap(results.images);
  return batch;
}

void lpe_query_names(std::map<int, std::string> objects, std::map<int, std::string> materials, int world)
{
  std::lock_guard lock(result_mutex);
  results.objects = std::move(objects);
  results.materials = std::move(materials);
  results.world = world;
}

QueryPixel::QueryPixel(const int x, const int y)
    : point{x, y}, total_light(group_events.load(std::memory_order_relaxed))
{
}

void QueryPixel::record(const QueryPath &path, bool photon, float r, float g, float b)
{
  if ((!r && !g && !b) || !std::isfinite(r) || !std::isfinite(g) || !std::isfinite(b)) {
    return;
  }
  auto key = std::make_pair(path, photon);
  if (total_light && !path.empty()) {
    /* Keep emitter identity, ordered surface identities and private count markers intact. */
    for (size_t i = 0; i + 1 < path.size(); ++i) {
      key.first[i][2] = 0;
    }
  }
  auto it = slots.find(key);
  if (it == slots.end()) {
    if (occupied.all()) {
      /* Keep pages private until the sample count and beauty for this pixel are available. */
      for (const auto &[route, lane] : slots) {
        QueryRGB &dst = overflow[route];
        for (int c = 0; c < 3; ++c) {
          dst[c] += values[lane][c];
        }
      }
      slots.clear();
      occupied.reset();
    }
    const int lane = int(slots.size());
    it = slots.emplace(key, lane).first;
    occupied.set(lane);
    values[lane] = {};
  }
  const int lane = it->second;
  values[lane][0] += r;
  values[lane][1] += g;
  values[lane][2] += b;
}

void QueryPixel::flush()
{
  if (occupied.none()) {
    return;
  }
  std::lock_guard lock(result_mutex);
  for (const auto &[key, rgb] : overflow) {
    QueryRGB &dst = results.images[key][point];
    for (int c = 0; c < 3; ++c) {
      dst[c] += rgb[c];
    }
  }
  overflow.clear();
  for (const auto &[key, lane] : slots) {
    QueryRGB &dst = results.images[key][point];
    for (int channel = 0; channel < 3; ++channel) {
      dst[channel] += values[lane][channel];
    }
  }
  slots.clear();
  occupied.reset();
}
CCL_NAMESPACE_END
