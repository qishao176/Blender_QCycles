/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: Apache-2.0 */

#pragma once

#include "util/atomic.h"
#include "util/math.h"
#include "kernel/integrator/deep_bounded_records.h"

CCL_NAMESPACE_BEGIN

/* Physical capture uses one in-flight camera sample per pixel. A completed sample
 * is reduced on its rendering device before the next sample starts. Geometry and
 * opacity are shared by all requested LPE colours; no LPE buffer exists otherwise.
 * The bounded grouping policy and depth representation are adapted from Industrial
 * CG Platform 670a1e140609 (RolandVyens). Colour side planes follow every grouping. */
constexpr int DEEP_PHYSICAL_CAPACITY = 32;
struct DeepPhysicalHeader {
  uint raw_count, count, samples, active;
  float transmission;
  uint error, reserved0, reserved1;
  float4 albedo[4]; /* Exact native pass divisors; accumulated independently of depth reduction. */
};
using DeepPhysicalRecord = DeepBoundedRecord<true>;
static_assert(sizeof(DeepPhysicalHeader) == 96);
static_assert(sizeof(DeepPhysicalRecord) == 36);

ccl_device_inline size_t deep_physical_pixel_bytes(const int channels)
{
  return sizeof(DeepPhysicalHeader) + 2 * DEEP_PHYSICAL_CAPACITY *
      (sizeof(DeepPhysicalRecord) + size_t(channels) * sizeof(float4));
}

struct DeepPhysicalPixel {
  ccl_global DeepPhysicalHeader *header;
  ccl_global DeepPhysicalRecord *raw;
  ccl_global DeepPhysicalRecord *history;
  ccl_global float4 *colours;
  int channels;
};

ccl_device_inline DeepPhysicalPixel deep_physical_pixel(const uint64_t address,
                                                       const uint pixel,
                                                       const int channels)
{
  ccl_global uchar *base = (ccl_global uchar *)address +
                           size_t(pixel) * deep_physical_pixel_bytes(channels);
  auto *header = (ccl_global DeepPhysicalHeader *)base;
  auto *raw = (ccl_global DeepPhysicalRecord *)(header + 1);
  return {header, raw, raw + DEEP_PHYSICAL_CAPACITY,
          (ccl_global float4 *)(raw + 2 * DEEP_PHYSICAL_CAPACITY), channels};
}

ccl_device_inline void deep_physical_begin(const uint64_t address,
                                          const uint pixel,
                                          const int channels,
                                          const bool additive = false)
{
  auto p = deep_physical_pixel(address, pixel, channels);
  p.header->raw_count = 0;
  p.header->transmission = 1.0f;
  p.header->active = additive ? 2 : 1;
}

ccl_device_inline int deep_physical_find(const DeepPhysicalPixel &p,
                                       const float anchor,
                                       const uint kind)
{
  int best = -1;
  float distance = FLT_MAX;
  float extent = FLT_MAX;
  bool best_kind_matches = false;
  for (uint i = 0; i < p.header->raw_count; ++i) {
    const auto &r = p.raw[i];
    const float d = fmaxf(r.z - anchor, fmaxf(anchor - r.z_back, 0.0f));
    const uint record_kind = (r.object & ~DEEP_BOUNDED_ELIGIBLE) != DEEP_BOUNDED_NONSURFACE ?
                                1u : (r.shader == 1 ? 3u : 2u);
    /* An opacity interval can contain a separately recorded direct-scatter
     * point. Resolve the exact point before a wider interval at the same depth. */
    const float width = r.z_back - r.z;
    const bool kind_matches = record_kind == kind;
    if (d < distance || (d == distance &&
                         ((kind_matches && !best_kind_matches) ||
                          (kind_matches == best_kind_matches && width < extent)))) {
      best_kind_matches = kind_matches;
      best = i;
      distance = d;
      extent = width;
      if (d == 0.0f && width == 0.0f && record_kind == kind) break;
    }
  }
  return best;
}

ccl_device_inline void deep_physical_make_room(const DeepPhysicalPixel &p)
{
  /* Raw chains retain the front anchor, as in upstream's raw pressure path.
   * Late shadow contributions re-resolve an anchor, never a moved slot. */
  int pair = 1;
  float least = FLT_MAX;
  for (int i = 1; i < DEEP_PHYSICAL_CAPACITY; ++i) {
    const float mass = p.raw[i - 1].alpha + p.raw[i].alpha;
    if (mass < least) { least = mass; pair = i; }
  }
  DeepBoundedGroup<true, false> group(p.raw[pair - 1], false);
  group.append(p.raw[pair], false);
  auto merged = group.finish();
  merged.z_back = fmaxf(merged.z_back, p.raw[pair].z_back);
  p.raw[pair - 1] = merged;
  for (int c = 0; c < p.channels; ++c) {
    auto *colours = p.colours + c * 2 * DEEP_PHYSICAL_CAPACITY;
    colours[pair - 1] += colours[pair];
    for (int i = pair; i < DEEP_PHYSICAL_CAPACITY - 1; ++i) colours[i] = colours[i + 1];
  }
  for (int i = pair; i < DEEP_PHYSICAL_CAPACITY - 1; ++i) p.raw[i] = p.raw[i + 1];
  --p.header->raw_count;
}

ccl_device_inline void deep_physical_append(const uint64_t address,
                                           const uint pixel,
                                           const int channels,
                                           float z,
                                           float back,
                                           const float alpha,
                                           const uint object,
                                           const uint shader,
                                           const uint normal)
{
  auto p = deep_physical_pixel(address, pixel, channels);
  if (!p.header->active) return; /* Light-side photon tracing never owns a camera root. */
  if (!isfinite_safe(z) || !isfinite_safe(back) || !isfinite_safe(alpha)) {
    p.header->error = 1;
    return;
  }
  z = fmaxf(z, 1e-8f);
  back = fmaxf(z, back);
  if (p.header->raw_count == DEEP_PHYSICAL_CAPACITY) deep_physical_make_room(p);
  DeepPhysicalRecord r = {};
  r.z = z;
  r.z_back = back;
  r.alpha = saturatef(alpha) * p.header->transmission;
  p.header->transmission *= 1.0f - saturatef(alpha);
  r.object = object | DEEP_BOUNDED_ELIGIBLE;
  r.shader = shader;
  r.normal = normal;
  uint slot = p.header->raw_count++;
  while (slot && p.raw[slot - 1].z > z) {
    p.raw[slot] = p.raw[slot - 1];
    for (int c = 0; c < channels; ++c) {
      auto *colour = p.colours + c * 2 * DEEP_PHYSICAL_CAPACITY;
      colour[slot] = colour[slot - 1];
    }
    --slot;
  }
  p.raw[slot] = r;
  for (int c = 0; c < channels; ++c)
    p.colours[c * 2 * DEEP_PHYSICAL_CAPACITY + slot] = zero_float4();
}

ccl_device_inline void deep_physical_write(const uint64_t address,
                                          const uint pixel,
                                          const int channels,
                                          const float anchor,
                                          const uint kind,
                                          const float3 rgb,
                                          const int channel = -1)
{
  if (is_zero(rgb)) return;
  auto p = deep_physical_pixel(address, pixel, channels);
  if (!p.header->active) return;
  const int slot = deep_physical_find(p, anchor, kind);
  if (!kind || kind > 3 || slot < 0 || !isfinite_safe(rgb)) {
    p.header->error = 1;
    return;
  }
  float *dst = channel < 0 ? p.raw[slot].rgb :
      (float *)(p.colours + channel * 2 * DEEP_PHYSICAL_CAPACITY + slot);
  atomic_add_and_fetch_float(dst, rgb.x);
  atomic_add_and_fetch_float(dst + 1, rgb.y);
  atomic_add_and_fetch_float(dst + 2, rgb.z);
}

ccl_device_inline float3 deep_physical_side_sum(const DeepPhysicalPixel &p,
                                               uint64_t mask,
                                               const int channel,
                                               const float old_weight,
                                               const float new_weight)
{
  float3 sum = zero_float3();
  const auto *base = p.colours + channel * 2 * DEEP_PHYSICAL_CAPACITY;
  for (int i = 0; mask; ++i, mask >>= 1) {
    if (mask & 1) {
      const float4 rgb = base[i];
      sum += make_float3(rgb.x, rgb.y, rgb.z) *
             (i < DEEP_PHYSICAL_CAPACITY ? new_weight : old_weight);
    }
  }
  return sum;
}

/* Called after ALL shadows for the pixel's current camera sample have completed.
 * Absolute coverage mass and film-weighted RGB are averaged, not composited across
 * different camera samples. A zero-contribution sample still dilutes the estimate. */
ccl_device void deep_physical_finish(const uint64_t address,
                                    const uint pixel,
                                    const int channels)
{
  auto p = deep_physical_pixel(address, pixel, channels);
  if (!p.header->active) return;
  /* Independent photon paths add radiance at their own depth. Coverage and
   * its sample count already belong to the completed native camera paths. */
  const bool additive = p.header->active == 2;
  const float nw = 1.0f / float(additive ? max(p.header->samples, 1u) : p.header->samples + 1);
  const float ow = additive ? 1.0f : float(p.header->samples) * nw;
  DeepPhysicalRecord candidates[2 * DEEP_PHYSICAL_CAPACITY];
  uint64_t masks[2 * DEEP_PHYSICAL_CAPACITY];
  int count = 0;
  for (int source = 0; source < 2; ++source) {
    const uint n = source ? p.header->count : p.header->raw_count;
    for (uint i = 0; i < n; ++i) {
      auto record = source ? p.history[i] : p.raw[i];
      if (additive && !source) record.alpha = 0.0f;
      bool empty = record.alpha == 0.0f && record.rgb[0] == 0.0f &&
                   record.rgb[1] == 0.0f && record.rgb[2] == 0.0f;
      for (int c = 0; c < channels && empty; ++c) {
        const float4 rgb = p.colours[(c * 2 + source) * DEEP_PHYSICAL_CAPACITY + i];
        empty = rgb.x == 0.0f && rgb.y == 0.0f && rgb.z == 0.0f;
      }
      /* In particular, transparent black environment placeholders must not turn
       * into far-away points in DeepToPoints or consume the bounded depth budget. */
      if (empty) continue;
      if (!source) deep_bounded_classify_raw_surface(record, p.header->raw_count);
      const float weight = source ? ow : nw;
      record.alpha *= weight;
      for (int c = 0; c < 3; ++c) record.rgb[c] *= weight;
      uint64_t mask = uint64_t(1) << (i + source * DEEP_PHYSICAL_CAPACITY);
      int j = count++;
      while (j && candidates[j - 1].z > record.z) {
        candidates[j] = candidates[j - 1]; masks[j] = masks[j - 1]; --j;
      }
      candidates[j] = record; masks[j] = mask;
    }
  }
  /* Ordinary reduction uses upstream geometry/normal/colour compatibility. All
   * requested LPE colours also gate a merge; a flat Beauty must not erase an LPE. */
  const DeepBoundedMergePolicy policy{1e-5f, 0.0f, 1e-6f, 0.01f, 1e-5f};
  int ordinary = 0;
  float prefix = 0.0f;
  for (int i = 0; i < count;) {
    DeepBoundedGroup<true, true> group(candidates[i], true);
    uint64_t mask = masks[i++];
    while (i < count && deep_bounded_ordinary(group.record, candidates[i], policy,
                                             1.0f - prefix, true)) {
      bool compatible = true;
      for (int c = 0; c < channels && compatible; ++c) {
        auto a = group.record, b = candidates[i];
        const float3 ar = deep_physical_side_sum(p, mask, c, ow, nw);
        const float3 br = deep_physical_side_sum(p, masks[i], c, ow, nw);
        a.rgb[0] = ar.x; a.rgb[1] = ar.y; a.rgb[2] = ar.z;
        b.rgb[0] = br.x; b.rgb[1] = br.y; b.rgb[2] = br.z;
        compatible = deep_bounded_ordinary(a, b, policy, 1.0f - prefix, true);
      }
      if (!compatible) break;
      group.append(candidates[i]); mask |= masks[i++];
    }
    candidates[ordinary] = group.finish(); masks[ordinary++] = mask;
    prefix += group.record.alpha;
  }
  int written = ordinary;
  if (ordinary > DEEP_PHYSICAL_CAPACITY) {
    /* Coverage quantiles can drag a zero-alpha lighting event into an unlit fog
     * interval. Preserve each requested channel's occupied support first. Only
     * merge unlike supports if their distinct classes exceed the fixed capacity.
     * Signatures are a fast rejection; channels beyond bit 62 are checked exactly. */
    uint64_t support[2 * DEEP_PHYSICAL_CAPACITY];
    for (int i = 0; i < written; ++i) {
      const auto &r = candidates[i];
      support[i] = (r.rgb[0] != 0 || r.rgb[1] != 0 || r.rgb[2] != 0) ? 1u : 0u;
      for (int c = 0; c < channels; ++c) {
        if (!is_zero(deep_physical_side_sum(p, masks[i], c, ow, nw)))
          support[i] |= uint64_t(1) << min(c + 1, 63);
      }
    }
    while (written > DEEP_PHYSICAL_CAPACITY) {
      int first = 0, second = 1;
      bool found_same = false;
      float best_span = FLT_MAX;
      for (int a = 0; a < written - 1; ++a) {
        for (int b = a + 1; b < written; ++b) {
          bool same = support[a] == support[b];
          for (int c = 62; c < channels && same; ++c) {
            same = is_zero(deep_physical_side_sum(p, masks[a], c, ow, nw)) ==
                   is_zero(deep_physical_side_sum(p, masks[b], c, ow, nw));
          }
          const float span = fmaxf(candidates[a].z_back, candidates[b].z_back) -
                             candidates[a].z;
          if ((same && !found_same) || (same == found_same && span < best_span)) {
            first = a; second = b; best_span = span; found_same = same;
          }
        }
      }
      const float front = candidates[first].z;
      const float back = fmaxf(candidates[first].z_back, candidates[second].z_back);
      DeepBoundedGroup<true, false> group(candidates[first], true);
      group.append(candidates[second], false);
      candidates[first] = group.finish();
      candidates[first].z = front;
      candidates[first].z_back = back;
      masks[first] |= masks[second];
      support[first] |= support[second];
      for (int i = second; i < written - 1; ++i) {
        candidates[i] = candidates[i + 1]; masks[i] = masks[i + 1];
        support[i] = support[i + 1];
      }
      --written;
    }
  }
  for (int c = 0; c < channels; ++c) {
    float4 output[DEEP_PHYSICAL_CAPACITY];
    for (int i = 0; i < written; ++i) {
      const float3 rgb = deep_physical_side_sum(p, masks[i], c, ow, nw);
      output[i] = make_float4(rgb.x, rgb.y, rgb.z, 0);
    }
    auto *dst = p.colours + (c * 2 + 1) * DEEP_PHYSICAL_CAPACITY;
    for (int i = 0; i < written; ++i) dst[i] = output[i];
  }
  for (int i = 0; i < written; ++i) p.history[i] = candidates[i];
  p.header->count = written;
  if (!additive) ++p.header->samples;
  p.header->active = 0;
}

CCL_NAMESPACE_END
