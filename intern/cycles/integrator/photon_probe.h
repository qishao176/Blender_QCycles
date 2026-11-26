/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: Apache-2.0 */
#pragma once

#include "util/types.h"

CCL_NAMESPACE_BEGIN

struct PhotonViewportProbe {
  uint maps = 0;
  uint64_t emitted = 0;
  uint64_t stored = 0;
  uint64_t sampled = 0;
  uint64_t isolated = 0;
  uint64_t wider_helped = 0;
  uint64_t in_frame = 0;
  uint64_t subpixel = 0;
  uint64_t radius_limited = 0;
  uint64_t pixel_scale_sum = 0;
  float radius = 0;
  float max_scale = 1;
  uint64_t volume_stored = 0;
  uint64_t device_maps = 0;
  uint completed = 0;
  uint requested = 0;
  uint capacity = 0;
  uint max_emission = 0;
  uint interval = 1;
  uint sample_limit = 0;
  bool has_volumes = false;
  double count_m2[2] = {};
  double deposit_cv = 0;
};

CCL_NAMESPACE_END
