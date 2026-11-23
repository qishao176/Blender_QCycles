/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: Apache-2.0 */
#pragma once

#include "util/math.h"

CCL_NAMESPACE_BEGIN

/* Randomized systematic selection. Integer phase avoids the rounding drift of
 * floor(i*K/N+u) on long chains. Every phase selects exactly K of N entries;
 * averaging the N possible phases includes each entry K times. */
struct PhotonGatherSelection {
  uint count, limit, phase;
  float weight;
};

ccl_device_inline PhotonGatherSelection photon_gather_selection(const uint count,
                                                               const uint limit,
                                                               const uint random)
{
  const uint selected = limit ? min(count, limit) : count;
  return {count,
          selected,
          uint((uint64_t(random) * count) >> 32),
          selected ? float(count) / float(selected) : 1.0f};
}

ccl_device_inline bool photon_gather_select(ccl_private PhotonGatherSelection *selection)
{
  if (selection->limit >= selection->count) {
    return true;
  }
  selection->phase += selection->limit;
  if (selection->phase < selection->count) {
    return false;
  }
  selection->phase -= selection->count;
  return true;
}

CCL_NAMESPACE_END
