/* SPDX-FileCopyrightText: 2026 Blender Foundation
 * SPDX-License-Identifier: Apache-2.0 */
#pragma once

#include "util/hash.h"
#include "util/types.h"

CCL_NAMESPACE_BEGIN

/* Ordered identity, independent of outputs, sample slots, normals and power. */
ccl_device_inline uint photon_path_hash_append(const uint previous,
                                               const int object,
                                               const int material,
                                               const int event)
{
  return hash_uint2(previous, hash_uint3(uint(object), uint(material), uint(event)));
}

template<typename Event>
ccl_device_inline bool photon_path_events_equal(const ccl_global Event *a,
                                                const ccl_global Event *b,
                                                const uint length,
                                                const int material_mask)
{
  for (uint i = 0; i < length; ++i) {
    if (a[i].object != b[i].object || a[i].event != b[i].event ||
        (a[i].shader & material_mask) != (b[i].shader & material_mask))
    {
      return false;
    }
  }
  return true;
}

CCL_NAMESPACE_END
