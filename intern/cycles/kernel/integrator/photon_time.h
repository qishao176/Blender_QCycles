/* SPDX-FileCopyrightText: 2026 Blender Foundation
 * SPDX-License-Identifier: Apache-2.0 */
#pragma once
#include "kernel/util/lookup_table.h"
#include "util/hash.h"
CCL_NAMESPACE_BEGIN

/* A map and all its camera queries share one shutter sample. Reusing the map
 * reuses its time, too. Never merge photons from different moving geometries. */
ccl_device_inline float photon_caustics_time_random(const uint iteration)
{
  return hash_uint2_to_float(iteration, 0x3c6ef372u);
}
ccl_device_inline float photon_caustics_time(KernelGlobals kg, const uint iteration)
{
  return kernel_data.cam.shuttertime == -1.0f ?
             0.5f :
             lookup_table_read(kg,
                               photon_caustics_time_random(iteration),
                               kernel_data.cam.shutter_table_offset,
                               SHUTTER_TABLE_SIZE);
}
CCL_NAMESPACE_END
