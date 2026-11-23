/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: Apache-2.0 */

// clang-format off
#include "util/color.h"
#include "kernel/device/cpu/vcm_connect.h"
#include "kernel/globals.h"
#include "kernel/integrator/state.h"
#include "kernel/integrator/state_flow.h"
#include "kernel/integrator/state_util.h"
#include "kernel/integrator/vcm_emit.h"
#include "kernel/util/image.h"
#include "kernel/camera/camera.h"
// clang-format on

#include "integrator/vcm_merge_round.h"

#include <algorithm>
#include <cmath>
#include <limits>

CCL_NAMESPACE_BEGIN
namespace vcm {
using Result = VCMConnectionResult;
using MergeRound = VCMMergeRound;
}  // namespace vcm
CCL_NAMESPACE_END

#include "kernel/integrator/vcm_connect_impl.h"

CCL_NAMESPACE_BEGIN
void vcm_connect_cpu(const ThreadKernelGlobalsCPU *kg,
                     const VCMBatch &batch,
                     const VCMLightPath &light,
                     const VCMVertex *vertices,
                     const VCMMergeRound *merging,
                     VCMConnectionResult &result)
{
  IntegratorStateCPU state = {};
  if (batch.native_bsdf || light.light < 0 || vcm_point_source(&kernel_data_fetch(lights, light.light))) {
    vcm::connect_native_surface(kg, &state, batch, light, vertices, merging, result);
    return;
  }
  VCMConnectionNode scratch[2 * (64 + 1)];
  vcm::connect_path(kg, &state, scratch, batch, light, vertices, merging, result);
}
CCL_NAMESPACE_END
