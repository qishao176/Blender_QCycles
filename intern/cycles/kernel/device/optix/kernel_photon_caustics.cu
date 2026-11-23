/* SPDX-FileCopyrightText: 2026 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#include "kernel/device/optix/compat.h"
#include "kernel/device/optix/globals.h"

#include "kernel/device/gpu/image.h"

#include "kernel/geom/primitive.h"
#include "kernel/integrator/photon_caustics.h"

extern "C" __global__ void __raygen__kernel_optix_integrator_photon_caustics_emit()
{
  /* This prepass borrows the main-path SoA slots before camera paths are initialized.
   * Its ABI uses the launch size and shared integrator state, not path_index_array. */
  const uint photon_index = optixGetLaunchIndex().x;
  if (photon_index < kernel_integrator_state.photon_caustics_capacity) {
    integrator_photon_caustics_emit(
        nullptr, photon_index, photon_index, kernel_integrator_state.photon_caustics_iteration);
  }
}
