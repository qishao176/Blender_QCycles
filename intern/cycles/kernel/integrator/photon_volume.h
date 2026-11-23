/* SPDX-FileCopyrightText: 2026 Blender Foundation
 * SPDX-License-Identifier: Apache-2.0 */
#pragma once
CCL_NAMESPACE_BEGIN
#if defined(__PHOTON_CAUSTICS__) && defined(__VOLUME__)
ccl_device void photon_caustics_volume_gather(KernelGlobals kg,
                                              ConstIntegratorState state,
                                              const ccl_private ShaderData *sd,
                                              const ccl_private ShaderVolumePhases *phases,
                                              ccl_global float *render_buffer);
#endif
CCL_NAMESPACE_END
