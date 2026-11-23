/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: Apache-2.0 */

#pragma once

#include "kernel/types.h"

CCL_NAMESPACE_BEGIN

/* Material settings only filter caster events, never reception or ordinary BSDF
 * shading. The scene and object switches are checked by the owning estimator. */
ccl_device_inline bool material_caustics_allowed(KernelGlobals kg,
                                                const int shader,
                                                const int label)
{
  const uint flags = kernel_data_fetch(shaders, shader & SHADER_MASK).caustics_flags;
  return ((label & LABEL_REFLECT) && (flags & SHADER_CAUSTICS_REFLECT)) ||
         ((label & LABEL_TRANSMIT) && (flags & SHADER_CAUSTICS_TRANSMIT));
}

CCL_NAMESPACE_END
