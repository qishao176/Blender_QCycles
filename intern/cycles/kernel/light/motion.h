/* SPDX-FileCopyrightText: 2026 Blender Foundation
 * SPDX-License-Identifier: Apache-2.0 */
#pragma once
#include "kernel/geom/object.h"
CCL_NAMESPACE_BEGIN
/* Analytic lamps keep their center-frame representation. Rigid motion preserves
 * area, solid angle and ray distance, so sampling and evaluation share this frame. */
ccl_device_inline bool light_motion_transform(KernelGlobals kg,
                                              const ccl_global KernelLight *light,
                                              const float time,
                                              ccl_private Transform *to_world,
                                              ccl_private Transform *to_center)
{
#ifdef __OBJECT_MOTION__
  if (light->use_rigid_motion) {
    const Transform motion = object_fetch_transform_motion(kg, light->object_id, time);
    const Transform inverse = transform_inverse(motion);
    *to_world = motion * object_fetch_transform(kg, light->object_id, OBJECT_INVERSE_TRANSFORM);
    *to_center = object_fetch_transform(kg, light->object_id, OBJECT_TRANSFORM) * inverse;
    return true;
  }
#endif
  return false;
}
ccl_device_inline float3 light_motion_direction_to_center(KernelGlobals kg,
                                                          const ccl_global KernelLight *light,
                                                          const float time,
                                                          const float3 D)
{
  Transform world, center;
  return light_motion_transform(kg, light, time, &world, &center) ?
             transform_direction(&center, D) :
             D;
}
CCL_NAMESPACE_END
