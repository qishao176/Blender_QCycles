/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: Apache-2.0 */
#pragma once

#include "util/defines.h"

CCL_NAMESPACE_BEGIN

/* The first word is the existing emission counter. The other words are used only
 * by the viewport probe on selected maps, sharing the normal map publication fence. */
enum PhotonMapCounter {
  PHOTON_MAP_STORED,
  PHOTON_PROBE_ENABLED,
  PHOTON_PROBE_SURFACES,
  PHOTON_PROBE_ISOLATED,
  PHOTON_PROBE_WIDER_HELPED,
  PHOTON_PROBE_IN_FRAME,
  PHOTON_PROBE_SUBPIXEL,
  PHOTON_PROBE_SUBPIXEL_RADIUS_LIMITED,
  PHOTON_PROBE_PIXEL_SCALE_SUM,
  PHOTON_PROBE_VOLUME_STORED,
  PHOTON_MAP_COUNTERS
};

CCL_NAMESPACE_END
