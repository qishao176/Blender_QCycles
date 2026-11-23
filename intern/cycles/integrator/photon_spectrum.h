/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: Apache-2.0 */
#pragma once

#include "kernel/sample/photon_spectrum.h"
#include <cstdlib>
#include <cstring>

CCL_NAMESPACE_BEGIN

/* Host-only diagnostic controls. Keep all device code identical while comparing
 * spectral stratification against the thinner per-group spatial photon density. */
inline uint photon_wavelength_groups_from_environment()
{
  if (getenv("CYCLES_PHOTON_WAVELENGTH_SINGLE") || getenv("CYCLES_PHOTON_WAVELENGTH_LEGACY")) {
    return 1u;
  }
  const char *value = getenv("CYCLES_PHOTON_WAVELENGTH_GROUPS");
  if (value) {
    if (std::strcmp(value, "1") == 0) return 1u;
    if (std::strcmp(value, "2") == 0) return 2u;
    if (std::strcmp(value, "4") == 0) return 4u;
    if (std::strcmp(value, "8") == 0) return 8u;
  }
  return PHOTON_WAVELENGTH_GROUPS;
}

CCL_NAMESPACE_END
