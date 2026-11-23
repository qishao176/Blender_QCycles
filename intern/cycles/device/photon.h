/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: Apache-2.0 */
#pragma once

#include "device/device.h"
#include <cstdlib>
#include <cstring>

CCL_NAMESPACE_BEGIN

/* Test-only: execute two independent replicas sequentially on one OptiX device.
 * Never duplicate the CUDA/OptiX device context itself. */
inline bool photon_test_two_devices(const DeviceInfo &info)
{
  const char *value = std::getenv("CYCLES_PHOTON_TEST_DEVICES");
  return info.type == DEVICE_OPTIX && info.multi_devices.empty() && value &&
         std::strcmp(value, "2") == 0;
}

inline bool photon_sample_parallel(const DeviceInfo &info)
{
  return info.multi_devices.size() > 1 || photon_test_two_devices(info);
}

/* Opt-in test of real concurrent devices, retaining the production same-model rule. */
inline bool photon_test_hybrid(const DeviceInfo &info)
{
  const char *value = std::getenv("CYCLES_PHOTON_TEST_HYBRID");
  if (!value || std::strcmp(value, "1") != 0 || info.multi_devices.size() != 2) {
    return false;
  }
  int cpus = 0, optix = 0;
  for (const DeviceInfo &device : info.multi_devices) {
    cpus += device.type == DEVICE_CPU;
    optix += device.type == DEVICE_OPTIX;
  }
  return cpus == 1 && optix == 1;
}

inline bool photon_device_supported(const DeviceInfo &info)
{
  if (info.multi_devices.empty()) {
    return info.type == DEVICE_CPU || info.type == DEVICE_CUDA || info.type == DEVICE_OPTIX;
  }
  if (photon_test_hybrid(info)) {
    return true;
  }
  const string &model = info.multi_devices.front().description;
  for (const DeviceInfo &device : info.multi_devices) {
    if (device.type != DEVICE_OPTIX || device.description != model) {
      return false;
    }
  }
  return true;
}

CCL_NAMESPACE_END
