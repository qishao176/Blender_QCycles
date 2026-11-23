/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

/** \file
 * \ingroup render
 */

#pragma once

#include <vector>
#include <map>
#include <string>

#include "IMB_deep_sample.hh"

namespace blender {

/**
 * Owned deep EXR sample storage for RenderResult and compositor access.
 */
struct RenderDeepData {
  std::vector<std::vector<DeepSample>> pixels;
  std::map<std::string, std::vector<std::vector<DeepSample>>> passes;
  bool has_display_window = false;
  int display_size[2] = {0, 0};
  int display_offset[2] = {0, 0};
  int data_offset[2] = {0, 0};
};

}  // namespace blender
