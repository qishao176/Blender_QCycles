/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: Apache-2.0 */

#pragma once

#include <array>

#include "kernel/types.h"

#include "util/vector.h"

CCL_NAMESPACE_BEGIN

struct LightgroupSplitPassDescriptor {
  const char *property_name;
  const char *pass_name_suffix;
  PassType pass_type;
};

using LightgroupSplitPassDescriptors = std::array<LightgroupSplitPassDescriptor, 16>;

const LightgroupSplitPassDescriptors &lightgroup_split_pass_descriptors();
int *lightgroup_pass_offset(KernelFilm &kernel_film, PassType pass_type);
void set_lightgroup_split_offset(vector<int> &offsets,
                                 int &row,
                                 int group_count,
                                 int group,
                                 int film_offset);

CCL_NAMESPACE_END
