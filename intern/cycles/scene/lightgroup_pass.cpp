/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: Apache-2.0 */

#include "scene/lightgroup_pass.h"

CCL_NAMESPACE_BEGIN

#define LIGHTGROUP_SPLIT_PASS_POLICY(M) \
  M("use_pass_diffuse_combined", \
    "Diff", \
    PASS_DIFFUSE, \
    pass_lightgroup_diffuse) \
  M("use_pass_diffuse_direct", \
    "Diff_Dir", \
    PASS_DIFFUSE_DIRECT, \
    pass_lightgroup_diffuse_direct) \
  M("use_pass_diffuse_indirect", \
    "Diff_Ind", \
    PASS_DIFFUSE_INDIRECT, \
    pass_lightgroup_diffuse_indirect) \
  M("use_pass_glossy_combined", "Glos", PASS_GLOSSY, pass_lightgroup_glossy) \
  M("use_pass_glossy_direct", \
    "Glos_Dir", \
    PASS_GLOSSY_DIRECT, \
    pass_lightgroup_glossy_direct) \
  M("use_pass_glossy_indirect", \
    "Glos_Ind", \
    PASS_GLOSSY_INDIRECT, \
    pass_lightgroup_glossy_indirect) \
  M("use_pass_transmission_combined", \
    "Trans", \
    PASS_TRANSMISSION, \
    pass_lightgroup_transmission) \
  M("use_pass_transmission_direct", \
    "Trans_Dir", \
    PASS_TRANSMISSION_DIRECT, \
    pass_lightgroup_transmission_direct) \
  M("use_pass_transmission_indirect", \
    "Trans_Ind", \
    PASS_TRANSMISSION_INDIRECT, \
    pass_lightgroup_transmission_indirect) \
  M("use_pass_subsurface_combined", \
    "SSS", \
    PASS_SUBSURFACE, \
    pass_lightgroup_subsurface) \
  M("use_pass_subsurface_direct", \
    "SSS_Dir", \
    PASS_SUBSURFACE_DIRECT, \
    pass_lightgroup_subsurface_direct) \
  M("use_pass_subsurface_indirect", \
    "SSS_Ind", \
    PASS_SUBSURFACE_INDIRECT, \
    pass_lightgroup_subsurface_indirect) \
  M("use_pass_volume_combined", "Vol", PASS_VOLUME, pass_lightgroup_volume) \
  M("use_pass_volume_direct", \
    "Vol_Dir", \
    PASS_VOLUME_DIRECT, \
    pass_lightgroup_volume_direct) \
  M("use_pass_volume_indirect", \
    "Vol_Ind", \
    PASS_VOLUME_INDIRECT, \
    pass_lightgroup_volume_indirect) \
  M("use_pass_photon", "Photon", PASS_CAUSTICS, pass_lightgroup_photon)

const LightgroupSplitPassDescriptors &lightgroup_split_pass_descriptors()
{
#define LIGHTGROUP_DESCRIPTOR(property_name, pass_name_suffix, pass_type, kernel_offset) \
  LightgroupSplitPassDescriptor{property_name, pass_name_suffix, pass_type},
  static const LightgroupSplitPassDescriptors descriptors = {
      LIGHTGROUP_SPLIT_PASS_POLICY(LIGHTGROUP_DESCRIPTOR)};
#undef LIGHTGROUP_DESCRIPTOR
  return descriptors;
}

int *lightgroup_pass_offset(KernelFilm &kernel_film, const PassType pass_type)
{
  if (pass_type == PASS_COMBINED) {
    return &kernel_film.pass_lightgroup;
  }

  switch (pass_type) {
#define LIGHTGROUP_OFFSET_CASE(property_name, pass_name_suffix, type, kernel_offset) \
  case type: \
    return &kernel_film.kernel_offset;
    LIGHTGROUP_SPLIT_PASS_POLICY(LIGHTGROUP_OFFSET_CASE)
#undef LIGHTGROUP_OFFSET_CASE
    default:
      return nullptr;
  }
}

void set_lightgroup_split_offset(vector<int> &offsets,
                                 int &row,
                                 const int group_count,
                                 const int group,
                                 const int film_offset)
{
  if (row == PASS_UNUSED) {
    row = int(offsets.size());
    offsets.resize(offsets.size() + group_count, PASS_UNUSED);
  }
  offsets[row + group] = film_offset;
}

#undef LIGHTGROUP_SPLIT_PASS_POLICY

CCL_NAMESPACE_END
