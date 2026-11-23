/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: GPL-2.0-or-later */

#pragma once

#include "gpu_shader_material_interface.bsl.hh"

[[node]]
void node_photon_path(const ShadingData &sd,
                      float &is_photon_ray,
                      float &is_singular_ray,
                      float &photon_ray_length,
                      float &photon_depth,
                      float &diffuse_length,
                      float &diffuse_depth,
                      float &glossy_length,
                      float &glossy_depth,
                      float &reflection_length,
                      float &reflection_depth,
                      float &transmission_length,
                      float &transmission_depth,
                      float &volume_length,
                      float &volume_depth,
                      float &volume_layers,
                      float &portal_depth,
                      float &is_camera_ray)
{
  /* EEVEE has no light-side photon tracing. */
  is_photon_ray = 0.0f;
  is_singular_ray = 0.0f;
  photon_ray_length = 0.0f;
  photon_depth = 0.0f;
  diffuse_length = 0.0f;
  diffuse_depth = 0.0f;
  glossy_length = 0.0f;
  glossy_depth = 0.0f;
  reflection_length = 0.0f;
  reflection_depth = 0.0f;
  transmission_length = 0.0f;
  transmission_depth = 0.0f;
  volume_length = 0.0f;
  volume_depth = 0.0f;
  volume_layers = 0.0f;
  portal_depth = 0.0f;
  is_camera_ray = float(sd.ray_type == RAY_TYPE_CAMERA);
}
