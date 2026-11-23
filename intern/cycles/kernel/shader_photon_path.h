/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: Apache-2.0 */

#pragma once

#include "kernel/types.h"

CCL_NAMESPACE_BEGIN

/* Light-side state lives only in the emitter invocation, never in the camera
 * wavefront queues or the photon map. Lengths exclude portal teleportation. */
enum NodePhotonPath {
  NODE_PP_IS_PHOTON,
  NODE_PP_IS_SINGULAR,
  NODE_PP_LENGTH,
  NODE_PP_DEPTH,
  NODE_PP_DIFFUSE_LENGTH,
  NODE_PP_DIFFUSE_DEPTH,
  NODE_PP_GLOSSY_LENGTH,
  NODE_PP_GLOSSY_DEPTH,
  NODE_PP_REFLECTION_LENGTH,
  NODE_PP_REFLECTION_DEPTH,
  NODE_PP_TRANSMISSION_LENGTH,
  NODE_PP_TRANSMISSION_DEPTH,
  NODE_PP_VOLUME_LENGTH,
  NODE_PP_VOLUME_DEPTH,
  NODE_PP_VOLUME_LAYERS,
  NODE_PP_PORTAL_DEPTH,
  NODE_PP_NUM_OUTPUTS,
};

struct PhotonShaderPath {
  float length, diffuse_length, glossy_length, reflection_length, transmission_length,
      volume_length;
  int depth, diffuse_depth, glossy_depth, reflection_depth, transmission_depth, volume_depth;
  int volume_layers, portal_depth;
  int last_label;
  bool in_volume, emission;
  float3 origin;
};

/* Class lengths describe distance following the corresponding surface event.
 * Glossy depth matches Cycles' glossy-reflection counter; transmission is separate.
 * Volume length is actual travel inside participating media, not the distance
 * following a scatter. Overlapping volumes count this physical distance once. */
ccl_device_inline void photon_shader_path_advance(ccl_private PhotonShaderPath *p, const float d)
{
  p->length += d;
  if (p->last_label & LABEL_REFLECT) {
    p->reflection_length += d;
    if (p->last_label & LABEL_DIFFUSE)
      p->diffuse_length += d;
    else
      p->glossy_length += d;
  }
  else if (p->last_label & LABEL_TRANSMIT)
    p->transmission_length += d;
  if (p->in_volume)
    p->volume_length += d;
}

ccl_device_inline float photon_shader_path_value(const ccl_private PhotonShaderPath *p,
                                                 const float3 position,
                                                 const int output)
{
  if (!p)
    return 0.0f;
  const float d = p->emission ? 0.0f : distance(position, p->origin);
  switch (output) {
    case NODE_PP_IS_PHOTON:
      return 1.0f;
    case NODE_PP_IS_SINGULAR:
      return (p->last_label & LABEL_SINGULAR) != 0;
    case NODE_PP_LENGTH:
      return p->length + d;
    case NODE_PP_DEPTH:
      return float(p->depth);
    case NODE_PP_DIFFUSE_LENGTH:
      return p->diffuse_length +
             ((p->last_label & LABEL_REFLECT) && (p->last_label & LABEL_DIFFUSE) ? d : 0.0f);
    case NODE_PP_DIFFUSE_DEPTH:
      return float(p->diffuse_depth);
    case NODE_PP_GLOSSY_LENGTH:
      return p->glossy_length +
             ((p->last_label & LABEL_REFLECT) && !(p->last_label & LABEL_DIFFUSE) ? d : 0.0f);
    case NODE_PP_GLOSSY_DEPTH:
      return float(p->glossy_depth);
    case NODE_PP_REFLECTION_LENGTH:
      return p->reflection_length + ((p->last_label & LABEL_REFLECT) ? d : 0.0f);
    case NODE_PP_REFLECTION_DEPTH:
      return float(p->reflection_depth);
    case NODE_PP_TRANSMISSION_LENGTH:
      return p->transmission_length + ((p->last_label & LABEL_TRANSMIT) ? d : 0.0f);
    case NODE_PP_TRANSMISSION_DEPTH:
      return float(p->transmission_depth);
    case NODE_PP_VOLUME_LENGTH:
      return p->volume_length + (p->in_volume ? d : 0.0f);
    case NODE_PP_VOLUME_DEPTH:
      return float(p->volume_depth);
    case NODE_PP_VOLUME_LAYERS:
      return float(p->volume_layers);
    case NODE_PP_PORTAL_DEPTH:
      return float(p->portal_depth);
  }
  return 0.0f;
}

CCL_NAMESPACE_END
