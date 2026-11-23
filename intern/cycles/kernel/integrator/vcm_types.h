/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: Apache-2.0 */

#pragma once

#include "util/types.h"

CCL_NAMESPACE_BEGIN

/* Static triangle surfaces in the first path replay implementation. All PDFs
 * are conditional on their preceding vertex. Flux has not been divided by path count. */
/* Private vertex flags kept separate from the Cycles event label bits. */
enum VCMVertexFlag {
  VCM_VERTEX_MOTION_TRIANGLE = 1 << 26,
  VCM_VERTEX_RECEIVER = 1 << 27,
  VCM_VERTEX_SPECTRAL = 1 << 28,
};

struct VCMVertex {
  packed_float3 position, normal, incoming, outgoing, flux;
  float u, v;
  float pdf_forward, pdf_reverse;
  float eta;
  packed_float3 shading_normal;
  int object, prim, shader, label;
};

enum VCMPathStatus {
  VCM_PATH_ESCAPE = 1,
  VCM_PATH_DEPTH_LIMIT,
  VCM_PATH_ABSORBED,
  VCM_PATH_UNSUPPORTED,
  VCM_PATH_CACHE_MISS,
};

struct VCMLightPath {
  uint index, round, count, status;
  int light;
  packed_float3 source, direction, flux;
  float light_pdf, area_pdf, direction_pdf;
};

struct VCMBatch {
  uint first, round, seed, max_depth;
  int guide_light = -1;
  float guide_fraction = 0.0f;
  packed_float3 guide_target = {};
  float guide_sigma = 0.0f;
  bool native_bsdf = false;
};

enum VCMTechnique {
  VCM_CAMERA_HIT,
  VCM_NEXT_EVENT,
  VCM_PAIRED,
  VCM_CAMERA_SPLAT,
  VCM_SURFACE_MERGE,
  VCM_TECHNIQUES
};

struct VCMColor {
  double values[3];
  ccl_device_inline_method double &operator[](int i)
  {
    return values[i];
  }
  ccl_device_inline_method double operator[](int i) const
  {
    return values[i];
  }
};

struct VCMConnectionNode {
  float3 p, n, ns;
  double incident_projection = 1.0;
  VCMColor beta = {1, 1, 1}, albedo = {};
  int object = -1, prim = -1;
  int source_light = -2;
  bool delta = false;
};

struct VCMGuideMoments {
  double weight, squared_weight, second, first[3];
};

struct VCMLaunch {
  VCMBatch batch;
  VCMLightPath *paths;
  VCMVertex *vertices;
  /* Resident film mode. A null grid selects the original emission capture ABI. */
  uint *heads = nullptr, *links = nullptr;
  uint bucket_mask = 0;
  double radius = 0, eta = 0;
  VCMConnectionNode *scratch = nullptr;
  float *film = nullptr, *accumulate = nullptr, *adaptive_sum = nullptr;
  uint *status = nullptr;
  VCMGuideMoments *moments = nullptr;
  const int *source_routes = nullptr;
  bool connect = false;
};

static_assert(sizeof(VCMVertex) == 108);
static_assert(sizeof(VCMLightPath) == 68);
static_assert(sizeof(VCMBatch) == 44);

CCL_NAMESPACE_END
