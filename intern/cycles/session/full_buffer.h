/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: Apache-2.0 */

#pragma once

#include <memory>

#include "kernel/types.h"
#include "session/buffers.h"
#include "util/map.h"
#include "util/vector.h"

CCL_NAMESPACE_BEGIN

struct LPEProgram;

/* Host-owned deferred results: no Device or Scene pointer survives in this state.
 * The host retains it across Cycles Session recreation between view layers. */
struct FullBufferState {
  struct PhotonCacheResult {
    BufferParams params;
    vector<float> pixels;
    vector<int> offsets;
    vector<uint> merge;
    KernelFilm native_film;
  };
  map<string, PhotonCacheResult> photons;
  map<string, std::shared_ptr<const LPEProgram>> lpe_programs;

  void clear()
  {
    photons.clear();
    lpe_programs.clear();
  }
};

CCL_NAMESPACE_END
