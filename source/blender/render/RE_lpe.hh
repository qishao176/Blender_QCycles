/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

#pragma once

#include <string>
#include <vector>
#include <cstdint>
#include <array>
#include <map>
#include <memory>
#include <functional>

namespace blender {
struct Main;
struct Scene;
struct bNodeTree;
struct RenderResult;
struct ID;
struct RenderEngine;
struct ViewLayer;
struct wmWindow;
struct Render;

namespace render {

struct LPEGraphStatus {
  int lighting_states = 0;
  int mask_states = 0;
  /* Both modes allocate unique edit-path/light-instance buckets. Render Only
   * roots are omitted from the viewport graph unless an active Viewer needs them. */
  int render_edit_states = 0;
  int viewport_edit_states = 0;
  /* Same totals minus the chains that only feed render-only outputs. */
  int viewport_lighting_states = 0;
  int viewport_mask_states = 0;
  int native_states = 0;
  int photon_states = 0;
  bool independent_photon = false;
  int limit = 64;
  bool available = false;
  bool has_dynamic_edits = false;
  std::string error;
  std::vector<std::string> readonly_light_scopes;
  /* Edit scopes whose sampled lights have a near-zero color channel: cached
   * reconstruction is unreliable there and the user should fix the light.
   * The string is a ready-to-show hint naming the light and its weak channels. */
  std::vector<std::pair<std::string, std::string>> weak_light_entries;
  int viewer_node = 0;
  bool viewer_direct = false;

  int states() const { return lighting_states + mask_states + render_edit_states; }
  int viewport_states() const { return viewport_lighting_states + viewport_mask_states + viewport_edit_states; }
};

using LPEGraphEvaluator = LPEGraphStatus (*)(const bNodeTree &tree, const Scene *scene);
/* -1 needs dependency sync, 0 has no sessions, 1 fully rebound every session,
 * 2 preserved sampling but some managed requirements still need new data. */
using LPETreeUpdater = int (*)(bNodeTree &tree);
struct LPEApplyScope {
  int node = 0;
  std::string item;
  bool output = false;
};

struct LPEApplyChanges {
  std::vector<ID *> ids;
  bool node_tree_changed = false;
  uint64_t revision = 0;
};

struct LPEMixInput {
  const float *pixels = nullptr;
  int channels = 3;
};
struct LPEMixTerm {
  int input = 0;
  std::array<float, 9> matrix = {};
};
struct LPEMixOutput {
  std::string name;
  int channels = 3;
  int alpha_input = -1;
  std::array<float, 3> bias = {};
  std::vector<LPEMixTerm> terms;
};
class LPEGPUImage {
 public:
  virtual ~LPEGPUImage() = default;
  virtual void tag_inputs_changed() = 0;
  virtual bool evaluate(const std::vector<LPEMixInput> &inputs,
                        const std::vector<LPEMixOutput> &outputs,
                        std::map<std::string, std::vector<float>> &pixels,
                        const std::function<bool()> &cancel = {},
                        const std::function<void(float)> &progress = {}) = 0;
};
std::unique_ptr<LPEGPUImage> lpe_gpu_image_create(int width, int height);

using LPELightApplier = bool (*)(Main &, bNodeTree &, const LPEApplyScope &,
                                  LPEApplyChanges &, std::string &);
}  // namespace render

void RE_lpe_light_applier_set(render::LPELightApplier applier);
void RE_lpe_tree_updater_set(render::LPETreeUpdater updater);
bool RE_lpe_tree_update(Main &bmain, bNodeTree &tree);
bool RE_lpe_update_viewport(bNodeTree &tree);
bool RE_lpe_edit_apply(Main &bmain, Scene &scene, bNodeTree &tree,
                       const render::LPEApplyScope &scope, std::string &error);
bool RE_lpe_has_readonly_lights(const bNodeTree &tree, const render::LPEApplyScope &scope);
std::string RE_lpe_weak_light_message(const bNodeTree &tree, const render::LPEApplyScope &scope);
bool RE_lpe_has_weak_light_channel(const bNodeTree &tree, const render::LPEApplyScope &scope);
void RE_lpe_graph_evaluator_set(render::LPEGraphEvaluator evaluator);
render::LPEGraphStatus RE_lpe_graph_status(const bNodeTree &tree, const Scene *scene = nullptr);
}  // namespace blender
