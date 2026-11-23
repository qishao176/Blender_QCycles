/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: Apache-2.0 */

#pragma once

#include "scene/lpe.h"
#include "session/output_driver.h"
#include <memory>
#include <functional>
#include <set>

namespace blender {
struct bNode;
struct bNodeSocket;
struct bNodeTree;
struct Main;
struct Scene;
struct ViewLayer;
struct RenderResult;
struct RenderEngine;
namespace render {
struct LPEGraphStatus;
struct LPEApplyScope;
struct LPEApplyChanges;
}  // namespace render
}  // namespace blender

CCL_NAMESPACE_BEGIN

class BlenderSession;
class Camera;
class Film;
struct LPEGraph;
struct LPEPreviewCache;
std::shared_ptr<LPEPreviewCache> create_lpe_preview_cache();
void register_lpe_session(BlenderSession &session);
void unregister_lpe_session(BlenderSession &session);
int update_lpe_sessions(blender::bNodeTree &tree);
void select_lpe_viewport(void *space, const string &name, bool overlay, const string &status);
bool apply_lpe_lights(blender::Main &, blender::bNodeTree &,
                      const blender::render::LPEApplyScope &,
                      blender::render::LPEApplyChanges &, string &);
size_t stateful_edit_count(const LPEGraph &graph);
std::set<string> lpe_resident_keys(const LPEGraph &graph);
std::set<string> lpe_sampling_keys(const LPEGraph &graph);
bool lpe_can_reuse_sampling(const LPEGraph &graph, const std::set<string> &sampled_keys);
bool lpe_paths_disjoint(const std::vector<const blender::bNodeSocket *> &a,
                        const std::vector<const blender::bNodeSocket *> &b);
bool rebind_lpe_program(LPEGraph &graph, const LPEProgram &sampled,
                       const blender::bNodeTree &tree, string &error,
                       bool clamp_missing = false);
bool evaluate_lpe_preview(const LPEProgram &program, const OutputDriver::Tile &tile,
                          std::vector<float> &pixels, LPEPreviewCache *cache = nullptr,
                          int expression = -1);
void configure_lpe_preview(Film &film, std::shared_ptr<const LPEProgram> program, bool enabled);
bool update_lpe_preview_plan(Film &film, LPEGraph graph, const blender::bNodeTree &tree,
                             bool *complete = nullptr);

struct LPEEditRequest {
  std::vector<const blender::bNodeSocket *> path;
  int expression = -1;
  int manager_node = 0;
  string manager_item;
  std::vector<int> operation_nodes;
  int output_node = 0;
  string output_item;
  std::vector<int> scene_operation_nodes;
};

struct LPEGraph {
  /* Expressions lowered from Always outputs and Light Edit, before the Viewer. */
  int resident_expressions = 0;
  bool shadow_catcher = false;
  std::shared_ptr<LPEProgram> program = std::make_shared<LPEProgram>();
  std::vector<std::vector<const blender::bNodeSocket *>> paths;
  /* Union alternatives share a source and one accepting state/raw pass. */
  std::vector<std::vector<std::vector<const blender::bNodeSocket *>>> unions;
  std::vector<std::pair<std::vector<const blender::bNodeSocket *>, int>> masks;
  std::vector<LPEEditRequest> edits;
  std::vector<std::vector<const blender::bNodeSocket *>> shadows;
};

/* Only the selected Render Only output becomes a temporary preview root. */
LPEGraph compile_lpe_graph(const blender::bNodeTree &tree, string &error, bool preview = false,
                          const string &preview_output = "", bool vcm_sources_only = false);
std::vector<std::pair<string, bool>> lpe_viewport_catalog(const blender::bNodeTree &tree);

/* The current device matcher stores its active states in a uint64_t. */
constexpr size_t LPE_MAX_STATES = 64;
blender::render::LPEGraphStatus lpe_graph_status(const blender::bNodeTree &tree,
                                                 const blender::Scene *scene = nullptr);
bool lpe_independent_photon_budget(const blender::Scene &scene, bool preview, bool deep_output);
bool validate_lpe_state_budget(const LPEGraph &graph,
                               const char *view_layer_name,
                               string &error,
                               size_t edit_states = 0,
                               bool independent_photon = false);

std::map<string, std::vector<float>> evaluate_lpe_outputs(const LPEProgram &program,
                                                          const OutputDriver::Tile &tile);

bool update_lpe_edit_terms(LPEProgram &program, const blender::bNodeTree &tree, string &error,
                           bool clamp_missing = false);
bool evaluate_lpe_light_target(const LPEEditTerm &term, const blender::bNodeTree &tree,
                                LPELightParameters &target, string &error);
float3 lpe_light_emission(const LPELightParameters &parameters);
bool evaluate_lpe_combined(const LPEProgram &program, const OutputDriver::Tile &tile,
                           std::vector<float> &pixels);
CCL_NAMESPACE_END
