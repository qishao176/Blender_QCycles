/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

#include "RE_lpe.hh"
#include "RE_pipeline.h"
#include "RE_engine.h"
#include "render_result.h"
#include "render_types.h"

#include "BKE_global.hh"
#include "BKE_main.hh"
#include "BKE_image.hh"
#include "BKE_node_runtime.hh"
#include "BKE_node_tree_update.hh"
#include "BKE_main_invariants.hh"
#include "BKE_screen.hh"
#include "BKE_scene.hh"
#include "BLI_listbase.hh"
#include "BLI_string.hh"
#include "BLI_rect.hh"
#include "BLT_translation.hh"
#include "DNA_image_types.h"
#include "DNA_scene_types.h"
#include "DNA_layer_types.h"
#include "DNA_screen_types.h"
#include "DNA_space_types.h"
#include "DNA_view3d_types.h"
#include "DNA_windowmanager_types.h"
#include "IMB_imbuf_types.hh"
#include "DEG_depsgraph.hh"
#include "WM_api.hh"
#include "WM_types.hh"

#include <map>
#include <mutex>
#include <algorithm>

namespace blender {

namespace {
std::mutex graph_status_mutex;
render::LPEGraphEvaluator graph_evaluator = nullptr;
render::LPETreeUpdater tree_updater = nullptr;
thread_local bool apply_in_progress = false;
}  // namespace


void RE_lpe_tree_updater_set(render::LPETreeUpdater updater) { tree_updater = updater; }
static render::LPELightApplier light_applier = nullptr;
void RE_lpe_light_applier_set(render::LPELightApplier applier) { light_applier = applier; }
bool RE_lpe_tree_update(Main & /*bmain*/, bNodeTree &tree)
{
  return !G.is_rendering && !apply_in_progress && tree_updater && tree_updater(tree) > 0;
}

void RE_lpe_graph_evaluator_set(render::LPEGraphEvaluator evaluator)
{
  const std::lock_guard lock(graph_status_mutex);
  graph_evaluator = evaluator;
}

render::LPEGraphStatus RE_lpe_graph_status(const bNodeTree &tree, const Scene *scene)
{
  const std::lock_guard lock(graph_status_mutex);
  if (!graph_evaluator || tree.type != NTREE_LPE) {
    return {};
  }
  /* Compile the current graph on demand. RNA edits, collection membership and
   * instance changes must not depend on a viewport/depsgraph redraw revision. */
  return graph_evaluator(tree, scene);
}

bool RE_lpe_update_viewport(bNodeTree &tree)
{
  /* A partial plan preserves existing pixels while missing paths await sampling.
   * Only a complete rebind may satisfy the Update Lighting action. */
  return tree_updater && tree_updater(tree) == 1;
}

bool RE_lpe_has_readonly_lights(const bNodeTree &tree, const render::LPEApplyScope &scope)
{
  const auto status = RE_lpe_graph_status(tree);
  const std::string key = std::string(scope.output ? "O:" : "M:") +
                          std::to_string(scope.node) + ":" + scope.item;
  return std::find(status.readonly_light_scopes.begin(), status.readonly_light_scopes.end(), key) !=
         status.readonly_light_scopes.end();
}

std::string RE_lpe_weak_light_message(const bNodeTree &tree, const render::LPEApplyScope &scope)
{
  const auto status = RE_lpe_graph_status(tree);
  const std::string key = std::string(scope.output ? "O:" : "M:") +
                          std::to_string(scope.node) + ":" + scope.item;
  for (const auto &[entry_key, message] : status.weak_light_entries) {
    if (entry_key == key)
      return message;
  }
  return {};
}

bool RE_lpe_has_weak_light_channel(const bNodeTree &tree, const render::LPEApplyScope &scope)
{
  return !RE_lpe_weak_light_message(tree, scope).empty();
}

bool RE_lpe_edit_apply(Main &bmain, Scene & /*scene*/, bNodeTree &tree,
                       const render::LPEApplyScope &scope, std::string &error)
{
  if (G.is_rendering || !light_applier) {
    error = "Light editing is unavailable while rendering";
    return false;
  }
  render::LPEApplyChanges changes;
  if (!light_applier(bmain, tree, scope, changes, error)) return false;
  apply_in_progress = true;
  for (ID *id : changes.ids) {
    DEG_id_tag_update(id, 0);
    WM_main_add_notifier(NC_LAMP | ND_LIGHTING, id);
  }
  if (changes.node_tree_changed) {
    BKE_ntree_update_tag_all(&tree);
    BKE_main_ensure_invariants(bmain, tree.id);
  }
  apply_in_progress = false;
  WM_main_add_notifier(NC_NODE | NA_EDITED, &tree);
  return true;
}
}  // namespace blender
