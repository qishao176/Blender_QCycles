/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: Apache-2.0 */

#include "blender/device.h"
#include "blender/lpe.h"
#include "blender/session.h"
#include "blender/sync.h"
#include "integrator/path_trace.h"
#include "integrator/path_trace_tile.h"
#include "scene/film.h"
#include "scene/camera.h"
#include "scene/integrator.h"
#include "scene/image.h"
#include "util/scoped_defer.h"
#include "BKE_layer.hh"
#include "DEG_depsgraph.hh"
#include "DEG_depsgraph_build.hh"
#include "DEG_depsgraph_query.hh"
#include "DNA_layer_types.h"
#include "DNA_node_types.h"
#include "DNA_scene_types.h"
#include "BKE_node_runtime.hh"

#include <algorithm>
#include <map>
#include <mutex>
#include <set>
#include <cstring>
#include <Python.h>

CCL_NAMESPACE_BEGIN
namespace {
std::mutex sessions_mutex;
std::map<blender::RenderEngine *, BlenderSession *> sessions;
}

void register_lpe_session(BlenderSession &session)
{
  const std::lock_guard lock(sessions_mutex);
  sessions[&session.b_engine] = &session;
}

void unregister_lpe_session(BlenderSession &session)
{
  const std::lock_guard lock(sessions_mutex);
  sessions.erase(&session.b_engine);
}

void select_lpe_viewport(void *space, const string &name, const bool overlay, const string &status)
{
  const std::lock_guard lock(sessions_mutex);
  for (const auto &[engine, owner] : sessions) {
    if (owner->b_v3d != space || !owner->session || !owner->scene || owner->background) continue;
    if (blender_scene_uses_vcm(*owner->b_scene)) continue;
    owner->viewport_pass_status = status;
    Film &film = *owner->scene->film;
    const string previous = film.get_lpe_display_output();
    if (previous == name) {
      if (owner->sync && owner->sync->viewport_pass_resample_pending) owner->tag_update();
      continue;
    }
    /* A deleted Viewer is already absent from the catalog. Decide using the destination's
     * actual buffers, so removing an observer always restores the sampled pass in place. */
    const auto program = film.get_lpe_program();
    const bool available = program && (program->edit_previews.contains(name) ||
        std::any_of(program->outputs.begin(), program->outputs.end(),
                    [&](const auto &output) { return output.name == name; }));
    const bool cached_switch = available || (owner->sync &&
                                             owner->sync->lpe_viewport_native_output_is_sampled(name));
    film.set_lpe_display_output(name);
    if (cached_switch && owner->sync && !owner->sync->viewport_pass_resample_pending) {
      configure_lpe_preview(film, film.get_lpe_program(), true);
      owner->session->queue_display_plan({});
    }
    else if (owner->sync) {
      /* Display redraw alone never runs sync_film or resets the sample scheduler.
       * Route unsampled native/uncached changes through the normal locked scene sync. */
      if (overlay) {
        /* The UI can see a new Viewer before its graph has been rebound. Let sync_lpe
         * decide whether it needs additional buffers instead of invalidating the film. */
        owner->sync->tag_update();
      }
      else {
        owner->sync->request_viewport_pass_resample();
      }
      owner->tag_update();
    }
    owner->tag_redraw();
  }
}

int update_lpe_sessions(blender::bNodeTree &tree)
{
  if (getenv("LPE_TRACE")) fprintf(stderr, "[LPE] updater enter\n");
  const std::lock_guard lock(sessions_mutex);
  tree.ensure_topology_cache();
  int handled = 0;
  for (const auto &[engine, owner] : sessions) {
    if (!owner->session || !owner->sync || owner->background ||
        owner->sync->lpe_synced_tree != tree.id.session_uid || blender_scene_uses_vcm(*owner->b_scene))
      continue;
    /* Compile RNA on the main thread, then publish immutable image plans. Do not
     * wait for scene/buffer locks here: a render batch can be waiting for the
     * main thread's graphics context while holding the buffer lock. Sampling
     * requirements are checked against the published plan before accepting. */
    string error;
    LPEGraph graph = compile_lpe_graph(tree, error, true, owner->sync->lpe_viewport_sampled_output());
    if (!error.empty()) {
      /* Invalid graphs cannot be rebound, even while sampling is paused. Use
       * the normal session error channel; the main-thread UI exits rendered
       * shading after its callback returns, without taking render locks here. */
      owner->session->progress.set_error("LPE: " + error);
      owner->tag_redraw();
      handled = 1;
      continue;
    }
    if (owner->sync->lpe_synced_revision != blender::DEG_get_render_data_update_count()) {
      if (getenv("LPE_TRACE")) fprintf(stderr, "[LPE] updater rev-mismatch -> -1\n");
      return -1;
    }
    if (lpe_resident_keys(graph) != owner->sync->lpe_synced_resident_keys ||
        !lpe_can_reuse_sampling(graph, owner->sync->lpe_synced_sampling_keys)) {
      if (getenv("LPE_TRACE")) fprintf(stderr, "[LPE] updater resident paths changed -> sync\n");
      return -1;
    }
    bool complete = false;
    const bool updated = error.empty() &&
        update_lpe_preview_plan(*owner->scene->film, std::move(graph), tree, &complete);
    if (getenv("LPE_TRACE")) fprintf(stderr, "[LPE] updater plan updated=%d complete=%d\n", int(updated), int(complete));
    if (!updated) return -1;
    owner->session->queue_display_plan({});
    owner->tag_redraw();
    handled = !complete || handled == 2 ? 2 : 1;
  }
  return handled;
}

CCL_NAMESPACE_END
