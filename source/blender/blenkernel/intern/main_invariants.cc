/* SPDX-FileCopyrightText: 2024 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

/** \file
 * \ingroup bke
 */

#include "BKE_main.hh"
#include "BKE_main_invariants.hh"
#include "BKE_node.hh"
#include "BKE_node_tree_update.hh"
#include "BLI_listbase.hh"
#include "BLI_set.hh"

#include "DEG_depsgraph.hh"

#include "DNA_layer_types.h"
#include "DNA_node_types.h"
#include "DNA_scene_types.h"

#include "WM_api.hh"
#include "WM_types.hh"
#include <optional>

namespace blender {

static void send_notifiers_after_node_tree_change(ID *id, bNodeTree *ntree)
{
  WM_main_add_notifier(NC_NODE | NA_EDITED, id);

  if (ntree->type == NTREE_SHADER && id != nullptr) {
    if (GS(id->name) == ID_MA) {
      WM_main_add_notifier(NC_MATERIAL | ND_SHADING, id);
    }
    else if (GS(id->name) == ID_LA) {
      WM_main_add_notifier(NC_LAMP | ND_LIGHTING, id);
    }
    else if (GS(id->name) == ID_WO) {
      WM_main_add_notifier(NC_WORLD | ND_WORLD, id);
    }
  }
  else if (ntree->type == NTREE_COMPOSIT) {
    /* The notifier category `NC_SCENE` has some special handling in #wm_event_do_notifiers. When
     * the id that is passed as the reference is not a scene or not the active scene/sequencer
     * scene, then the notifier is dropped. So we need to pass `nullptr` here to make sure the
     * notifier is sent. */
    WM_main_add_notifier(NC_SCENE | ND_NODES, nullptr);
    WM_main_add_notifier(NC_SCENE | ND_COMPO_RESULT, nullptr);
  }
  else if (ntree->type == NTREE_TEXTURE) {
    WM_main_add_notifier(NC_TEXTURE | ND_NODES, id);
  }
  else if (ntree->type == NTREE_GEOMETRY) {
    WM_main_add_notifier(NC_OBJECT | ND_MODIFIER, id);
    if (ntree->geometry_node_asset_traits) {
      if (ntree->geometry_node_asset_traits->flag & GEO_NODE_ASSET_TOOL) {
        /* Notifier to re-register node group operators. */
        WM_main_add_notifier(NC_NODE | ND_NODE_ASSET_DATA, id);
      }
    }
  }
}

static void propagate_node_tree_changes(Main &bmain, const std::optional<Span<ID *>> modified_ids)
{
  NodeTreeUpdateExtraParams params;
  Set<Scene *> lpe_scenes;
  Set<Scene *> lpe_sampling_scenes;
  Set<bNodeTree *> cached_preview_trees;
  params.tree_changed_fn = [&](bNodeTree &ntree, ID &owner_id) {
    send_notifiers_after_node_tree_change(&owner_id, &ntree);
    const bool cached = ntree.typeinfo->update_cached_preview &&
                        ntree.typeinfo->update_cached_preview(bmain, ntree);
    if (cached) {
      cached_preview_trees.add(&ntree);
    }
    /* Keep evaluated trees current even for native viewer operators that have
     * no subsequent RNA property update. Sampling reset is decided separately. */
    DEG_id_tag_update(&ntree.id, ID_RECALC_SYNC_TO_EVAL);
    if (ntree.type == NTREE_LPE) {
      for (Scene &scene : bmain.scenes) {
        for (const ViewLayer &view_layer : scene.view_layers) {
          if (view_layer.lpe_node_tree == &ntree) {
            lpe_scenes.add(&scene);
            if (!cached) lpe_sampling_scenes.add(&scene);
          }
        }
      }
    }
  };
  params.tree_output_changed_fn = [&](bNodeTree &ntree, ID & /*owner_id*/) {
    if (cached_preview_trees.contains(&ntree))
      return;
    DEG_id_tag_update(&ntree.id, ID_RECALC_NTREE_OUTPUT);
  };

  std::optional<Vector<bNodeTree *>> modified_trees;
  if (modified_ids.has_value()) {
    modified_trees.emplace();
    for (ID *id : *modified_ids) {
      if (GS(id->name) == ID_NT) {
        modified_trees->append(reinterpret_cast<bNodeTree *>(id));
      }
    }
  }

  BKE_ntree_update(bmain, modified_trees, params);
  if (!lpe_scenes.is_empty()) {
    /* Refresh Render Layers declarations after their scene's LPE graph changed. */
    for (Scene *scene : lpe_scenes) {
      BKE_ntree_update_tag_id_changed(&bmain, &scene->id);
      if (lpe_sampling_scenes.contains(scene))
        DEG_id_tag_update_for_lpe(&scene->id, ID_RECALC_SYNC_TO_EVAL);
      WM_main_add_notifier(NC_SCENE | ND_RENDER_OPTIONS, scene);
    }
    BKE_ntree_update(bmain);
  }
}

void BKE_main_ensure_invariants(Main &bmain, const std::optional<Span<ID *>> modified_ids)
{
  propagate_node_tree_changes(bmain, modified_ids);
}

void BKE_main_ensure_invariants(Main &bmain, ID &modified_id)
{
  BKE_main_ensure_invariants(bmain, {{&modified_id}});
}

}  // namespace blender
