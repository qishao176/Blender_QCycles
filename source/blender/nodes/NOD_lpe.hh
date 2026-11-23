/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

#pragma once

#include "DNA_node_types.h"
#include "BLI_listbase.hh"
#include "NOD_socket_items.hh"
#include "RNA_prototypes.hh"

namespace blender {
void register_node_tree_type_lpe();
void register_lpe_nodes();

namespace nodes {

struct LPELightItemsTraits {
  static constexpr StringRefNull node_idname = "LPENodeLight";
  static constexpr eNodeSocketDatatype socket_type = SOCK_OBJECT;
  inline static StructRNA **item_srna = &RNA_LPELightItem;
  struct operator_idnames {
    static constexpr StringRefNull add_item = "NODE_OT_lpe_light_item_add";
    static constexpr StringRefNull remove_item = "NODE_OT_lpe_light_item_remove";
    static constexpr StringRefNull move_item = "NODE_OT_lpe_light_item_move";
  };
  struct ui_idnames {
    static constexpr StringRefNull list = "NODE_UL_lpe_light_items";
  };
};

struct LPEObjectItemsTraits {
  static constexpr StringRefNull node_idname = "LPENodeObject";
  static constexpr eNodeSocketDatatype socket_type = SOCK_OBJECT;
  inline static StructRNA **item_srna = &RNA_LPEObjectItem;
  struct operator_idnames {
    static constexpr StringRefNull add_item = "NODE_OT_lpe_object_item_add";
    static constexpr StringRefNull remove_item = "NODE_OT_lpe_object_item_remove";
    static constexpr StringRefNull move_item = "NODE_OT_lpe_object_item_move";
  };
  struct ui_idnames {
    static constexpr StringRefNull list = "NODE_UL_lpe_object_items";
  };
};

struct LPEMaterialItemsTraits {
  static constexpr StringRefNull node_idname = "LPENodeMaterial";
  static constexpr eNodeSocketDatatype socket_type = SOCK_MATERIAL;
  inline static StructRNA **item_srna = &RNA_LPEMaterialItem;
  struct operator_idnames {
    static constexpr StringRefNull add_item = "NODE_OT_lpe_material_item_add";
    static constexpr StringRefNull remove_item = "NODE_OT_lpe_material_item_remove";
    static constexpr StringRefNull move_item = "NODE_OT_lpe_material_item_move";
  };
  struct ui_idnames {
    static constexpr StringRefNull list = "NODE_UL_lpe_material_items";
  };
};

struct LPEOutputItemsTraits {
  static constexpr StringRefNull node_idname = "LPENodeOutput";
  static constexpr eNodeSocketDatatype socket_type = SOCK_RGBA;
  inline static StructRNA **item_srna = &RNA_LPEOutputItem;
  struct operator_idnames {
    static constexpr StringRefNull add_item = "NODE_OT_lpe_output_item_add";
    static constexpr StringRefNull remove_item = "NODE_OT_lpe_output_item_remove";
    static constexpr StringRefNull move_item = "NODE_OT_lpe_output_item_move";
  };
  struct ui_idnames {
    static constexpr StringRefNull list = "NODE_UL_lpe_output_items";
  };
};

struct LPESceneEditItemsTraits {
  static constexpr StringRefNull node_idname = "LPENodeSceneEdit";
  static constexpr eNodeSocketDatatype socket_type = SOCK_RGBA;
  inline static StructRNA **item_srna = &RNA_LPESceneEditItem;
  struct operator_idnames {
    static constexpr StringRefNull add_item = "NODE_OT_lpe_scene_edit_item_add";
    static constexpr StringRefNull remove_item = "NODE_OT_lpe_scene_edit_item_remove";
    static constexpr StringRefNull move_item = "NODE_OT_lpe_scene_edit_item_move";
  };
  struct ui_idnames {
    static constexpr StringRefNull list = "NODE_UL_lpe_scene_edit_items";
  };
};

template<typename Traits>
struct LPEItemsAccessor : socket_items::SocketItemsAccessorDefaults, Traits {
  using ItemT = NodeLPEItem;
  static constexpr bool has_type = false;
  static constexpr bool has_name = true;
  struct rna_names {
    static constexpr StringRefNull items = "path_items";
    static constexpr StringRefNull active_index = "active_index";
  };
  static socket_items::SocketItemsRef<ItemT> get_items_from_node(bNode &node)
  {
    auto &data = *static_cast<NodeLPE *>(node.storage);
    return {&data.items, &data.items_num, &data.active_index};
  }
  static void copy_item(const ItemT &src, ItemT &dst)
  {
    dst = src;
    dst.name = BLI_strdup_null(src.name);
    BLI_duplicatelist(&dst.operations, &src.operations);
  }
  static void destruct_item(ItemT *item)
  {
    MEM_SAFE_DELETE(item->name);
    BLI_freelistN(&item->operations);
  }
  static char **get_name(ItemT &item)
  {
    return &item.name;
  }
  static void init_with_name(bNode &node, ItemT &item, const char *name)
  {
    item.identifier = static_cast<NodeLPE *>(node.storage)->next_identifier++;
    socket_items::set_item_name_and_make_unique<LPEItemsAccessor>(node, item, name);
  }
  static std::string socket_identifier_for_item(const ItemT &item)
  {
    return "Item_" + std::to_string(item.identifier);
  }
  static void blend_write_item(BlendWriter *writer, const ItemT &item);
  static void blend_read_data_item(BlendDataReader *reader, ItemT &item);
};

using LPELightItemsAccessor = LPEItemsAccessor<LPELightItemsTraits>;
using LPEObjectItemsAccessor = LPEItemsAccessor<LPEObjectItemsTraits>;
using LPEMaterialItemsAccessor = LPEItemsAccessor<LPEMaterialItemsTraits>;
using LPEOutputItemsAccessor = LPEItemsAccessor<LPEOutputItemsTraits>;
using LPESceneEditItemsAccessor = LPEItemsAccessor<LPESceneEditItemsTraits>;

}  // namespace nodes
}  // namespace blender
