/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

#include <algorithm>
#include <climits>
#include <type_traits>

#include "BKE_context.hh"
#include "BKE_global.hh"
#include "BKE_lib_id.hh"
#include "BKE_main.hh"
#include "BKE_node.hh"
#include "BKE_node_tree_update.hh"
#include "BKE_report.hh"
#include "BLI_listbase.hh"
#include "BLO_read_write.hh"
#include "BLT_translation.hh"
#include "DNA_layer_types.h"
#include "DNA_scene_types.h"

#include "NOD_lpe.hh"
#include "NOD_node_declaration.hh"
#include "NOD_socket_declarations.hh"
#include "NOD_socket_items_blend.hh"
#include "NOD_socket_items_ops.hh"
#include "NOD_socket_items_ui.hh"

#include "RE_lpe.hh"
#include "RNA_access.hh"
#include "RNA_define.hh"
#include "UI_interface_layout.hh"
#include "UI_resources.hh"

#include "node_common.h"

namespace blender {

void register_node_tree_type_lpe()
{
  auto *type = MEM_new<bke::bNodeTreeType>(__func__);
  type->type = NTREE_LPE;
  type->idname = "LPENodeTree"_ustr;
  type->ui_name = N_("LPE Node Editor");
  type->ui_description = N_("Author light path selections and edits");
  type->ui_icon = ICON_NODETREE;
  type->rna_ext.srna = RNA_LPENodeTree;
  type->no_group_interface = true;
  type->update = [](bNodeTree *tree) { ntree_update_reroute_nodes(tree); };
  type->update_cached_preview = RE_lpe_tree_update;
  type->valid_socket_type = [](bke::bNodeTreeType *, bke::bNodeSocketType *socket) {
    return bke::node_is_static_socket_type(*socket) && ELEM(socket->type,
                                                            SOCK_OBJECT,
                                                            SOCK_COLLECTION,
                                                            SOCK_MATERIAL,
                                                            SOCK_RGBA,
                                                            SOCK_FLOAT,
                                                            SOCK_INT);
  };
  type->validate_link = [](eNodeSocketDatatype a, eNodeSocketDatatype b) {
    return a == b ||
           (ELEM(a, SOCK_RGBA, SOCK_FLOAT, SOCK_INT) && ELEM(b, SOCK_RGBA, SOCK_FLOAT, SOCK_INT));
  };
  type->foreach_nodeclass = [](void *data, bke::bNodeClassCallback callback) {
    callback(data, NODE_CLASS_INPUT, N_("Selection"));
    callback(data, NODE_CLASS_OP_COLOR, N_("Light Editing"));
    callback(data, NODE_CLASS_OUTPUT, N_("Output"));
    callback(data, NODE_CLASS_LAYOUT, N_("Layout"));
  };
  type->get_from_context =
      [](const bContext *C, bke::bNodeTreeType *, bNodeTree **tree, ID **id, ID **from) {
        Scene *scene = CTX_data_scene(C);
        ViewLayer *view_layer = CTX_data_view_layer(C);
        *tree = view_layer ? view_layer->lpe_node_tree : nullptr;
        *id = &scene->id;
        *from = nullptr;
      };
  bke::node_tree_type_add(*type);
}

namespace nodes {

template<typename Traits>
void LPEItemsAccessor<Traits>::blend_write_item(BlendWriter *writer, const ItemT &item)
{
  writer->write_string(item.name);
  writer->write_struct_list(&item.operations);
}

template<typename Traits>
void LPEItemsAccessor<Traits>::blend_read_data_item(BlendDataReader *reader, ItemT &item)
{
  BLO_read_string(reader, &item.name);
  BLO_read_struct_list(reader, NodeLPEOperation, &item.operations);
  item.operations_num = BLI_listbase_count(&item.operations);
}

namespace lpe {

static void node_type_base(bke::bNodeType &type, const UString idname, const char *name)
{
  bke::node_type_base(type, idname);
  type.ui_name = name;
  type.ui_description = "Select lighting contributions for LPE render outputs";
  type.default_width = 180;
  type.minwidth = 140;
  type.maxwidth = 600;
  type.poll = [](const bke::bNodeType *, const bNodeTree *tree, const char **hint) {
    if (tree->type != NTREE_LPE) {
      *hint = RPT_("Requires an LPE node tree");
      return false;
    }
    return true;
  };
  type.initfunc = [](bNodeTree *, bNode *node) { node->storage = MEM_new<NodeLPE>(__func__); };
  bke::node_type_storage(
      type,
      "NodeLPE",
      [](bNode *node) {
        socket_items::destruct_array<LPELightItemsAccessor>(*node);
        MEM_delete(static_cast<NodeLPE *>(node->storage));
      },
      [](bNodeTree *, bNode *dst, const bNode *src) {
        dst->storage = MEM_new<NodeLPE>(
            __func__, dna::shallow_copy(*static_cast<const NodeLPE *>(src->storage)));
        /* All LPE nodes use the same storage layout. */
        auto &storage = *static_cast<NodeLPE *>(dst->storage);
        if (storage.items_num > 0) {
          socket_items::copy_array<LPELightItemsAccessor>(*src, *dst);
        }
        else {
          storage.items = nullptr;
        }
      });
  /* Read/write the shared storage for fixed-input nodes too. */
  type.blend_write_storage_content =
      [](const bNodeTree &, const bNode &node, BlendWriter &writer) {
        socket_items::blend_write<LPELightItemsAccessor>(&writer, node);
      };
  type.blend_data_read_storage_content = [](bNodeTree &, bNode &node, BlendDataReader &reader) {
    socket_items::blend_read_data<LPELightItemsAccessor>(&reader, node);
    if (STREQ(node.idname, "LPENodeSceneEdit") && node.storage) {
      auto &storage = *static_cast<NodeLPE *>(node.storage);
      if (storage.additional_samples < 1) {
        storage.additional_samples = 32;
        if (!(node.flag & NODE_CUSTOM_COLOR)) {
          node.flag |= NODE_CUSTOM_COLOR;
          node.color[0] = 0.20f;
          node.color[1] = 0.32f;
          node.color[2] = 0.48f;
        }
      }
    }
  };
}

static void declare_path_outputs(NodeDeclarationBuilder &b)
{
  for (const char *name : {N_("Total Light"),
                           N_("Diffuse"),
                           N_("SSS"),
                           N_("Glossy"),
                           N_("Transmission"),
                           N_("Transparent"),
                           N_("Volume")})
  {
    b.add_output<decl::Color>(UString(name))
        .description("Light path selection; evaluated by the LPE render backend");
  }
}

static void declare_photon_output(NodeDeclarationBuilder &b)
{
  b.add_output<decl::Color>("Photon"_ustr)
      .description("Photon illumination from this source, including reflected and transmitted "
                   "views and subsequent indirect light; already included in Total Light");
}

template<typename Accessor> static void declare_items(NodeDeclarationBuilder &b)
{
  b.use_custom_socket_order();
  b.allow_any_socket_order();
  const bNode *node = b.node_or_null();
  const bNodeTree *tree = b.tree_or_null();
  if (!node || !tree) {
    return;
  }
  constexpr bool is_output = std::is_same_v<Accessor, LPEOutputItemsAccessor>;
  constexpr bool is_scene_edit = std::is_same_v<Accessor, LPESceneEditItemsAccessor>;
  constexpr bool is_selection = !is_output && !is_scene_edit;
  if constexpr (is_selection) {
    declare_path_outputs(b);
  }
  if constexpr (std::is_same_v<Accessor, LPELightItemsAccessor>) {
    declare_photon_output(b);
  }
  if constexpr (std::is_same_v<Accessor, LPEObjectItemsAccessor> ||
                std::is_same_v<Accessor, LPEMaterialItemsAccessor>)
  {
    b.add_output<decl::Color>("Emission"_ustr)
        .description(
            "Select emitted surface light, including its reflected and transmitted contribution");
    b.add_output<decl::Color>("Caustics"_ustr)
        .description(
            "Caustic paths through selected caster objects or materials; objects must enable "
            "Cast Shadow Caustics. Chain casters in Photon mode, connect a receiver, or output "
            "directly for all receivers");
    b.add_output<decl::Color>("Shadow"_ustr)
        .description(
            "Cast shadow from the preceding Light or World onto the following Object or Material "
            "receiver");
  }
  b.add_default_layout();
  if constexpr (std::is_same_v<Accessor, LPEObjectItemsAccessor> ||
                std::is_same_v<Accessor, LPEMaterialItemsAccessor>)
  {
    b.add_input<decl::Color>("Light Path"_ustr)
        .translation_context(BLT_I18NCONTEXT_ID_NODETREE)
        .hide_value()
        .description("Optional preceding path selection for ordered filtering");
    b.add_output<decl::Float>("Mask"_ustr)
        .description(
            "Path coverage of the selected objects or materials, including reflected and "
            "transmitted influence");
  }
  const auto &storage = *static_cast<const NodeLPE *>(node->storage);
  for (const NodeLPEItem &item : Span(storage.items, storage.items_num)) {
    const UString identifier(Accessor::socket_identifier_for_item(item));
    eNodeSocketDatatype socket_type = Accessor::socket_type;
    if constexpr (is_output) {
      const bNodeSocket *existing = bke::node_find_socket(*node, SOCK_IN, identifier);
      if (existing && existing->type == SOCK_FLOAT)
        socket_type = SOCK_FLOAT;
    }
    auto &input = b.add_input(socket_type, UString(item.name), identifier)
                      .socket_name_ptr(&tree->id, *Accessor::item_srna, &item, "name");
    if constexpr (is_selection) {
      input.optional_label();
    }
    else {
      input.hide_value();
    }
    if constexpr (is_scene_edit) {
      input.custom_draw([](CustomSocketDrawParams &params) {
        ui::Layout &row = params.layout.row(true);
        params.draw_standard(row);
        ui::Layout &button = row.row(true);
        const render::LPEApplyScope scope{params.node.identifier, params.socket.identifier, false};
        const bool readonly = RE_lpe_has_readonly_lights(params.tree, scope);
        button.red_alert_set(readonly);
        button.enabled_set(!G.is_rendering);
        PointerRNA op = button.op(
            "NODE_OT_lpe_apply", IFACE_("Apply"), readonly ? ICON_ERROR : ICON_CHECKMARK);
        RNA_int_set(&op, "tree_uid", params.tree.id.session_uid);
        RNA_int_set(&op, "node_identifier", params.node.identifier);
        RNA_string_set(&op, "item_identifier", params.socket.identifier);
      });
      b.add_output<decl::Color>(UString(item.name), identifier)
          .description("Pass through the corresponding input path selection");
    }
  }
  b.add_input<decl::Extend>(""_ustr, "__extend__"_ustr)
      .custom_draw([](CustomSocketDrawParams &params) {
        params.layout.emboss_set(ui::EmbossType::None);
        PointerRNA op = params.layout.op(Accessor::operator_idnames::add_item, "", ICON_ADD);
        RNA_int_set(&op, "node_identifier", params.node.identifier);
        RNA_boolean_set(&op, "show_dialog", false);
        RNA_boolean_set(&op, "init_from_active", false);
      });
}

template<typename Accessor>
static void draw_settings(ui::Layout &layout, bContext *C, PointerRNA *ptr)
{
  if constexpr (std::is_same_v<Accessor, LPESceneEditItemsAccessor>) {
    if (Scene *scene = CTX_data_scene(C)) {
      PointerRNA scene_ptr = RNA_id_pointer_create(&scene->id);
      if (RNA_struct_find_property(&scene_ptr, "cycles")) {
        PointerRNA cycles = RNA_pointer_get(&scene_ptr, "cycles");
        if (RNA_float_get(&cycles, "sample_clamp_direct") != 0.0f ||
            RNA_float_get(&cycles, "sample_clamp_indirect") != 0.0f)
          layout.label(IFACE_("IPR edits use clamped samples"), ICON_INFO);
      }
    }
    /* A slot whose light has a near-zero color channel cannot be rebuilt from
     * the cached contribution; name it here so the user can fix the light. */
    const auto &tree = *reinterpret_cast<const bNodeTree *>(ptr->owner_id);
    const auto status = RE_lpe_graph_status(tree);
    const std::string prefix = "M:" + std::to_string(ptr->data_as<bNode>()->identifier) + ":";
    std::vector<std::string> shown;
    for (const auto &[key, message] : status.weak_light_entries) {
      if (key.rfind(prefix, 0) != 0)
        continue;
      if (std::find(shown.begin(), shown.end(), message) != shown.end())
        continue;
      shown.push_back(message);
      layout.label(message + " (near-zero channel)", ICON_INFO);
    }
  }
  if constexpr (std::is_same_v<Accessor, LPEOutputItemsAccessor>) {
    layout.prop(ptr, "output_mode", UI_ITEM_NONE, std::nullopt, ICON_NONE);
  }
  if constexpr (std::is_same_v<Accessor, LPEObjectItemsAccessor>) {
    /* Preserve old files without offering the receiver-side switch for new graphs. */
    if (RNA_boolean_get(ptr, "shadow_caustics")) {
      layout.label(IFACE_("Legacy caustics selection"), ICON_INFO);
    }
  }
}

template<typename Accessor>
static void draw_items(ui::Layout &layout, bContext *C, PointerRNA *ptr)
{
  draw_settings<Accessor>(layout, C, ptr);
  auto &tree = *reinterpret_cast<bNodeTree *>(ptr->owner_id);
  auto &node = *ptr->data_as<bNode>();
  socket_items::ui::draw_items_list_with_operators<Accessor>(C, &layout, tree, node);
  if constexpr (std::is_same_v<Accessor, LPEOutputItemsAccessor>) {
    socket_items::ui::draw_active_item_props<Accessor>(tree, node, [&](PointerRNA *item) {
      layout.label("LPE_" + RNA_string_get(item, "name"), ICON_RENDERLAYERS);
    });
  }
}

static void register_apply_operator()
{
  WM_operatortype_append([](wmOperatorType *ot) {
    ot->name = "Apply Light Edit";
    ot->idname = "NODE_OT_lpe_apply";
    ot->description =
        "Apply node light settings to editable source data; linked lights require editable "
        "library overrides";
    ot->flag = OPTYPE_UNDO | OPTYPE_REGISTER;
    ot->poll = [](bContext *C) { return !G.is_rendering && CTX_data_scene(C); };
    ot->exec = [](bContext *C, wmOperator *op) {
      Main *bmain = CTX_data_main(C);
      const unsigned int uid = RNA_int_get(op->ptr, "tree_uid");
      bNodeTree *tree = nullptr;
      for (bNodeTree &candidate : bmain->nodetrees) {
        if (candidate.id.session_uid == uid && candidate.type == NTREE_LPE)
          tree = &candidate;
      }
      if (!tree || !BKE_id_is_editable(bmain, &tree->id)) {
        BKE_report(op->reports, RPT_ERROR, "Select an editable LPE node tree");
        return OPERATOR_CANCELLED;
      }
      render::LPEApplyScope scope;
      scope.node = RNA_int_get(op->ptr, "node_identifier");
      scope.item = RNA_string_get(op->ptr, "item_identifier");
      scope.output = RNA_boolean_get(op->ptr, "output_scope");
      std::string error;
      Scene &scene = *CTX_data_scene(C);
      ViewLayer &layer = *CTX_data_view_layer(C);
      if (!scope.output && layer.lpe_node_tree != tree) {
        BKE_report(op->reports,
                   RPT_ERROR,
                   "Select the view layer bound to this Light Edit tree before applying");
        return OPERATOR_CANCELLED;
      }
      if (!RE_lpe_edit_apply(*bmain, *CTX_data_scene(C), *tree, scope, error)) {
        BKE_report(op->reports, RPT_ERROR, error.c_str());
        return OPERATOR_CANCELLED;
      }
      return OPERATOR_FINISHED;
    };
    RNA_def_int(
        ot->srna, "tree_uid", 0, 0, INT_MAX, "Tree", "LPE tree session identifier", 0, INT_MAX);
    RNA_def_int(ot->srna,
                "node_identifier",
                0,
                0,
                INT_MAX,
                "Node",
                "Light Edit or Output node identifier",
                0,
                INT_MAX);
    RNA_def_string(
        ot->srna, "item_identifier", nullptr, 64, "Path", "Stable path socket identifier");
    RNA_def_boolean(ot->srna, "output_scope", false, "Output Path", "Apply only this output path");
  });
}

template<typename Accessor> static void register_items_node(const char *name)
{
  static bke::bNodeType type;
  node_type_base(type, UString(Accessor::node_idname), name);
  type.nclass = (std::is_same_v<Accessor, LPEOutputItemsAccessor> ||
                 std::is_same_v<Accessor, LPESceneEditItemsAccessor>) ?
                    NODE_CLASS_OUTPUT :
                    NODE_CLASS_INPUT;
  type.declare = declare_items<Accessor>;
  type.initfunc = [](bNodeTree *, bNode *node) {
    node->storage = MEM_new<NodeLPE>(__func__);
    if constexpr (std::is_same_v<Accessor, LPEOutputItemsAccessor>) {
      static_cast<NodeLPE *>(node->storage)->mode = 1;
    }
    if constexpr (std::is_same_v<Accessor, LPESceneEditItemsAccessor>) {
      static_cast<NodeLPE *>(node->storage)->mode = 1;
      node->flag |= NODE_CUSTOM_COLOR;
      node->color[0] = 0.20f;
      node->color[1] = 0.32f;
      node->color[2] = 0.48f;
    }
    const char *name = "Selection";
    if constexpr (std::is_same_v<Accessor, LPELightItemsAccessor>) {
      name = "Light";
    }
    else if constexpr (std::is_same_v<Accessor, LPEObjectItemsAccessor>) {
      name = "Object";
    }
    else if constexpr (std::is_same_v<Accessor, LPEMaterialItemsAccessor>) {
      name = "Material";
    }
    else if constexpr (std::is_same_v<Accessor, LPEOutputItemsAccessor>) {
      name = "Beauty";
    }
    else {
      name = "Path";
    }
    socket_items::add_item_with_name<Accessor>(*node, name);
  };
  type.draw_buttons = draw_settings<Accessor>;
  type.draw_buttons_ex = draw_items<Accessor>;
  type.register_operators = []() {
    socket_items::ops::make_common_operators<Accessor>();
    if constexpr (std::is_same_v<Accessor, LPESceneEditItemsAccessor>) {
      register_apply_operator();
    }
  };
  type.insert_link = [](bke::NodeInsertLinkParams &params) {
    if (params.link.tonode == &params.node &&
        STREQ(params.link.tosock->idname, "NodeSocketVirtual") &&
        params.link.fromsock->type != Accessor::socket_type &&
        !(std::is_same_v<Accessor, LPEOutputItemsAccessor> &&
          ELEM(params.link.fromsock->type, SOCK_FLOAT, SOCK_INT)))
    {
      return false;
    }
    const bool accepted = socket_items::try_add_item_via_any_extend_socket<Accessor>(
        params.ntree, params.node, params.node, params.link);
    if constexpr (std::is_same_v<Accessor, LPEOutputItemsAccessor>) {
      if (accepted && params.link.tonode == &params.node) {
        const int type = ELEM(params.link.fromsock->type, SOCK_FLOAT, SOCK_INT) ? SOCK_FLOAT :
                                                                                  SOCK_RGBA;
        if (params.link.tosock->type != type) {
          bke::node_modify_socket_type_static(
              &params.ntree, &params.node, params.link.tosock, type, 0);
          BKE_ntree_update_tag_node_property(&params.ntree, &params.node);
        }
      }
    }
    return accepted;
  };
  type.ignore_inferred_input_socket_visibility = true;
  if constexpr (std::is_same_v<Accessor, LPEOutputItemsAccessor>) {
    type.updatefunc = [](bNodeTree *tree, bNode *node) {
      for (bNodeLink &link : tree->links) {
        if (link.tonode != node || link.tosock->type == SOCK_CUSTOM)
          continue;
        const int type = ELEM(link.fromsock->type, SOCK_FLOAT, SOCK_INT) ? SOCK_FLOAT : SOCK_RGBA;
        if (link.tosock->type != type) {
          bke::node_modify_socket_type_static(tree, node, link.tosock, type, 0);
          BKE_ntree_update_tag_node_property(tree, node);
        }
      }
    };
  }
  if constexpr (std::is_same_v<Accessor, LPELightItemsAccessor> ||
                std::is_same_v<Accessor, LPEObjectItemsAccessor> ||
                std::is_same_v<Accessor, LPEMaterialItemsAccessor>)
  {
    type.updatefunc = [](bNodeTree *tree, bNode *node) {
      auto &storage = *static_cast<NodeLPE *>(node->storage);
      for (NodeLPEItem &item : MutableSpan(storage.items, storage.items_num)) {
        const bNodeSocket *socket = bke::node_find_socket(
            *node, SOCK_IN, UString(Accessor::socket_identifier_for_item(item)));
        if (!socket || !socket->default_value) {
          continue;
        }
        const ID *id;
        if constexpr (std::is_same_v<Accessor, LPEMaterialItemsAccessor>) {
          id = reinterpret_cast<const ID *>(
              socket->default_value_typed<bNodeSocketValueMaterial>()->value);
        }
        else {
          id = reinterpret_cast<const ID *>(
              socket->default_value_typed<bNodeSocketValueObject>()->value);
        }
        if (id) {
          const std::string old_name = item.name;
          socket_items::set_item_name_and_make_unique<Accessor>(*node, item, id->name + 2);
          if (old_name != item.name) {
            BKE_ntree_update_tag_node_property(tree, node);
          }
        }
      }
    };
  }
  if constexpr (std::is_same_v<Accessor, LPESceneEditItemsAccessor>) {
    type.nclass = NODE_CLASS_OP_COLOR;
    type.ui_description =
        "Use upstream light operations for rendering and IPR; Apply writes source data";
    type.internally_linked_input = [](const bNodeTree &,
                                      const bNode &node,
                                      const bNodeSocket &output) -> const bNodeSocket * {
      return bke::node_find_socket(node, SOCK_IN, UString(output.identifier));
    };
  }
  bke::node_register_type(type);
}

static void declare_light_edit(NodeDeclarationBuilder &b)
{
  b.add_input<decl::Color>("Light Path"_ustr)
      .translation_context(BLT_I18NCONTEXT_ID_NODETREE)
      .hide_value();
  b.add_output<decl::Color>("Light Path"_ustr).translation_context(BLT_I18NCONTEXT_ID_NODETREE);
}

static void draw_override(ui::Layout &layout, bContext *, PointerRNA *ptr)
{
  layout.active_set(!G.is_rendering);
  layout.prop(ptr, "color", UI_ITEM_NONE, std::nullopt, ICON_NONE);
  layout.prop(ptr, "use_temperature", UI_ITEM_NONE, std::nullopt, ICON_NONE);
  ui::Layout &temperature = layout.column(false);
  temperature.active_set(RNA_boolean_get(ptr, "use_temperature"));
  temperature.prop(ptr, "temperature", UI_ITEM_NONE, std::nullopt, ICON_NONE);
  layout.prop(ptr, "power", UI_ITEM_NONE, std::nullopt, ICON_NONE);
  layout.prop(ptr, "exposure", UI_ITEM_NONE, std::nullopt, ICON_NONE);
}

static void draw_offset(ui::Layout &layout, bContext *, PointerRNA *ptr)
{
  layout.active_set(!G.is_rendering);
  for (const char *property : {"hue", "saturation", "value", "factor"}) {
    layout.prop(ptr, property, ui::ITEM_R_SLIDER, std::nullopt, ICON_NONE);
  }
}

static void draw_hsl(ui::Layout &layout, bContext *, PointerRNA *ptr)
{
  layout.active_set(!G.is_rendering);
  for (const char *property : {"hue", "saturation", "lightness", "factor"}) {
    layout.prop(ptr, property, ui::ITEM_R_SLIDER, std::nullopt, ICON_NONE);
  }
}

static void draw_brightness(ui::Layout &layout, bContext *, PointerRNA *ptr)
{
  layout.active_set(!G.is_rendering);
  for (const char *property : {"brightness", "contrast", "factor"}) {
    layout.prop(ptr, property, ui::ITEM_R_SLIDER, std::nullopt, ICON_NONE);
  }
}

}  // namespace lpe
}  // namespace nodes

void register_lpe_nodes()
{
  using namespace nodes;
  lpe::register_items_node<LPELightItemsAccessor>(N_("Light"));
  lpe::register_items_node<LPEObjectItemsAccessor>(N_("Object"));
  lpe::register_items_node<LPEMaterialItemsAccessor>(N_("Material"));
  lpe::register_items_node<LPEOutputItemsAccessor>(N_("Output"));
  lpe::register_items_node<LPESceneEditItemsAccessor>(N_("Light Edit"));

  static bke::bNodeType collection_type;
  lpe::node_type_base(collection_type, "LPENodeCollection"_ustr, N_("Collection"));
  collection_type.nclass = NODE_CLASS_INPUT;
  collection_type.ui_description = "Select lights or objects recursively from a collection";
  collection_type.declare = [](NodeDeclarationBuilder &b) {
    b.add_input<decl::Collection>("Collection"_ustr);
    b.add_output<decl::Object>("Lights"_ustr)
        .description("All lights in this collection and its child collections");
    b.add_output<decl::Object>("Objects"_ustr)
        .description(
            "Mesh, volume and instance objects in this collection and its child collections");
  };
  bke::node_register_type(collection_type);

  static bke::bNodeType merge_type;
  lpe::node_type_base(merge_type, "LPENodeMerge"_ustr, N_("Merge Paths"));
  merge_type.nclass = NODE_CLASS_CONVERTER;
  merge_type.ui_description = "Union light paths without adding overlapping contributions twice";
  merge_type.declare = [](NodeDeclarationBuilder &b) {
    b.add_input<decl::Color>("Paths"_ustr)
        .translation_context(BLT_I18NCONTEXT_ID_NODETREE)
        .multi_input()
        .hide_value();
    b.add_output<decl::Color>("Light Path"_ustr).translation_context(BLT_I18NCONTEXT_ID_NODETREE);
  };
  bke::node_register_type(merge_type);

  static bke::bNodeType viewer_type;
  lpe::node_type_base(viewer_type, "LPENodeViewer"_ustr, N_("LPE Viewer"));
  viewer_type.nclass = NODE_CLASS_OUTPUT;
  viewer_type.ui_description = "Preview light paths and expressions while sampling or paused";
  viewer_type.declare = [](NodeDeclarationBuilder &b) {
    b.add_input<decl::Color>("Light Path"_ustr)
        .translation_context(BLT_I18NCONTEXT_ID_NODETREE)
        .hide_value();
  };
  viewer_type.initfunc = [](bNodeTree *tree, bNode *node) {
    node->storage = MEM_new<NodeLPE>(__func__);
    for (bNode &other : tree->nodes) {
      if (STREQ(other.idname, "LPENodeViewer") && other.storage) {
        static_cast<NodeLPE *>(other.storage)->mode = 0;
        other.flag &= ~NODE_DO_OUTPUT;
      }
    }
    static_cast<NodeLPE *>(node->storage)->mode = 1;
    node->flag |= NODE_DO_OUTPUT;
  };
  viewer_type.draw_buttons = [](ui::Layout &layout, bContext *, PointerRNA *ptr) {
    const auto &tree = *reinterpret_cast<const bNodeTree *>(ptr->owner_id);
    const auto status = RE_lpe_graph_status(tree);
    const bool live = status.viewer_direct &&
                      status.viewer_node == ptr->data_as<bNode>()->identifier;
    layout.label(live ? IFACE_("Path preview") : IFACE_("Expression preview"), ICON_INFO);
    layout.prop(ptr, "use_denoising", UI_ITEM_NONE, std::nullopt, ICON_NONE);
  };
  bke::node_register_type(viewer_type);

  static bke::bNodeType world_type;
  lpe::node_type_base(world_type, "LPENodeWorld"_ustr, N_("World"));
  world_type.nclass = NODE_CLASS_INPUT;
  world_type.ui_description =
      "Select illumination and background from the rendered world, including the view layer "
      "override";
  world_type.declare = [](NodeDeclarationBuilder &b) {
    lpe::declare_path_outputs(b);
    lpe::declare_photon_output(b);
  };
  bke::node_register_type(world_type);

  static bke::bNodeType shadow_type;
  lpe::node_type_base(shadow_type, "LPENodeShadowCatcher"_ustr, N_("Shadow Catcher"));
  shadow_type.nclass = NODE_CLASS_INPUT;
  shadow_type.ui_description =
      "Native shadow catcher result for this view layer; mark receiver objects as Shadow Catcher";
  shadow_type.declare = [](NodeDeclarationBuilder &b) {
    b.add_output<decl::Color>("Shadow"_ustr)
        .description(
            "Native RGB shadow factor: multiply the background by this value; white is neutral");
    b.add_output<decl::Float>("Shadow Mask"_ustr)
        .description(
            "Approximate shadow strength, one minus clamped average RGB; excludes foreground "
            "matte alpha");
  };
  shadow_type.draw_buttons = [](ui::Layout &layout, bContext *, PointerRNA *) {
    layout.label(IFACE_("Uses view layer shadow catchers"), ICON_INFO);
  };
  bke::node_register_type(shadow_type);

  static bke::bNodeType mask_type;
  lpe::node_type_base(mask_type, "LPENodeMask"_ustr, N_("Path Mask"));
  mask_type.nclass = NODE_CLASS_CONVERTER;
  mask_type.ui_description =
      "Convert an object/material scattering path to scalar influence coverage";
  mask_type.declare = [](NodeDeclarationBuilder &b) {
    b.add_input<decl::Color>("Light Path"_ustr)
        .translation_context(BLT_I18NCONTEXT_ID_NODETREE)
        .hide_value();
    b.add_output<decl::Float>("Mask"_ustr);
  };
  bke::node_register_type(mask_type);

  static bke::bNodeType override_type;
  lpe::node_type_base(override_type, "LPENodeLightOverride"_ustr, N_("Light Override"));
  override_type.nclass = NODE_CLASS_OP_COLOR;
  override_type.ui_description =
      "Override rendered light settings without changing the source data until Apply";
  override_type.declare = lpe::declare_light_edit;
  override_type.draw_buttons = lpe::draw_override;
  bke::node_register_type(override_type);

  static bke::bNodeType offset_type;
  lpe::node_type_base(offset_type, "LPENodeLightOffset"_ustr, N_("Hue Saturation Value"));
  offset_type.nclass = NODE_CLASS_OP_COLOR;
  offset_type.ui_description = "Offset emitted light color using sampled contributions";
  offset_type.declare = lpe::declare_light_edit;
  offset_type.draw_buttons = lpe::draw_offset;
  bke::node_register_type(offset_type);

  static bke::bNodeType hsl_type;
  lpe::node_type_base(hsl_type, "LPENodeLightHSL"_ustr, N_("Hue Saturation Lightness"));
  hsl_type.nclass = NODE_CLASS_OP_COLOR;
  hsl_type.declare = lpe::declare_light_edit;
  hsl_type.draw_buttons = lpe::draw_hsl;
  bke::node_register_type(hsl_type);

  static bke::bNodeType brightness_type;
  lpe::node_type_base(
      brightness_type, "LPENodeLightBrightnessContrast"_ustr, N_("Brightness Contrast"));
  brightness_type.nclass = NODE_CLASS_OP_COLOR;
  brightness_type.declare = lpe::declare_light_edit;
  brightness_type.draw_buttons = lpe::draw_brightness;
  bke::node_register_type(brightness_type);
}

}  // namespace blender
