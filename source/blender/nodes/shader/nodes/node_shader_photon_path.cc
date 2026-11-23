/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: GPL-2.0-or-later */

#include "node_shader_util.hh"

namespace blender {
namespace nodes::node_shader_photon_path_cc {

static void node_declare(NodeDeclarationBuilder &b)
{
  b.add_output<decl::Float>("Is Photon Ray"_ustr);
  b.add_output<decl::Float>("Is Singular Ray"_ustr);
  b.add_output<decl::Float>("Photon Ray Length"_ustr);
  b.add_output<decl::Float>("Photon Depth"_ustr);
  b.add_output<decl::Float>("Diffuse Length"_ustr);
  b.add_output<decl::Float>("Diffuse Depth"_ustr);
  b.add_output<decl::Float>("Glossy Length"_ustr);
  b.add_output<decl::Float>("Glossy Depth"_ustr);
  b.add_output<decl::Float>("Reflection Length"_ustr);
  b.add_output<decl::Float>("Reflection Depth"_ustr);
  b.add_output<decl::Float>("Transmission Length"_ustr);
  b.add_output<decl::Float>("Transmission Depth"_ustr);
  b.add_output<decl::Float>("Volume Length"_ustr);
  b.add_output<decl::Float>("Volume Depth"_ustr);
  b.add_output<decl::Float>("Volume Layers"_ustr);
  b.add_output<decl::Float>("Portal Depth"_ustr);
  b.add_output<decl::Float>("Is Camera Ray"_ustr);
}

static int node_gpu(
    GPUMaterial *mat, bNode *node, bNodeExecData *, GPUNodeStack *in, GPUNodeStack *out)
{
  return GPU_stack_link(mat, node, "node_photon_path", in, out, GPU_shading_data());
}

NODE_SHADER_MATERIALX_BEGIN
#ifdef WITH_MATERIALX
{
  return val(0.0f);
}
#endif
NODE_SHADER_MATERIALX_END
}  // namespace nodes::node_shader_photon_path_cc

void register_node_type_sh_photon_path()
{
  namespace file_ns = nodes::node_shader_photon_path_cc;
  static bke::bNodeType ntype;
  sh_node_type_base(&ntype, "ShaderNodePhotonPath"_ustr);
  ntype.ui_name = "Photon Path";
  ntype.ui_description =
      "Read light-side photon path information and native camera-ray classification";
  ntype.nclass = NODE_CLASS_INPUT;
  ntype.declare = file_ns::node_declare;
  ntype.gpu_fn = file_ns::node_gpu;
  ntype.materialx_fn = file_ns::node_shader_materialx;
  bke::node_register_type(ntype);
}
}  // namespace blender
