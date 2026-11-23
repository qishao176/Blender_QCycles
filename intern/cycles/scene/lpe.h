/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: Apache-2.0 */

#pragma once

#include <map>
#include <vector>
#include <functional>

#include "session/output_driver.h"

#include "util/string.h"
#include "util/types.h"

CCL_NAMESPACE_BEGIN

/* A post-accumulation program. Never evaluated independently on path samples. */
enum class LPEOperation { CONSTANT, PASS, MASK, MATH, MIX, TO_FLOAT, EDIT, SHADOW_MASK, SHADOW_RATIO };

struct LPELightParameters {
  float3 color = make_float3(1.0f);
  float power = 0.0f;
  float exposure = 0.0f;
  float temperature = 6500.0f;
  bool use_temperature = false;
};

/* An edit is evaluated against the sampling-time source, never against the
 * previously modified image. Distinct sources retain distinct contributions. */
struct LPEEditTerm {
  string pass;
  string global_pass;
  string lightgroup_pass;
  string source_name;
  unsigned int source_object_uid = 0;
  unsigned int source_light_uid = 0;
  string source_instance;
  int light_instance_count = 0;
  LPELightParameters source_parameters;
  LPELightParameters sampling_parameters;
  float3 multiplier = make_float3(1.0f);
  int manager_node = 0;
  string manager_item;
  std::vector<int> operation_nodes;
  std::vector<int> scene_operation_nodes;
  int output_node = 0;
  string output_item;
};

struct LPEExpression {
  LPEOperation operation = LPEOperation::CONSTANT;
  float3 value = make_float3(0.0f);
  int inputs[3] = {-1, -1, -1};
  int mode = 0;
  bool clamp = false;
  bool clamp_factor = true;
  bool scalar = false;
  string pass;
  std::vector<LPEEditTerm> edit_terms;
};

struct LPEOutput {
  string name;
  int expression = -1;
  int channels = 3;
};

struct LPEProgram {
  /* Which output roots were compiled; independent of snapshot engine lifetime. */
  bool viewport = false;
  bool viewer_denoising = false;
  /* Edits transform the sampled contribution. With sample clamping this is
   * post-clamp grading, not a prediction of a newly clamped physical render. */
  bool sample_clamped = false;
  std::vector<LPEExpression> expressions;
  std::vector<LPEOutput> outputs;
  /* Connected Light Edit sockets can be observed without registering render passes. */
  std::map<string, int> edit_previews;
  int viewer_node = 0;
  int viewer_expression = -1;
  /* Stable sampling requirements -> buffers retained by the session. Display
   * topology may change while these sampled contributions remain valid. */
  std::map<string, LPEExpression> sampling_bindings;
  std::vector<LPEEditTerm> combined_terms;

  /* Final-render display evaluates only the selected public output. Kept in
   * the immutable program so its evaluator and expressions change together. */
  std::function<bool(const LPEProgram &, int, const OutputDriver::Tile &,
                     std::vector<float> &)> display_evaluate;

  const LPEOutput *find_display_output(const string &name) const
  {
    if (viewport) return nullptr;
    for (const LPEOutput &output : outputs) {
      if (output.name == name && output.expression >= 0 &&
          size_t(output.expression) < expressions.size()) return &output;
    }
    return nullptr;
  }

  string display_buffer_name(const string &name) const
  {
    if (const LPEOutput *output = find_display_output(name)) {
      const LPEExpression &expression = expressions[output->expression];
      if (expression.operation == LPEOperation::PASS || expression.operation == LPEOperation::MASK)
        return expression.pass;
    }
    return name;
  }
};

CCL_NAMESPACE_END
