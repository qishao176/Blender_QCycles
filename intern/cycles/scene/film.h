/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#pragma once

#include <memory>
#include <functional>
#include "session/output_driver.h"

#include "scene/pass.h"

#include "kernel/types.h"

#include "graph/node.h"

#include "util/string.h"

CCL_NAMESPACE_BEGIN

class Device;
class DeviceScene;
class Scene;
class Object;
class Shader;
struct LPEProgram;
struct LPEPreviewCache;

struct LPESelection {
  Object *object = nullptr;
  Shader *shader = nullptr;
};

enum FilterType {
  FILTER_BOX,
  FILTER_GAUSSIAN,
  FILTER_BLACKMAN_HARRIS,

  FILTER_NUM_TYPES,
};

class Film : public Node {
 public:
  NODE_DECLARE

  NODE_SOCKET_API(float, exposure)
  NODE_SOCKET_API(float, pass_alpha_threshold)

  NODE_SOCKET_API(PassType, display_pass)
  NODE_SOCKET_API(bool, show_active_pixels)

  /* Render-window selection is a named pass, not just a PassType: several
   * light groups (and AOVs) share a type. Publish from the drawing thread and
   * latch once per display update so dimensions/mode/offset cannot disagree. */
  void set_display_pass_name(ustring name)
  {
    std::atomic_store(&pending_display_pass_name_, std::make_shared<const ustring>(name));
  }
  void update_display_pass_name()
  {
    display_pass_name_ = *std::atomic_load(&pending_display_pass_name_);
  }
  bool display_pass_name_changed() const
  {
    return display_pass_name_ != *std::atomic_load(&pending_display_pass_name_);
  }
  ustring get_display_pass_name() const
  {
    return display_pass_name_;
  }

  NODE_SOCKET_API(FilterType, filter_type)
  NODE_SOCKET_API(float, filter_width)

  NODE_SOCKET_API(float, mist_start)
  NODE_SOCKET_API(float, mist_depth)
  NODE_SOCKET_API(float, mist_falloff)

  NODE_SOCKET_API(CryptomatteType, cryptomatte_passes)
  NODE_SOCKET_API(int, cryptomatte_depth)

  /* Approximate shadow catcher pass into its matte pass, so that both artificial objects and
   * shadows can be alpha-overed onto a backdrop. */
  NODE_SOCKET_API(bool, use_approximate_shadow_catcher)

  NODE_SOCKET_API(bool, use_sample_count)

  NODE_SOCKET_API(bool, denoising_pass_follow_reflections);
  NODE_SOCKET_API(bool, denoising_pass_use_albedo_roughness_weighting);

  /* Deep EXR output settings. */
  NODE_SOCKET_API(bool, use_deep_output)
  NODE_SOCKET_API(int, deep_max_samples)
  NODE_SOCKET_API(bool, deep_simple_mode)
  NODE_SOCKET_API(int, deep_photon_samples)

  void set_deep_samples_ptr(uint64_t ptr)
  {
    deep_samples_ptr_ = ptr;
    tag_modified();
  }
  void set_deep_sample_counts_ptr(uint64_t ptr)
  {
    deep_sample_counts_ptr_ = ptr;
    tag_modified();
  }
  void set_deep_dimensions(int w, int h)
  {
    deep_width_ = w;
    deep_height_ = h;
    tag_modified();
  }

 private:
  std::shared_ptr<const ustring> pending_display_pass_name_ = std::make_shared<const ustring>();
  ustring display_pass_name_;
  size_t filter_table_offset_;
  bool prev_have_uv_pass = false;
  bool prev_have_motion_pass = false;
  bool prev_have_ao_pass = false;

  uint64_t deep_samples_ptr_ = 0;
  uint64_t deep_sample_counts_ptr_ = 0;
  int deep_width_ = 0;
  int deep_height_ = 0;

 public:
  Film();
  ~Film() override;

  vector<string> deep_lpe_passes;
  struct DeepLPEOutput {
    string name;
    vector<std::pair<int, float3>> terms;
    int divisor = -1;
    bool use_exposure = true;
  };
  struct DeepNativeSource { int channel; PassType type; ustring lightgroup; };
  vector<DeepNativeSource> deep_native_sources;
  vector<DeepLPEOutput> deep_lpe_outputs;
  vector<KernelLPEState> lpe_states;
  vector<KernelLPEState> photon_lpe_states;
  vector<ustring> photon_lpe_pass_names;
  uint64_t photon_lpe_initial = 0;
  bool lpe_independent_photon = false;
  vector<LPESelection> lpe_selections;
  /* Host-only selections requiring unambiguous volume-event provenance. */
  vector<LPESelection> lpe_volume_selections;
  vector<ustring> lpe_pass_names;
  struct LPEPreview {
    struct Term { ustring pass; float3 weight; };
    ustring pass;
    std::function<bool(const OutputDriver::Tile &, std::vector<float> &)> evaluate;
    bool frozen = false;
    vector<Term> terms;
    bool combined = false;
    bool denoise = false;
  };
 private:
  std::shared_ptr<const LPEPreview> lpe_preview_ = std::make_shared<LPEPreview>();
  /* Published plans are immutable. UI edits may replace a plan while the render
   * thread consumes its previous version, without waiting for a sample batch. */
  std::shared_ptr<const LPEProgram> lpe_program_;
  /* Main-thread viewport selection, published independently of sampled inputs. */
  std::shared_ptr<const string> lpe_display_output_ =
      std::make_shared<const string>("LPE Viewer");

 public:
  string get_lpe_display_output() const { return *std::atomic_load(&lpe_display_output_); }
  void set_lpe_display_output(const string &name) {
    std::atomic_store(&lpe_display_output_, std::make_shared<const string>(name));
  }
  std::shared_ptr<const LPEProgram> get_lpe_program() const {
    return std::atomic_load(&lpe_program_);
  }
  void set_lpe_program(std::shared_ptr<const LPEProgram> program) {
    std::atomic_store(&lpe_program_, std::move(program));
  }
  std::shared_ptr<const LPEPreview> get_lpe_preview() const {
    return std::atomic_load(&lpe_preview_);
  }
  void set_lpe_preview(ustring pass,
                      std::function<bool(const OutputDriver::Tile &, std::vector<float> &)> evaluate = {},
                      bool frozen = false, bool denoise = false) {
    auto state = std::make_shared<LPEPreview>();
    state->pass = pass;
    state->evaluate = std::move(evaluate);
    state->frozen = frozen;
    state->denoise = denoise;
    std::atomic_store(&lpe_preview_, std::shared_ptr<const LPEPreview>(state));
  }
  void set_lpe_linear_preview(vector<LPEPreview::Term> terms, bool combined, bool denoise = false) {
    auto state = std::make_shared<LPEPreview>();
    state->terms = std::move(terms);
    state->combined = combined;
    state->denoise = denoise;
    if (!combined && !state->terms.empty()) state->pass = state->terms.front().pass;
    std::atomic_store(&lpe_preview_, std::shared_ptr<const LPEPreview>(state));
  }
  uint64_t lpe_initial = 0;
  /* Contents are used only by display evaluation, never by UI compilation. */
  std::shared_ptr<LPEPreviewCache> lpe_preview_cache;

  /* add default passes to scene */
  static void add_default(Scene *scene);

  void device_update(Device *device, DeviceScene *dscene, Scene *scene);
  void device_free(Device *device, DeviceScene *dscene, Scene *scene);

  int get_aov_offset(Scene *scene, string name, bool &is_color);

  bool update_lightgroups(Scene *scene);

  /* Update passes so that they contain all passes required for the configured functionality. */
  void update_passes(Scene *scene);

  uint64_t get_kernel_features(const Scene *scene) const;

 private:
  void add_auto_pass(Scene *scene, PassType type, const char *name = nullptr);
  void add_auto_pass(Scene *scene, PassType type, PassMode mode, const char *name = nullptr);
  void remove_auto_passes(Scene *scene);
  void finalize_passes(Scene *scene, const bool use_denoise);
};

CCL_NAMESPACE_END
