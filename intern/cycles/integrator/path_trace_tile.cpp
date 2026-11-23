/* SPDX-FileCopyrightText: 2021-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#include "integrator/path_trace_tile.h"
#include "integrator/pass_accessor_cpu.h"
#include "integrator/path_trace.h"

#include "scene/pass.h"
#include "session/buffers.h"

CCL_NAMESPACE_BEGIN

PathTraceTile::PathTraceTile(PathTrace &path_trace, const bool display)
    : OutputDriver::Tile(display ? make_int2(0, 0) : path_trace.get_render_tile_offset(),
                         display ? path_trace.get_display_tile_size() : path_trace.get_render_tile_size(),
                         display ? path_trace.get_display_tile_size() : path_trace.get_render_size(),
                         path_trace.get_render_tile_params().layer,
                         path_trace.get_render_tile_params().view),
      path_trace_(path_trace),
      copied_from_device_(false), display_(display)
{
}

uint64_t PathTraceTile::get_content_revision() const
{
  return path_trace_.get_buffer_revision();
}

bool PathTraceTile::get_pass_pixels(const string_view pass_name,
                                    const int num_channels,
                                    float *pixels) const
{
  /* NOTE: The code relies on a fact that session is fully update and no scene/buffer modification
   * is happening while this function runs. */

  const BufferParams &buffer_params = path_trace_.get_render_tile_params();

  const BufferPass *pass = buffer_params.find_pass(pass_name);
  /* The viewport's automatically allocated Combined has no public name.
   * Final-render passes are named by BlenderSync::sync_render_passes. */
  if (!pass && display_ && pass_name == "Combined") {
    pass = buffer_params.find_pass(PASS_COMBINED, PassMode::NOISY);
  }
  if (pass == nullptr) {
    return false;
  }

  const bool has_denoised_result = path_trace_.has_denoised_result() ||
                                   is_volume_guiding_pass(pass->type);
  if (pass->mode == PassMode::DENOISED && (!has_denoised_result || display_)) {
    pass = buffer_params.find_pass(pass->type);
    if (pass == nullptr) {
      /* Happens when denoised result pass is requested but is never written by the kernel. */
      return false;
    }
  }

  pass = buffer_params.get_actual_display_pass(pass);
  if (pass == nullptr) {
    /* Happens when interactive session changes display pass but render
     * buffer does not contain it yet. */
    return false;
  }

  if (display_ && path_trace_.read_photon_preview_pass(*pass, num_channels, pixels)) {
    return true;
  }
  if (!copied_from_device_) {
    if (!path_trace_.copy_render_tile_from_device()) {
      return false;
    }
    copied_from_device_ = true;
  }

  const float exposure = buffer_params.exposure;
  const int num_samples = path_trace_.get_num_render_tile_samples();

  PassAccessor::PassAccessInfo pass_access_info(*pass);
  pass_access_info.use_approximate_shadow_catcher = buffer_params.use_approximate_shadow_catcher;
  pass_access_info.use_approximate_shadow_catcher_background =
      pass_access_info.use_approximate_shadow_catcher && !buffer_params.use_transparent_background;

  const PassAccessorCPU pass_accessor(pass_access_info, exposure, num_samples);
  const PassAccessor::Destination destination(pixels, num_channels);

  return path_trace_.get_render_tile_pixels(pass_accessor, destination, display_);
}

bool PathTraceTile::get_pass_pixels(const PassType pass_type,
                                    const int num_channels,
                                    float *pixels) const
{
  /* NOTE: The code relies on a fact that session is fully update and no scene/buffer modification
   * is happening while this function runs. */

  if (!copied_from_device_) {
    /* Copy from device on demand. */
    path_trace_.copy_render_tile_from_device();
    copied_from_device_ = true;
  }

  const BufferParams &buffer_params = path_trace_.get_render_tile_params();

  const BufferPass *pass = buffer_params.find_pass(pass_type);
  if (pass == nullptr) {
    return false;
  }

  const bool has_denoised_result = path_trace_.has_denoised_result() ||
                                   is_volume_guiding_pass(pass->type);
  if (pass->mode == PassMode::DENOISED && !has_denoised_result) {
    pass = buffer_params.find_pass(pass->type);
    if (pass == nullptr) {
      /* Happens when denoised result pass is requested but is never written by the kernel. */
      return false;
    }
  }

  pass = buffer_params.get_actual_display_pass(pass);
  if (pass == nullptr) {
    /* Happens when interactive session changes display pass but render
     * buffer does not contain it yet. */
    return false;
  }

  const float exposure = buffer_params.exposure;
  const int num_samples = path_trace_.get_num_render_tile_samples();

  PassAccessor::PassAccessInfo pass_access_info(*pass);
  pass_access_info.use_approximate_shadow_catcher = buffer_params.use_approximate_shadow_catcher;
  pass_access_info.use_approximate_shadow_catcher_background =
      pass_access_info.use_approximate_shadow_catcher && !buffer_params.use_transparent_background;

  const PassAccessorCPU pass_accessor(pass_access_info, exposure, num_samples);
  const PassAccessor::Destination destination(pixels, num_channels);

  return path_trace_.get_render_tile_pixels(pass_accessor, destination);
}

bool PathTraceTile::set_pass_pixels(const string_view pass_name,
                                    const int num_channels,
                                    const float *pixels) const
{
  if (display_) return false;
  /* NOTE: The code relies on a fact that session is fully update and no scene/buffer modification
   * is happening while this function runs. */

  const BufferParams &buffer_params = path_trace_.get_render_tile_params();
  const BufferPass *pass = buffer_params.find_pass(pass_name);
  if (!pass) {
    return false;
  }
  if (pass->offset == PASS_UNUSED) {
    /* Happens when attempting to set pixels of a pass with compositing when baking. */
    return false;
  }

  const float exposure = buffer_params.exposure;
  const int num_samples = 1;

  const PassAccessor::PassAccessInfo pass_access_info(*pass);
  PassAccessorCPU pass_accessor(pass_access_info, exposure, num_samples);
  const PassAccessor::Source source(pixels, num_channels);

  return path_trace_.set_render_tile_pixels(pass_accessor, source);
}

CCL_NAMESPACE_END
