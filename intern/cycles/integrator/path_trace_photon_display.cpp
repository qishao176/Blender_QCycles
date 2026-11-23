/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: Apache-2.0 */

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "integrator/pass_accessor_cpu.h"
#include "integrator/path_trace.h"
#include "integrator/path_trace_display.h"
#include "integrator/render_scheduler.h"
#include "scene/lpe.h"
#include "scene/scene.h"
#include "session/full_buffer.h"
#include "util/image.h"
#include "util/log.h"
#include "util/path.h"
#include "util/scoped_defer.h"
#include "util/time.h"

CCL_NAMESPACE_BEGIN

/* Preview conversion only. Never write this normalized data back into either estimator. */
static bool preview_pass(const RenderBuffers &buffer,
                         const string_view name,
                         const int channels,
                         float *pixels,
                         const int samples = 0)
{
  const BufferPass *pass = buffer.params.find_pass(name);
  if (!pass) {
    return false;
  }
  if (pass->mode == PassMode::DENOISED) {
    pass = buffer.params.find_pass(pass->type, PassMode::NOISY);
    if (!pass) {
      return false;
    }
  }
  if (pass->offset == PASS_UNUSED && pass->get_info().is_written) {
    /* Known-zero photon channels retain a descriptor, without an accumulation slot. */
    const size_t count = size_t(buffer.params.window_width) * buffer.params.window_height;
    std::fill_n(pixels, count * channels, 0.0f);
    if (channels == 4) {
      for (size_t p = 0; p < count; ++p) {
        pixels[p * 4 + 3] = 1.0f;
      }
    }
    return true;
  }
  PassAccessor::PassAccessInfo info(*pass);
  info.use_approximate_shadow_catcher = buffer.params.use_approximate_shadow_catcher;
  info.use_approximate_shadow_catcher_background = info.use_approximate_shadow_catcher &&
                                                   !buffer.params.use_transparent_background;
  const PassAccessorCPU accessor(
      info, buffer.params.exposure, max(1, samples ? samples : buffer.params.samples));
  /* PassAccessor padding uses the physical buffer dimensions, including overscan.
   * Read the native channel width first, then pad only the displayed window. */
  const int source_channels = is_volume_guiding_pass(pass->type) ? 3 :
                                                                   pass->get_info().num_components;
  if (source_channels == channels) {
    return accessor.get_render_tile_pixels(&buffer, PassAccessor::Destination(pixels, channels));
  }
  const size_t count = size_t(buffer.params.window_width) * buffer.params.window_height;
  vector<float> source(count * source_channels);
  if (!accessor.get_render_tile_pixels(&buffer,
                                       PassAccessor::Destination(source.data(), source_channels)))
  {
    return false;
  }
  for (size_t p = 0; p < count; ++p) {
    for (int c = 0; c < channels; ++c) {
      pixels[p * channels + c] = c < source_channels  ? source[p * source_channels + c] :
                                 c == 3               ? 1.0f :
                                 source_channels == 1 ? source[p] :
                                                        0.0f;
    }
  }
  return true;
}

class PhotonPreviewTile : public OutputDriver::Tile {
 public:
  using Getter = std::function<bool(string_view, int, float *)>;
  PhotonPreviewTile(const int2 size, const BufferParams &params, Getter getter)
      : Tile(make_int2(0, 0), size, size, params.layer, params.view), getter_(std::move(getter))
  {
  }
  bool get_pass_pixels(string_view name, int channels, float *pixels) const override
  {
    return getter_(name, channels, pixels);
  }
  bool set_pass_pixels(string_view, int, const float *) const override
  {
    return false;
  }

 private:
  Getter getter_;
};

void PathTrace::clear_photon_preview_tiles()
{
  if (photon_preview_tiles_file_) {
    fclose(photon_preview_tiles_file_);
    photon_preview_tiles_file_ = nullptr;
    path_remove(photon_preview_tiles_filename_);
  }
  photon_preview_tiles_filename_.clear();
  photon_preview_tiles_.clear();
  photon_preview_tiles_failed_ = false;
}

void PathTrace::store_photon_preview_tile(const RenderBuffers &buffer)
{
  if (!photon_cache_native_ || photon_preview_filename_.empty() ||
      (!display_ && !getenv("CYCLES_PHOTON_TEST_PREVIEW_FILE")) || photon_preview_tiles_failed_)
  {
    return;
  }
  /* A single full-frame native buffer remains available directly. */
  if (buffer.params.window_width == photon_preview_params_.width &&
      buffer.params.window_height == photon_preview_params_.height)
  {
    return;
  }
  if (!photon_preview_tiles_file_) {
    const string filename = photon_preview_filename_ + ".preview";
    photon_preview_tiles_file_ = path_fopen(filename, "wb+");
    if (photon_preview_tiles_file_) {
      photon_preview_tiles_filename_ = filename;
    }
  }
  PhotonPreviewNativeTile tile;
  tile.params = buffer.params;
  /* fpos_t/fsetpos preserve large file offsets on Windows as well as Unix. */
  if (!photon_preview_tiles_file_ || fseek(photon_preview_tiles_file_, 0, SEEK_END) != 0 ||
      fgetpos(photon_preview_tiles_file_, &tile.position) != 0 ||
      fwrite(
          buffer.buffer.data(), sizeof(float), buffer.buffer.size(), photon_preview_tiles_file_) !=
          buffer.buffer.size() ||
      fflush(photon_preview_tiles_file_) != 0)
  {
    LOG_WARNING
        << "Unable to preserve a tile for photon pass preview; final output is unaffected.";
    photon_preview_tiles_failed_ = true;
    return;
  }
  photon_preview_tiles_.push_back(std::move(tile));
}

void PathTrace::set_photon_cache_context(const string &filename, const BufferParams &params)
{
  clear_photon_preview_tiles();
  photon_preview_filename_ = filename;
  photon_preview_params_ = params;
  photon_preview_params_.window_x = photon_preview_params_.window_y = 0;
  photon_preview_params_.window_width = params.width;
  photon_preview_params_.window_height = params.height;
  photon_preview_params_.update_offset_stride();
  photon_preview_native_passes_.clear();
  photon_preview_test_pixels_.clear();
  photon_preview_pass_.clear();
  photon_preview_display_initialized_ = false;
}

bool PathTrace::load_photon_preview_pass(const string_view name,
                                         const int channels,
                                         vector<float> &pixels)
{
  unique_ptr<ImageInput> input(ImageInput::open(photon_preview_filename_));
  if (!input) {
    return false;
  }
  const ImageSpec spec = input->spec();
  BufferParams params = photon_preview_params_;
  params.samples = spec.get_int_attribute("cycles.buffer.samples", params.samples);
  if (spec.width != params.width || spec.height != params.height ||
      spec.nchannels != params.pass_stride || spec.tile_height <= 0)
  {
    return false;
  }
  pixels.resize(size_t(spec.width) * spec.height * channels);
  /* Stream the file in tile rows. Only the requested converted pass remains resident;
   * do not allocate another full-frame native/raw/photon film for live display. */
  RenderBuffers stripe(cpu_device_.get());
  for (int y = 0; y < spec.height; y += spec.tile_height) {
    params.height = min(spec.tile_height, spec.height - y);
    params.window_height = params.height;
    params.full_y = photon_preview_params_.full_y + y;
    params.update_offset_stride();
    stripe.reset(params);
    if (!input->read_tiles(0,
                           0,
                           spec.x,
                           spec.x + spec.width,
                           spec.y + y,
                           spec.y + y + params.height,
                           0,
                           1,
                           0,
                           spec.nchannels,
                           TypeDesc::FLOAT,
                           stripe.buffer.data()) ||
        !preview_pass(stripe, name, channels, pixels.data() + size_t(y) * spec.width * channels))
    {
      pixels.clear();
      return false;
    }
  }
  return input->close();
}

bool PathTrace::get_photon_preview_pixels(const RenderBuffers *current,
                                          const BufferParams &region,
                                          const bool photon_phase,
                                          const string &name,
                                          vector<float> &rgba)
{
  const int width = region.window_width, height = region.window_height;
  const size_t pixels = size_t(width) * height;
  if (width <= 0 || height <= 0) {
    return false;
  }
  const int x = region.full_x + region.window_x - photon_preview_params_.full_x;
  const int y = region.full_y + region.window_y - photon_preview_params_.full_y;
  if (x < 0 || y < 0 || width > photon_preview_params_.width - x ||
      height > photon_preview_params_.height - y)
  {
    return false;
  }
  const bool native_budget = device_scene_->data.integrator.photon_native_samples >
                             render_scheduler_.get_sample_offset();
  const bool native_on_disk = photon_phase && native_budget &&
                              path_exists(photon_preview_filename_);
  const int preview_samples = photon_preview_params_.samples *
                              (photon_replicas_ ? int(path_trace_works_.size()) : 1);
  RenderBuffers cached_photon(cpu_device_.get());
  const RenderBuffers *photon = photon_phase ? current : nullptr;
  if (!photon_phase) {
    const auto found = full_buffer_state_->photons.find(photon_preview_filename_);
    if (found != full_buffer_state_->photons.end()) {
      const auto &cache = found->second;
      if (cache.params.width != photon_preview_params_.width ||
          cache.params.height != photon_preview_params_.height ||
          cache.pixels.size() <
              size_t(cache.params.width) * cache.params.height * cache.params.pass_stride)
      {
        return false;
      }
      BufferParams part = cache.params;
      part.width = width;
      part.height = height;
      part.full_x = region.full_x + region.window_x;
      part.full_y = region.full_y + region.window_y;
      part.window_x = part.window_y = 0;
      part.window_width = width;
      part.window_height = height;
      part.update_offset_stride();
      cached_photon.reset(part);
      for (int row = 0; row < height; ++row) {
        memcpy(cached_photon.buffer.data() + size_t(row) * width * part.pass_stride,
               cache.pixels.data() + (size_t(y + row) * cache.params.width + x) * part.pass_stride,
               size_t(width) * part.pass_stride * sizeof(float));
      }
      photon = &cached_photon;
    }
  }

  /* The pass getter combines linear inputs before evaluating the LPE expression.
   * Non-radiance passes and alpha belong to the native estimator, once available. */
  PhotonPreviewTile tile(
      make_int2(width, height),
      region,
      [&](const string_view pass_name, const int channels, float *output) {
        const BufferPass *pass = photon_preview_params_.find_pass(pass_name);
        if (!pass) {
          return false;
        }
        if (pass->mode == PassMode::DENOISED) {
          pass = photon_preview_params_.find_pass(pass->type, PassMode::NOISY);
          if (!pass) {
            return false;
          }
        }
        const string source_name = pass->name.string();
        const bool radiance = pass->offset >= 0 &&
                              size_t(pass->offset) < device_scene_->photon_caustics_merge.size() &&
                              device_scene_->photon_caustics_merge.data()[pass->offset] == 1;
        bool have_native = false;
        if (!photon_phase && native_budget && current) {
          have_native = preview_pass(*current, source_name, channels, output, preview_samples);
        }
        else if (native_on_disk) {
          const string key = source_name + ":" + to_string(channels);
          auto found = photon_preview_native_passes_.find(key);
          if (found == photon_preview_native_passes_.end()) {
            vector<float> data;
            if (!load_photon_preview_pass(source_name, channels, data)) {
              return false;
            }
            found = photon_preview_native_passes_.emplace(key, std::move(data)).first;
          }
          memcpy(output, found->second.data(), pixels * channels * sizeof(float));
          have_native = true;
        }
        if (!have_native) {
          std::fill_n(output, pixels * channels, 0.0f);
        }
        if (photon && (radiance || !have_native)) {
          vector<float> contribution(pixels * channels);
          if (!preview_pass(*photon, source_name, channels, contribution.data(), preview_samples))
          {
            return have_native;
          }
          for (size_t p = 0; p < pixels; ++p) {
            for (int c = 0; c < channels; ++c) {
              if (!have_native) {
                output[p * channels + c] = contribution[p * channels + c];
              }
              else if (radiance && c < 3) {
                output[p * channels + c] += contribution[p * channels + c];
              }
            }
          }
        }
        return true;
      });

  rgba.assign(pixels * 4, 0.0f);
  std::vector<float> rgb;
  const auto program = film_->get_lpe_program();
  const LPEOutput *expression = program ? program->find_display_output(name) : nullptr;
  if (expression) {
    if (!program->display_evaluate ||
        !program->display_evaluate(*program, expression->expression, tile, rgb) ||
        rgb.size() != pixels * 3)
    {
      return false;
    }
    for (size_t p = 0; p < pixels; ++p) {
      std::copy_n(rgb.data() + p * 3, 3, rgba.data() + p * 4);
      rgba[p * 4 + 3] = 1.0f;
    }
  }
  else if (!getenv("CYCLES_PHOTON_TEST_PREVIEW_PASS") && film_->get_lpe_preview()->evaluate) {
    if (!film_->get_lpe_preview()->evaluate(tile, rgb) || rgb.size() != pixels * 3) {
      return false;
    }
    for (size_t p = 0; p < pixels; ++p) {
      std::copy_n(rgb.data() + p * 3, 3, rgba.data() + p * 4);
      rgba[p * 4 + 3] = 1.0f;
    }
  }
  else if (!getenv("CYCLES_PHOTON_TEST_PREVIEW_PASS") && !film_->get_lpe_preview()->terms.empty())
  {
    const auto plan = film_->get_lpe_preview();
    if (plan->combined && !tile.get_pass_pixels("Combined", 4, rgba.data())) {
      return false;
    }
    vector<float> values(pixels * 3);
    for (const auto &term : plan->terms) {
      if (!tile.get_pass_pixels(term.pass, 3, values.data())) {
        return false;
      }
      for (size_t p = 0; p < pixels; ++p) {
        rgba[p * 4] += values[p * 3] * term.weight.x;
        rgba[p * 4 + 1] += values[p * 3 + 1] * term.weight.y;
        rgba[p * 4 + 2] += values[p * 3 + 2] * term.weight.z;
        if (!plan->combined) {
          rgba[p * 4 + 3] = 1.0f;
        }
      }
    }
  }
  else if (!tile.get_pass_pixels(name, 4, rgba.data())) {
    return false;
  }
  return true;
}

void PathTrace::write_photon_preview_pixels(const BufferParams &region,
                                            const vector<float> &rgba,
                                            half4 *mapped)
{
  const int width = region.window_width, height = region.window_height;
  const int x = region.full_x + region.window_x - photon_preview_params_.full_x;
  const int y = region.full_y + region.window_y - photon_preview_params_.full_y;
  if (mapped) {
    for (int row = 0; row < height; ++row) {
      for (int col = 0; col < width; ++col) {
        const size_t p = size_t(row) * width + col;
        mapped[size_t(y + row) * photon_preview_params_.width + x + col] = {
            float_to_half_display(rgba[p * 4]),
            float_to_half_display(rgba[p * 4 + 1]),
            float_to_half_display(rgba[p * 4 + 2]),
            float_to_half(rgba[p * 4 + 3])};
      }
    }
  }
  if (getenv("CYCLES_PHOTON_TEST_PREVIEW_FILE")) {
    photon_preview_test_pixels_.resize(size_t(photon_preview_params_.width) *
                                       photon_preview_params_.height * 4);
    for (int row = 0; row < height; ++row) {
      memcpy(photon_preview_test_pixels_.data() +
                 (size_t(y + row) * photon_preview_params_.width + x) * 4,
             rgba.data() + size_t(row) * width * 4,
             size_t(width) * 4 * sizeof(float));
    }
  }
}

void PathTrace::update_photon_cache_display(const RenderWork &work, const bool force)
{
  const char *test_file = getenv("CYCLES_PHOTON_TEST_PREVIEW_FILE");
  if ((!display_ && !test_file) || photon_preview_filename_.empty() || device_->have_error()) {
    return;
  }
  if (!force && !work.display.update && !film_->display_pass_name_changed()) {
    return;
  }
  const double start_time = time_dt();
  SCOPED_DEFER(render_scheduler_.report_display_update_time(work, time_dt() - start_time));
  film_->update_display_pass_name();
  string name = film_->get_display_pass_name().string();
  if (const char *test_pass = getenv("CYCLES_PHOTON_TEST_PREVIEW_PASS")) {
    name = test_pass;
  }
  if (name.empty()) {
    const BufferPass *pass = photon_preview_params_.find_pass(film_->get_display_pass(),
                                                              PassMode::NOISY);
    name = pass ? pass->name.string() : "Combined";
  }
  if (test_file && !device_scene_->data.film.photon_compact) {
    const char *switch_pass = getenv("CYCLES_PHOTON_TEST_PREVIEW_SWITCH_PASS");
    const char *switch_tiles = getenv("CYCLES_PHOTON_TEST_PREVIEW_SWITCH_AFTER_TILES");
    if (switch_pass && switch_tiles &&
        photon_preview_tiles_.size() >= size_t(max(atoi(switch_tiles), 1)))
    {
      name = switch_pass;
    }
  }
  if (name != photon_preview_pass_) {
    if (display_ && !photon_preview_pass_.empty()) {
      display_->zero();
    }
    photon_preview_native_passes_.clear();
    photon_preview_pass_ = name;
    photon_preview_display_initialized_ = false;
    photon_preview_test_pixels_.clear();
  }

  PathTraceWork *aggregate = photon_replicas_ ? photon_reduce_work_.get() :
                                                path_trace_works_.front().get();
  if (!aggregate->copy_render_buffers_from_device()) {
    return;
  }
  RenderBuffers *current = aggregate->get_render_buffers();
  const bool photon_phase = device_scene_->data.film.photon_compact;
  /* Map once: a pass switch may restore many stripes and tiles, but must not
   * upload the full display texture once for every individual region. */
  half4 *mapped = nullptr;
  bool display_update = false;
  SCOPED_DEFER([&]() {
    if (display_update) {
      if (mapped) {
        display_->unmap_texture_buffer();
      }
      display_->update_end();
    }
  }());
  if (display_) {
    display_update = display_->update_begin(photon_preview_params_.width,
                                            photon_preview_params_.height);
    if (!display_update) {
      return;
    }
    mapped = display_->map_texture_buffer();
    if (!mapped) {
      return;
    }
    if (!photon_preview_display_initialized_) {
      /* zero() hides the old texture; explicitly clear the new PBO as well. */
      memset(mapped,
             0,
             size_t(photon_preview_params_.width) * photon_preview_params_.height * sizeof(half4));
    }
  }
  vector<float> rgba;
  if (!photon_phase && !photon_preview_display_initialized_) {
    /* Unrendered native tiles must still show the already-computed photon image.
     * Convert it in stripes rather than copying another full raw photon film. */
    if (full_buffer_state_->photons.find(photon_preview_filename_) !=
        full_buffer_state_->photons.end())
    {
      for (int row = 0; row < photon_preview_params_.height; row += 64) {
        BufferParams stripe = photon_preview_params_;
        stripe.height = min(64, photon_preview_params_.height - row);
        stripe.window_height = stripe.height;
        stripe.full_y += row;
        stripe.update_offset_stride();
        if (!get_photon_preview_pixels(nullptr, stripe, false, name, rgba)) {
          photon_preview_display_initialized_ = false;
          return;
        }
        write_photon_preview_pixels(stripe, rgba, mapped);
      }
    }
    RenderBuffers previous(cpu_device_.get());
    for (const PhotonPreviewNativeTile &tile : photon_preview_tiles_) {
      previous.reset(tile.params);
      if (!photon_preview_tiles_file_ ||
          fsetpos(photon_preview_tiles_file_, &tile.position) != 0 ||
          fread(previous.buffer.data(),
                sizeof(float),
                previous.buffer.size(),
                photon_preview_tiles_file_) != previous.buffer.size() ||
          !get_photon_preview_pixels(&previous, tile.params, false, name, rgba))
      {
        LOG_WARNING << "Unable to restore a completed tile for photon pass preview.";
        photon_preview_display_initialized_ = false;
        return;
      }
      write_photon_preview_pixels(tile.params, rgba, mapped);
    }
  }
  if (!get_photon_preview_pixels(current, current->params, photon_phase, name, rgba)) {
    photon_preview_display_initialized_ = false;
    return;
  }
  write_photon_preview_pixels(current->params, rgba, mapped);
  photon_preview_display_initialized_ = true;
  if (test_file) {
    FILE *file = path_fopen(test_file, "wb");
    if (file) {
      fwrite(photon_preview_test_pixels_.data(),
             sizeof(float),
             photon_preview_test_pixels_.size(),
             file);
      fclose(file);
    }
    const BufferParams &region = current->params;
    fprintf(stderr,
            "PHOTON_PREVIEW phase=%s sample=%d force=%d region=%d,%d,%d,%d\n",
            photon_phase ? "photon" : "native",
            completed_buffer_samples_,
            int(force),
            region.full_x + region.window_x - photon_preview_params_.full_x,
            region.full_y + region.window_y - photon_preview_params_.full_y,
            region.window_width,
            region.window_height);
  }
}

CCL_NAMESPACE_END
