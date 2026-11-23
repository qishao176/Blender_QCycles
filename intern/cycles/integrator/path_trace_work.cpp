/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#include "device/device.h"

#include "integrator/pass_accessor_cpu.h"
#include "integrator/path_trace_display.h"
#include "integrator/path_trace_work.h"
#include "integrator/path_trace_work_cpu.h"
#include "integrator/path_trace_work_gpu.h"

#include "scene/film.h"
#include "scene/lpe.h"
#include "scene/scene.h"

#include "session/buffers.h"

#include "kernel/types.h"
#include "util/hash.h"
#include "util/log.h"

CCL_NAMESPACE_BEGIN

void PathTraceWork::set_photon_replica(const uint index)
{
  photon_replica_ = true;
  photon_replica_index_ = index;
}

uint PathTraceWork::photon_sampling_seed() const
{
  const uint seed = device_scene_->data.integrator.seed;
  return photon_replica_index_ && device_scene_->data.integrator.use_photon_caustics ?
             hash_uint2(seed, photon_replica_index_) :
             seed;
}

void PathTraceWork::set_estimator_progress(const int end_sample, const int sample_offset)
{
  estimator_sample_offset_ = sample_offset;
  if (photon_estimator_ != 1) {
    estimator_end_sample_[0] = min(end_sample,
                                   device_scene_->data.integrator.photon_native_samples);
  }
  if (photon_estimator_ != 0) {
    estimator_end_sample_[1] = min(end_sample, device_scene_->data.integrator.photon_samples);
  }
}

void PathTraceWork::copy_estimator_convergence(const PathTraceWork &source)
{
  estimator_done_[0] = source.estimator_done_[0];
  estimator_done_[1] = source.estimator_done_[1];
}

bool PathTraceWork::copy_convergence_masks(const RenderBuffers &source)
{
  const KernelFilm &film = device_scene_->data.film;
  const int stride = source.params.pass_stride;
  const int segment = film.photon_segment_stride ? film.photon_segment_stride : stride;
  for (size_t pixel = 0; pixel < source.buffer.size() / stride; ++pixel) {
    for (int offset = film.photon_segment_stride; offset < stride; offset += segment) {
      const size_t index = pixel * stride + offset + film.pass_adaptive_aux_buffer + 3;
      buffers_->buffer.data()[index] = source.buffer.data()[index];
    }
  }
  return copy_render_buffers_to_device();
}

void PathTraceWork::reset_photon_sampling()
{
  photon_only_ = false;
  estimator_done_[0] = estimator_done_[1] = false;
  estimator_end_sample_[0] = estimator_end_sample_[1] = 0;
  estimator_sample_offset_ = 0;
}

int PathTraceWork::trace_buffer_offset() const
{
  return device_scene_->data.film.photon_segment_stride * (photon_only_ ? 2 : 1);
}

float *PathTraceWork::trace_buffer() const
{
  return buffers_->buffer.data() + trace_buffer_offset();
}

device_ptr PathTraceWork::trace_buffer_device_pointer() const
{
  return buffers_->buffer.device_pointer + sizeof(float) * trace_buffer_offset();
}

void PathTraceWork::render_estimator_samples(RenderStatistics &statistics,
                                             const int start_sample,
                                             const int samples_num,
                                             const int sample_offset)
{
  const KernelIntegrator &integrator = device_scene_->data.integrator;
  if (!device_scene_->data.film.photon_segment_stride && !device_scene_->data.film.photon_compact) {
    render_samples(statistics, start_sample, samples_num, sample_offset);
    return;
  }
  estimator_sample_offset_ = sample_offset;
  float occupancy = 0;
  int work_samples = 0;
  for (int estimator = 0; estimator < 2; ++estimator) {
    photon_only_ = estimator == 1;
    if ((photon_estimator_ >= 0 && estimator != photon_estimator_) ||
        (photon_cache_native_ && photon_only_) ||
        (device_scene_->data.film.photon_compact && !photon_only_))
    {
      continue;
    }
    const int maximum = photon_only_ ? integrator.photon_samples :
                                       integrator.photon_native_samples;
    const int count = min(samples_num, maximum - start_sample);
    if (count <= 0 || (photon_only_ && !integrator.use_photon_caustics)) {
      estimator_done_[estimator] = true;
      continue;
    }
    if (estimator_done_[estimator]) {
      continue;
    }
    RenderStatistics current;
    render_samples(current, start_sample, count, sample_offset);
    estimator_end_sample_[estimator] = start_sample + count;
    estimator_done_[estimator] = estimator_end_sample_[estimator] >= maximum;
    occupancy += current.occupancy * count;
    work_samples += count;
  }
  photon_only_ = false;
  statistics.occupancy = work_samples ? occupancy / work_samples : 0.0f;
}

int PathTraceWork::adaptive_sampling_converge_filter_count_active(const float threshold,
                                                                  bool reset)
{
  if (!device_scene_->data.film.photon_segment_stride && !device_scene_->data.film.photon_compact) {
    return adaptive_sampling_converge_filter_count_active_impl(threshold, reset);
  }
  const KernelIntegrator &integrator = device_scene_->data.integrator;
  const int num_pixels = effective_buffer_params_.width * effective_buffer_params_.height;
  int active = 0;
  for (int estimator = 0; estimator < 2; ++estimator) {
    photon_only_ = estimator == 1;
    if ((photon_estimator_ >= 0 && estimator != photon_estimator_) ||
        (photon_cache_native_ && photon_only_) ||
        (device_scene_->data.film.photon_compact && !photon_only_))
    {
      continue;
    }
    const int maximum = photon_only_ ? integrator.photon_samples :
                                       integrator.photon_native_samples;
    const int end = estimator_end_sample_[estimator];
    if (end >= maximum || (photon_only_ && !integrator.use_photon_caustics)) {
      continue;
    }
    if (reset && !photon_only_) {
      estimator_done_[estimator] = false;
    }
    if (estimator_done_[estimator]) {
      continue;
    }
    /* Photon pixels remain active until their budget is exhausted, even if
     * sparse emission rounds have not found a contributing path yet. */
    if (photon_only_ || !integrator.photon_native_adaptive || threshold <= 0.0f ||
        end - estimator_sample_offset_ < integrator.photon_native_min_samples)
    {
      active += num_pixels;
      continue;
    }
    const int current = adaptive_sampling_converge_filter_count_active_impl(threshold, reset);
    estimator_done_[estimator] = current == 0;
    active += current;
    LOG_DEBUG << "Native independent convergence: " << current
              << " active pixels, sample " << end << ", limit " << maximum;
  }
  photon_only_ = false;
  return min(active, num_pixels);
}

unique_ptr<PathTraceWork> PathTraceWork::create(Device *device,
                                                Film *film,
                                                DeviceScene *device_scene,
                                                const bool *cancel_requested_flag)
{
  if (device->info.type == DEVICE_CPU) {
    return make_unique<PathTraceWorkCPU>(device, film, device_scene, cancel_requested_flag);
  }
  if (device->info.type == DEVICE_DUMMY) {
    /* Dummy devices can't perform any work. */
    return nullptr;
  }

  return make_unique<PathTraceWorkGPU>(device, film, device_scene, cancel_requested_flag);
}

PathTraceWork::PathTraceWork(Device *device,
                             Film *film,
                             DeviceScene *device_scene,
                             const bool *cancel_requested_flag)
    : device_(device),
      film_(film),
      device_scene_(device_scene),
      buffers_(make_unique<RenderBuffers>(device)),
      effective_buffer_params_(buffers_->params),
      effective_denoised_buffer_params_(buffers_->params),
      cancel_requested_flag_(cancel_requested_flag)
{
}

PathTraceWork::~PathTraceWork() = default;

RenderBuffers *PathTraceWork::get_render_buffers()
{
  return buffers_.get();
}

void PathTraceWork::set_effective_buffer_params(
    const BufferParams &effective_big_tile_params,
    const BufferParams &effective_buffer_params,
    const BufferParams &effective_denoised_big_tile_params,
    const BufferParams &effective_denoised_buffer_params)
{
  effective_big_tile_params_ = effective_big_tile_params;
  effective_buffer_params_ = effective_buffer_params;
  effective_denoised_big_tile_params_ = effective_denoised_big_tile_params;
  effective_denoised_buffer_params_ = effective_denoised_buffer_params;
}

bool PathTraceWork::has_multiple_works() const
{
  /* Assume if there are multiple works working on the same big tile none of the works gets the
   * entire big tile to work on. */
  return !(effective_big_tile_params_.width == effective_buffer_params_.width &&
           effective_big_tile_params_.height == effective_buffer_params_.height &&
           effective_big_tile_params_.full_x == effective_buffer_params_.full_x &&
           effective_big_tile_params_.full_y == effective_buffer_params_.full_y);
}

void PathTraceWork::copy_to_render_buffers(RenderBuffers *render_buffers)
{
  copy_render_buffers_from_device();

  const int64_t width = effective_buffer_params_.width;
  const int64_t height = effective_buffer_params_.height;
  const int64_t pass_stride = effective_buffer_params_.pass_stride;
  const int64_t row_stride = width * pass_stride;
  const int64_t data_size = row_stride * height * sizeof(float);

  const int64_t offset_y = effective_buffer_params_.full_y - effective_big_tile_params_.full_y;
  const int64_t offset_in_floats = offset_y * row_stride;

  const float *src = buffers_->buffer.data();
  float *dst = render_buffers->buffer.data() + offset_in_floats;

  memcpy(dst, src, data_size);
}

void PathTraceWork::copy_from_render_buffers(const RenderBuffers *render_buffers)
{
  const int64_t width = effective_buffer_params_.width;
  const int64_t height = effective_buffer_params_.height;
  const int64_t pass_stride = effective_buffer_params_.pass_stride;
  const int64_t row_stride = width * pass_stride;
  const int64_t data_size = row_stride * height * sizeof(float);

  const int64_t offset_y = effective_buffer_params_.full_y - effective_big_tile_params_.full_y;
  const int64_t offset_in_floats = offset_y * row_stride;

  const float *src = render_buffers->buffer.data() + offset_in_floats;
  float *dst = buffers_->buffer.data();

  memcpy(dst, src, data_size);

  copy_render_buffers_to_device();
}

void PathTraceWork::copy_from_denoised_render_buffers(const RenderBuffers *render_buffers)
{
  const int64_t width = effective_denoised_buffer_params_.width;
  const int64_t offset_y = effective_denoised_buffer_params_.full_y -
                           effective_denoised_big_tile_params_.full_y;
  const int64_t offset = offset_y * width;

  render_buffers_host_copy_denoised(buffers_.get(),
                                    effective_denoised_buffer_params_,
                                    render_buffers,
                                    effective_denoised_buffer_params_,
                                    offset);

  copy_render_buffers_to_device();
}

bool PathTraceWork::get_render_tile_pixels(const PassAccessor &pass_accessor,
                                           const PassAccessor::Destination &destination,
                                           const bool effective)
{
  const int offset_y = (effective_buffer_params_.full_y + effective_buffer_params_.window_y) -
                       (effective_big_tile_params_.full_y + effective_big_tile_params_.window_y);
  const int width = effective_buffer_params_.width;

  PassAccessor::Destination slice_destination = destination;
  slice_destination.offset += offset_y * width;

  return pass_accessor.get_render_tile_pixels(buffers_.get(),
                                               effective ? effective_buffer_params_ : buffers_->params,
                                               slice_destination);
}

bool PathTraceWork::read_preview_pass(const PassAccessor::PassAccessInfo &info,
                                      float exposure, int samples, int channels, float *pixels)
{
  const BufferParams &slice = effective_buffer_params_;
  const BufferParams &tile = effective_big_tile_params_;
  const int y = slice.full_y + slice.window_y - tile.full_y - tile.window_y;
  if (slice.window_width != tile.window_width || y < 0 ||
      y + slice.window_height > tile.window_height) {
    return false;
  }
  const PassAccessorCPU accessor(info, exposure, samples);
  PassAccessor::Destination destination(pixels, channels);
  destination.offset = y * tile.window_width;
  return accessor.get_render_tile_pixels(buffers_.get(), slice, destination);
}

bool PathTraceWork::set_render_tile_pixels(PassAccessor &pass_accessor,
                                           const PassAccessor::Source &source)
{
  const int offset_y = effective_buffer_params_.full_y - effective_big_tile_params_.full_y;
  const int width = effective_buffer_params_.width;

  PassAccessor::Source slice_source = source;
  slice_source.offset += offset_y * width;

  return pass_accessor.set_render_tile_pixels(buffers_.get(), slice_source);
}

PassAccessor::PassAccessInfo PathTraceWork::get_display_pass_access_info(PassMode pass_mode) const
{
  const KernelFilm &kfilm = device_scene_->data.film;
  const KernelBackground &kbackground = device_scene_->data.background;

  const BufferParams &params = buffers_->params;

  const auto lpe_preview = film_->get_lpe_preview();
  const ustring display_name = film_->get_display_pass_name();
  const auto program = film_->get_lpe_program();
  const string buffer_name = program ? program->display_buffer_name(display_name.string()) :
                                       display_name.string();
  const auto find_display_pass = [&](const PassMode mode) {
    return display_name.empty() ? params.get_actual_display_pass(film_->get_display_pass(), mode) :
                                  params.get_actual_display_pass(buffer_name, mode);
  };
  const BufferPass *display_pass = !lpe_preview->pass.empty() ?
                                      params.find_pass(lpe_preview->pass.string()) :
                                      find_display_pass(pass_mode);
  if (!display_pass && lpe_preview->pass.empty()) {
    display_pass = find_display_pass(PassMode::NOISY);
  }
  if (!display_pass) {
    /* The requested pass may not exist in a newly synchronized buffer yet. */
    return PassAccessor::PassAccessInfo();
  }

  /* Retain the exact offset, lightgroup flag and albedo convention. Rebuilding
   * this from type alone would read the first group or even a nonexistent
   * global direct/color pass when showing a raw group lobe. */
  PassAccessor::PassAccessInfo pass_access_info(*display_pass);
  for (const auto &term : lpe_preview->terms) {
    if (lpe_preview->combined && display_pass->type != PASS_COMBINED) break;
    const BufferPass *input = params.find_pass(term.pass.string());
    if (!input || input->offset == PASS_UNUSED) return PassAccessor::PassAccessInfo();
    pass_access_info.lpe_terms.push_back({input->offset, term.weight});
  }

  pass_access_info.use_approximate_shadow_catcher = kfilm.use_approximate_shadow_catcher;
  pass_access_info.use_approximate_shadow_catcher_background =
      kfilm.use_approximate_shadow_catcher && !kbackground.transparent;

  if (pass_access_info.mode == PassMode::DENOISED &&
      (effective_denoised_buffer_params_.width != effective_buffer_params_.width ||
       effective_denoised_buffer_params_.height != effective_buffer_params_.height))
  {
    /* Avoid using sample count to filter pass after upscaling, since it is stored at a different
     * resolution. The denoiser should have applied scaling again in this case. */
    pass_access_info.use_sample_count = false;
    pass_access_info.use_approximate_shadow_catcher_background = false;
  }

  pass_access_info.show_active_pixels = film_->get_show_active_pixels();

  return pass_access_info;
}

PassAccessor::Destination PathTraceWork::get_display_destination_template(
    const PathTraceDisplay *display, const PassMode mode) const
{
  const PassType type = get_display_pass_access_info(mode).type;
  PassAccessor::Destination destination(type == PASS_NONE ? film_->get_display_pass() : type, mode);

  const BufferParams &effective_big_tile_params = (mode == PassMode::DENOISED) ?
                                                      effective_denoised_big_tile_params_ :
                                                      effective_big_tile_params_;
  const BufferParams &effective_buffer_params = (mode == PassMode::DENOISED) ?
                                                    effective_denoised_buffer_params_ :
                                                    effective_buffer_params_;
  const int2 display_texture_size = display->get_texture_size();
  const int texture_x = effective_buffer_params.full_x - effective_big_tile_params.full_x +
                        effective_buffer_params.window_x - effective_big_tile_params.window_x;
  const int texture_y = effective_buffer_params.full_y - effective_big_tile_params.full_y +
                        effective_buffer_params.window_y - effective_big_tile_params.window_y;

  destination.offset = texture_y * display_texture_size.x + texture_x;
  destination.stride = display_texture_size.x;

  return destination;
}

CCL_NAMESPACE_END
