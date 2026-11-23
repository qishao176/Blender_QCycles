/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: Apache-2.0 */

#include <algorithm>
#include <cstring>

#include "device/photon.h"
#include "integrator/path_trace.h"
#include "integrator/render_scheduler.h"
#include "scene/scene.h"
#include "util/log.h"

CCL_NAMESPACE_BEGIN

void PathTrace::configure_photon_replicas()
{
  const bool enabled = device_scene_->data.integrator.use_photon_caustics &&
                       path_trace_works_.size() > 1;
  render_scheduler_.set_photon_replica_batch(
      enabled ? 32 * min(device_scene_->data.integrator.photon_caustics_update_interval, 1024) :
                0);
  if (enabled == photon_replicas_) {
    return;
  }
  photon_replicas_ = enabled;
  render_state_.need_reset_params = true;
  render_scheduler_.set_need_schedule_rebalance(!enabled && path_trace_works_.size() > 1);
  if (enabled) {
    photon_reduce_work_ = PathTraceWork::create(path_trace_works_.front()->get_device(),
                                                film_,
                                                device_scene_,
                                                &render_cancel_.is_requested);
    /* Restore the original constants before film kernels on the first device. */
    photon_reduce_work_->set_photon_replica(0);
    photon_reduce_work_->set_photon_cache_native(photon_cache_native_);
    photon_reduce_work_->set_photon_estimator(photon_estimator_);
    LOG_INFO << "Photon sample replicas: " << path_trace_works_.size()
             << (photon_test_replicas_             ? " (sequential single-GPU simulation)" :
                 photon_test_hybrid(device_->info) ? " (concurrent CPU + OptiX test)" :
                                                     " (OptiX)")
             << "; full tile and per-device sample budgets";
  }
  else {
    if (photon_reduce_work_ && display_) {
      photon_reduce_work_->destroy_gpu_resources(display_.get());
    }
    photon_reduce_work_.reset();
  }
}

/* IDs and sample counts are bit patterns, not floating-point radiance. In particular,
 * adding integer counts as floats can flush these denormals to zero on the host. */
static void add_sample_count(float &destination, const float &source)
{
  uint a, b;
  memcpy(&a, &destination, sizeof(a));
  memcpy(&b, &source, sizeof(b));
  a += b;
  memcpy(&destination, &a, sizeof(a));
}

static void merge_crypto(float *destination, const float *source, const int slots)
{
  for (int j = 0; j < slots; ++j) {
    const float id = source[j * 2], weight = source[j * 2 + 1];
    if (id == 0.0f || weight == 0.0f) {
      continue;
    }
    for (int k = 0; k < slots; ++k) {
      if (destination[k * 2] == 0.0f) {
        destination[k * 2] = id;
      }
      if (destination[k * 2] == id || k == slots - 1) {
        /* Same bounded-slot overflow convention as native Cryptomatte. */
        destination[k * 2 + 1] += weight;
        break;
      }
    }
  }
}

void PathTrace::merge_photon_replica(PathTraceWork &work, const int replica)
{
  /* A completed device reads back independently while other devices continue tracing.
   * Rebuild the aggregate from cumulative snapshots once per batch, never re-add an
   * old snapshot to an existing aggregate. No photon maps cross the bus. */
  if (!work.copy_render_buffers_from_device()) {
    work.get_device()->set_error("Failed to read photon replica film");
    return;
  }
  const thread_scoped_lock lock(photon_reduce_mutex_);
  RenderBuffers *destination = photon_reduce_work_->get_render_buffers();
  const RenderBuffers *source = work.get_render_buffers();
  const KernelFilm &film = device_scene_->data.film;
  const int stride = destination->params.pass_stride;
  const int segment = film.photon_segment_stride ? film.photon_segment_stride : stride;
  enum { ADD, FIRST, COUNT, CRYPTO };
  vector<int> operation(segment, ADD);
  for (const BufferPass &pass : destination->params.passes) {
    if (pass.offset == PASS_UNUSED || pass.offset >= segment) {
      continue;
    }
    const int components = pass.get_info().num_components;
    int op = ADD;
    if (pass.type == PASS_SAMPLE_COUNT || pass.type == PASS_PHOTON_SAMPLE_COUNT) {
      op = COUNT;
    }
    else if (pass.type == PASS_CRYPTOMATTE) {
      op = CRYPTO;
    }
    else if (pass.mode == PassMode::DENOISED || pass.type == PASS_DEPTH ||
             pass.type == PASS_POSITION || pass.type == PASS_OBJECT_ID ||
             pass.type == PASS_MATERIAL_ID)
    {
      op = FIRST;
    }
    std::fill_n(operation.begin() + pass.offset, components, op);
    if (pass.type == PASS_ADAPTIVE_AUX_BUFFER) {
      operation[pass.offset + 3] = FIRST;
    }
  }
  const size_t pixels = destination->buffer.size() / stride;
  float *dst = destination->buffer.data();
  const float *src = source->buffer.data();
  for (size_t pixel = 0; pixel < pixels; ++pixel) {
    for (int offset = 0; offset < stride; offset += segment) {
      float *d = dst + pixel * stride + offset;
      const float *s = src + pixel * stride + offset;
      for (int component = 0; component < segment; ++component) {
        switch (operation[component]) {
          case ADD:
            d[component] += s[component];
            break;
          case FIRST:
            if (replica == 0) {
              memcpy(d + component, s + component, sizeof(float));
            }
            break;
          case COUNT:
            add_sample_count(d[component], s[component]);
            break;
          case CRYPTO:
            break;
        }
      }
      if (film.cryptomatte_passes) {
        int crypto_offset = film.pass_cryptomatte;
        for (const int type : {CRYPT_OBJECT, CRYPT_MATERIAL, CRYPT_ASSET}) {
          if (film.cryptomatte_passes & type) {
            merge_crypto(d + crypto_offset, s + crypto_offset, 2 * film.cryptomatte_depth);
            crypto_offset += 4 * film.cryptomatte_depth;
          }
        }
      }
    }
  }
}

void PathTrace::broadcast_photon_convergence()
{
  const KernelFilm &film = device_scene_->data.film;
  if (film.pass_adaptive_aux_buffer == PASS_UNUSED ||
      !photon_reduce_work_->copy_render_buffers_from_device())
  {
    return;
  }
  const RenderBuffers *aggregate = photon_reduce_work_->get_render_buffers();
  for (auto &work : path_trace_works_) {
    work->copy_estimator_convergence(*photon_reduce_work_);
    if (!work->copy_convergence_masks(*aggregate)) {
      work->get_device()->set_error("Failed to update photon replica convergence mask");
    }
  }
}

CCL_NAMESPACE_END
