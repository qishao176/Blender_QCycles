/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: Apache-2.0 */

#include "integrator/path_trace.h"
#include "integrator/path_trace_work.h"
#include "integrator/path_trace_work_cpu.h"
#include "integrator/path_trace_work_gpu.h"
#include "integrator/vcm_merge_round.h"
#include "kernel/device/cpu/vcm_connect.h"
#include "kernel/integrator/vcm_types.h"
#include "scene/scene.h"
#include "util/log.h"
#include "util/scoped_defer.h"
#include "util/tbb.h"
#include "util/time.h"

#include <bit>
#include <charconv>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <string_view>

CCL_NAMESPACE_BEGIN

namespace {

bool capture_setting(const char *name, const uint default_value, const uint limit, uint &value)
{
  value = default_value;
  if (const char *text = getenv(name)) {
    const char *end = text + strlen(text);
    const auto result = std::from_chars(text, end, value);
    return result.ec == std::errc() && result.ptr == end && value >= 1 && value <= limit;
  }
  return true;
}

bool vcm_source_pass(const BufferPass &pass)
{
  return pass.mode == PassMode::NOISY && pass.offset != PASS_UNUSED &&
         ((pass.type == PASS_COMBINED && !pass.lightgroup.empty()) || pass.type == PASS_LPE ||
          pass.type == PASS_EMISSION || pass.type == PASS_BACKGROUND);
}

struct VCMTransfer {
  VCMBatch batch;
  vector<VCMLightPath> paths;
  vector<VCMVertex> vertices;
  bool ready = false;
};

}  // namespace

void PathTrace::reset_vcm(const int paths, const int rounds)
{
  vcm_paths_ = paths;
  vcm_rounds_ = rounds;
  vcm_adaptive_film_.free_memory();
  for (const auto &work : path_trace_works_) {
    if (work->get_device()->info.type == DEVICE_OPTIX) {
      static_cast<PathTraceWorkGPU *>(work.get())->reset_vcm_guides();
    }
  }
  vcm_point_targets_.clear();
  vcm_point_fractions_.clear();
  if (paths) {
    vcm_film_.clear();
  }
  else {
    vcm_film_.free_memory();
  }
}

bool PathTrace::write_vcm_tile(const uint film_samples)
{
  /* Session has already scaled the camera raster to this render work's pixel size. */
  const int width = int(device_scene_->data.cam.width);
  const int height = int(device_scene_->data.cam.height);
  for (const auto &work : path_trace_works_) {
    RenderBuffers *buffers = work->get_render_buffers();
    const BufferParams &params = work->get_effective_buffer_params();
    if (!params.width || !params.height) {
      continue;
    }
    vector<int> pass_offsets;
    for (const BufferPass &pass : params.passes) {
      if (vcm_source_pass(pass)) {
        pass_offsets.push_back(pass.offset);
      }
    }
    const int combined = params.get_pass_offset(PASS_COMBINED);
    const int adaptive = params.get_pass_offset(PASS_ADAPTIVE_AUX_BUFFER);
    const int samples = params.get_pass_offset(PASS_SAMPLE_COUNT);
    if (combined == PASS_UNUSED || params.full_x < 0 || params.full_y < 0 ||
        params.full_x + params.width > width || params.full_y + params.height > height)
    {
      device_->set_error("VCM film and render tile dimensions do not match.");
      return false;
    }
    float *dst = buffers->buffer.data();
    std::fill_n(dst, buffers->buffer.size(), 0.0f);
    for (int y = 0; y < params.height; ++y) {
      const float3 *row = &vcm_film_[size_t(params.full_y + y) * width + params.full_x];
      for (int x = 0; x < params.width; ++x, dst += params.pass_stride) {
        for (size_t g = 0; g < pass_offsets.size(); ++g) {
          const float3 &color = row[(g + 1) * size_t(width) * height + x];
          const int offset = pass_offsets[g];
          dst[offset] = color.x;
          dst[offset + 1] = color.y;
          dst[offset + 2] = color.z;
        }
        dst[combined] = row[x].x;
        dst[combined + 1] = row[x].y;
        dst[combined + 2] = row[x].z;
        if (adaptive != PASS_UNUSED) {
          const float3 a = vcm_adaptive_film_[size_t(params.full_y + y) * width + params.full_x + x];
          dst[adaptive] = a.x;
          dst[adaptive + 1] = a.y;
          dst[adaptive + 2] = a.z;
        }
        /* Cycles stores transparency in Combined.w. Supported VCM films are opaque. */
        if (samples != PASS_UNUSED) {
          dst[samples] = std::bit_cast<float>(film_samples);
        }
      }
    }
    /* Publish on the same device queue as tile clearing, pass extraction and IPR display. */
    if (!work->copy_render_buffers_to_device()) {
      return false;
    }
  }
  return !device_->have_error();
}

bool PathTrace::render_vcm(const uint first_round, const uint round_count)
{
  namespace fs = std::filesystem;
  const KernelData &data = device_scene_->data;
  if (data.integrator.use_volumes ||
      data.kernel_features & (KERNEL_FEATURE_OSL_SHADING | KERNEL_FEATURE_NESTED_DIELECTRICS))
  {
    device_->set_error("VCM requires an SVM surface scene.");
    return false;
  }
  const bool final_render = vcm_paths_ != 0;
  /* The destination of a splat is its actual emitter, including the photon prefix in VM.
   * Compile source-only LPE and light-group predicates once for the whole batch. */
  vector<const BufferPass *> passes;
  for (const BufferPass &pass : full_params_.passes) {
    if (final_render && vcm_source_pass(pass)) {
      passes.push_back(&pass);
    }
  }
  const int planes = int(passes.size());
  vector<int> routes;
  if (planes) {
    const int lights = data.integrator.num_lights;
    /* Slots after the real lamps classify primary camera hits for Emission and Environment. */
    routes.resize(lights + 4);
    for (int light = -1; light < lights + 2; ++light) {
      routes[light + 1] = int(routes.size());
      const bool is_lamp = light >= 0 && light < lights;
      const int object = is_lamp ? device_scene_->lights[light].object_id : data.background.object_index;
      const int group = is_lamp && object != OBJECT_NONE ? device_scene_->objects[object].lightgroup :
                                                          data.background.lightgroup;
      for (int i = 0; i < planes; ++i) {
        const BufferPass &pass = *passes[i];
        bool match = false;
        if (light >= lights) {
          match = pass.type == (light == lights ? PASS_EMISSION : PASS_BACKGROUND);
        }
        else if (pass.type == PASS_COMBINED) {
          match = group != LIGHTGROUP_NONE && pass.offset == data.film.pass_lightgroup + 3 * group;
        }
        else if (pass.type == PASS_LPE) {
          for (int j = 0; j < data.film.lpe_state_count && !match; ++j) {
            const KernelLPEState &s = device_scene_->lpe_states[data.film.lpe_state_offset + j];
            if (s.pass_offset != pass.offset) {
              continue;
            }
            match = s.light_count < 0;
            for (int k = 0; k < s.light_count && !match; ++k) {
              match = device_scene_->lpe_selections[s.light_offset + k] == object;
            }
          }
        }
        if (match) {
          routes.push_back(i);
        }
      }
    }
    routes[lights + 3] = int(routes.size());
  }
  const bool adaptive = final_render && data.film.pass_adaptive_aux_buffer != PASS_UNUSED;
  if (final_render && full_params_.use_transparent_background) {
    device_->set_error("VCM requires an opaque film.");
    return false;
  }
  uint count = vcm_paths_, rounds = vcm_rounds_, depth = uint(data.integrator.max_bounce);
  uint batch_size = 1024;
  if (final_render && (depth == 0 || depth > 64)) {
    device_->set_error("VCM supports up to 63 maximum bounces.");
    return false;
  }
  if (!final_render && (!capture_setting("CYCLES_VCM_PATHS", 4096, 4194304, count) ||
      !capture_setting("CYCLES_VCM_ROUNDS", 1, 4096, rounds) ||
      !capture_setting("CYCLES_VCM_DEPTH", 4, 64, depth) ||
      !capture_setting("CYCLES_VCM_BATCH", 256, 4096, batch_size)))
  {
    device_->set_error("Invalid VCM capture budget, batch size, or depth.");
    return false;
  }

  const char *connect_setting = getenv("CYCLES_VCM_CONNECT");
  const bool connect = final_render || (connect_setting && std::string_view(connect_setting) == "1");
  const char *dump_option = getenv("CYCLES_VCM_DUMP_PATHS");
  const bool dump_paths = !final_render && (!connect || !dump_option || std::string_view(dump_option) != "0");
  double initial_radius = final_render ? 0.02 : 0.0;
  if (const char *setting = final_render ? nullptr : getenv("CYCLES_VCM_MERGE_RADIUS")) {
    const char *end = setting + strlen(setting);
    const auto parsed = std::from_chars(setting, end, initial_radius);
    if (parsed.ptr != end || parsed.ec != std::errc() || !std::isfinite(initial_radius) ||
        initial_radius < 0.0 || (initial_radius > 0 && !connect))
    {
      device_->set_error(
          "VCM merge radius must be zero, or positive and finite with connections enabled.");
      return false;
    }
  }
  PathTraceWorkCPU *consumer = nullptr;
  vector<PathTraceWork *> producers;
  for (const auto &work : path_trace_works_) {
    const DeviceType type = work->get_device()->info.type;
    if (type != DEVICE_CPU && type != DEVICE_OPTIX) {
      device_->set_error("VCM path capture requires CPU or OptiX producers.");
      return false;
    }
    if (type == DEVICE_CPU && !consumer) {
      consumer = static_cast<PathTraceWorkCPU *>(work.get());
    }
    if (type == DEVICE_OPTIX) {
      /* Render replicas can reference one device and share its path slots. */
      const bool already_present = std::any_of(producers.begin(), producers.end(), [&](auto *p) {
        return p->get_device() == work->get_device();
      });
      if (!already_present) {
        producers.push_back(work.get());
      }
    }
  }
  if (producers.empty()) {
    producers.push_back(path_trace_works_.front().get());
  }

  if (connect && (data.cam.type != CAMERA_PERSPECTIVE || data.cam.aperturesize != 0 ||
                  data.cam.interocular_offset != 0 ||
                  data.cam.width < 1 || data.cam.height < 1))
  {
    device_->set_error(
        "VCM requires a pinhole perspective camera.");
    return false;
  }

  bool area = false, point = false, textured_emitter = false;
  if (connect) {
    if (data.kernel_features & (KERNEL_FEATURE_LIGHT_LINKING | KERNEL_FEATURE_SHADOW_LINKING)) {
      device_->set_error("VCM does not yet support light or shadow linking.");
      return false;
    }
    for (int i = 0; i < data.integrator.num_lights; ++i) {
      const KernelLight &l = device_scene_->lights[i];
      if (l.type == LIGHT_BACKGROUND) {
        continue;
      }
      const bool delta = (l.type == LIGHT_POINT || l.type == LIGHT_SPOT) && l.spot.radius == 0;
      point |= delta;
      area |= l.type == LIGHT_AREA;
      textured_emitter |= !(device_scene_->shaders[l.shader_id & SHADER_MASK].flags & SD_HAS_CONSTANT_EMISSION);
      if ((!delta && (l.type != LIGHT_AREA || l.area.tan_half_spread != FLT_MAX)) ||
          (l.shader_id & (SHADER_EXCLUDE_DIFFUSE | SHADER_EXCLUDE_GLOSSY | SHADER_EXCLUDE_TRANSMIT)) ||
          !(l.shader_id & SHADER_CAST_SHADOW)) {
        device_->set_error("VCM requires full-spread area lights or zero-radius point/spot lights with shadows and surface lobes enabled.");
        return false;
      }
    }
  }
  const bool native_bsdf = device_scene_->vcm_native_bsdf || textured_emitter || (area && point);
  if (final_render && !consumer) {
    const uint end = round_count ? first_round + round_count : rounds;
    const uint stride = uint(producers.size());
    const size_t pixels = size_t(data.cam.width) * size_t(data.cam.height);
    const size_t film_pixels = pixels * (1 + size_t(planes));
    vector<vector<float>> films(stride);
    const double start = time_dt();
    progress_set_status("Rendering OptiX VCM", "Tracing and merging on GPU");
    parallel_for(0u, stride, [&](uint i) {
      const uint first = first_round + (i + stride - first_round % stride) % stride;
      if (first < end) {
        static_cast<PathTraceWorkGPU *>(producers[i])->vcm_render_rounds(
            first, end, stride, count, depth, area, native_bsdf, planes, routes, films[i]);
      }
    });
    if (is_cancel_requested() || device_->have_error()) {
      return false;
    }
    if (vcm_film_.empty()) {
      vcm_film_.assign(pixels * (1 + size_t(planes)), zero_float3());
      if (adaptive) {
        vcm_adaptive_film_.assign(pixels, zero_float3());
      }
    }
    const double scale = double(pixels) / double(count);
    for (uint i = 0; i < stride; ++i) {
      if (first_round + (i + stride - first_round % stride) % stride >= end) {
        continue;
      }
      if (films[i].size() != 3 * (film_pixels + (adaptive ? pixels : 0))) {
        device_->set_error("OptiX VCM did not deliver its completed rounds.");
        return false;
      }
      for (size_t p = 0; p < film_pixels; ++p) {
        const float *v = &films[i][3*p];
        if (adaptive && p < pixels) {
          const float *a = v + 3 * film_pixels;
          vcm_adaptive_film_[p] += make_float3(float(a[0]*scale), float(a[1]*scale), float(a[2]*scale));
        }
        vcm_film_[p] += make_float3(float(v[0]*scale), float(v[1]*scale), float(v[2]*scale));
      }
    }
    LOG_INFO << "OptiX VCM resident rounds [" << first_round << ", " << end << ") on "
             << stride << " GPU(s), " << time_dt() - start << " s";
    return true;
  }
  if (connect && !consumer) {
    device_->set_error("Reference path captures with connection diagnostics require a CPU.");
    return false;
  }
  /* Regression on one physical device: maintain two disjoint ranges, complete them
   * in reverse order, and discard and replay the first delivery before publication. */
  const bool simulate_two = !final_render && producers.size() == 1 && getenv("CYCLES_VCM_TEST_REPLAY");
  const size_t window = simulate_two ? 2 : producers.size();
  struct Probe {
    std::array<double, 3> p, n;
  };
  vector<Probe> probes;
  double radius = 1.0;
  if (const char *source = final_render ? nullptr : getenv("CYCLES_VCM_PROBES")) {
    std::ifstream input(fs::u8path(source));
    if (!(input >> radius) || !std::isfinite(radius) || radius <= 0.0) {
      device_->set_error("VCM probe file must begin with a positive finite world-space radius.");
      return false;
    }
    while (input >> std::ws && !input.eof()) {
      Probe probe;
      if (!(input >> probe.p[0] >> probe.p[1] >> probe.p[2] >> probe.n[0] >> probe.n[1] >>
            probe.n[2]) ||
          !std::isfinite(probe.p[0]) || !std::isfinite(probe.p[1]) || !std::isfinite(probe.p[2]) ||
          !(std::abs(std::hypot(probe.n[0], probe.n[1], probe.n[2]) - 1.0) < 1e-6))
      {
        device_->set_error("Invalid VCM probe position or unit normal.");
        return false;
      }
      probes.push_back(probe);
    }
    if (probes.empty()) {
      device_->set_error("VCM probe file contains no receivers.");
      return false;
    }
  }
  const fs::path directory = final_render ? fs::path{} : fs::u8path(vcm_capture_directory_);
  std::error_code error;
  if (!final_render && !fs::create_directory(directory, error)) {
    device_->set_error("VCM capture needs a new output directory: " + vcm_capture_directory_);
    return false;
  }
  const fs::path temporary = directory /
                             (dump_paths ? "paths.partial" : "connections.pfm.partial");
  const fs::path final = directory / (dump_paths ? "paths.vcm" : "connections.pfm");
  bool published = false;
  const fs::path probe_path = directory / "surface_vm.tsv";
  const fs::path connection_path = directory / "connections.tsv";
  const fs::path image_path = directory / "connections.pfm";
  const fs::path stats_path = directory / "connections_stats.tsv";
  SCOPED_DEFER(if (!final_render && !published) {
    fs::remove(connection_path, error);
    fs::remove(image_path, error);
    fs::remove(stats_path, error);
    fs::remove(temporary, error);
    fs::remove(probe_path, error);
    fs::remove(directory, error);
  });
  const int width = int(data.cam.width);
  const int height = int(data.cam.height);
  const size_t pixels = size_t(width) * height;
  using Channels = std::array<std::array<double, 3>, VCM_TECHNIQUES>;
  std::vector<Channels> film(connect && !final_render ? pixels : 0);
  std::vector<std::array<double, 3>> combined(final_render ? pixels * (1 + size_t(planes)) : 0);
  std::vector<std::array<double, 3>> half_sum(adaptive ? pixels : 0);
  uint64_t shadows = 0, blocked = 0, singular_skipped = 0;
  double weight_error = 0.0, emission_seconds = 0.0, connection_seconds = 0.0, build_seconds = 0.0;
  const bool check_weights = !final_render && getenv("CYCLES_VCM_CHECK_WEIGHTS") != nullptr;
  auto replay = [&](const VCMBatch &batch,
                    uint n,
                    const VCMLightPath *paths,
                    const VCMVertex *vertices,
                    const VCMMergeRound *merging) {
    const double start = time_dt();
    std::vector<VCMConnectionResult> results(n);
    if (planes) {
      for (VCMConnectionResult &r : results) {
        r.route_sources = true;
        r.num_lights = data.integrator.num_lights;
      }
    }
    if (check_weights) {
      for (VCMConnectionResult &r : results) {
        r.check_weights = true;
      }
    }
    if (!consumer->vcm_connect_batch(batch, n, paths, vertices, merging, results.data())) {
      return false;
    }
    const bool half_sample = adaptive && vcm_adaptive_half(batch.round, batch.seed);
    for (size_t i = 0; i < results.size(); ++i) {
      const VCMConnectionResult &r = results[i];
      if (r.failure) {
        device_->set_error("VCM round " + to_string(batch.round) + ", path " +
                           to_string(paths[i].index) + ": " + r.failure + " (object " +
                           to_string(r.object) + ", primitive " + to_string(r.prim) + ").");
        return false;
      }
      for (const VCMFilmSample &sample : r.samples) {
        if (sample.source_light >= -1) {
          const int source = sample.source_light + 1;
          for (int j = routes[source]; j < routes[source + 1]; ++j) {
            auto &target = combined[size_t(routes[j] + 1) * pixels + sample.pixel];
            for (int c = 0; c < 3; ++c) {
              target[c] += sample.rgb[c];
            }
          }
          continue;
        }
        auto &target = final_render ? combined[sample.pixel] : film[sample.pixel][sample.technique];
        for (int c = 0; c < 3; ++c) {
          target[c] += sample.rgb[c];
          if (half_sample) {
            half_sum[sample.pixel][c] += 2.0 * sample.rgb[c];
          }
        }
      }
      shadows += r.shadow;
      blocked += r.blocked;
      singular_skipped += r.singular_skipped;
      weight_error = std::max(weight_error, r.max_weight_error);
    }
    connection_seconds += time_dt() - start;
    return true;
  };
  std::ofstream probe_file;
  if (!probes.empty()) {
    probe_file.open(probe_path);
    probe_file << std::setprecision(17) << "round\tprobe\thits\tr\tg\tb"
               << (simulate_two ? "\tids\n" : "\n");
  }
  std::ofstream file;
  if (dump_paths) {
    file.open(temporary, std::ios::binary);
    const uint header[] = {2, count, rounds, depth, sizeof(VCMLightPath), sizeof(VCMVertex)};
    file.write("CYCVCM02", 8);
    file.write(reinterpret_cast<const char *>(header), sizeof(header));
  }
  const uint end_round = round_count ? first_round + round_count : rounds;
  const size_t num_lights = size_t(data.integrator.num_lights);
  vcm_point_targets_.resize(num_lights);
  vcm_point_fractions_.resize(num_lights, 0.0f);
  const bool fit_points = final_render || !getenv("CYCLES_VCM_TEST_UNGUIDED");
  for (uint round = first_round; round < end_round; ++round) {
    /* Weight, squared weight, and first/second spatial moments of confirmed receivers.
     * An unobserved region always retains the original emitter proposal. */
    struct Moments {
      double weight = 0, squared_weight = 0, second = 0;
      std::array<double, 3> first = {};
    };
    std::vector<Moments> moments(num_lights);
    int guide_light = -1;
    for (size_t j = 0; fit_points && j < num_lights; ++j) {
      const int i = int((round + j) % num_lights);
      if (vcm_point_fractions_[i] > 0.0f) {
        guide_light = i;
        break;
      }
    }
    if (final_render) {
      progress_set_status("Rendering VCM", "Round " + to_string(round + 1) + "/" + to_string(rounds));
    }
    VCMSurfaceMap surface_map(radius, count);
    std::unique_ptr<VCMMergeRound> merging;
    if (initial_radius > 0) {
      /* Progressive surface radius, alpha = 2/3. Indexes and PDFs share this exact radius. */
      merging = std::make_unique<VCMMergeRound>(initial_radius * std::pow(round + 1.0, -1.0 / 6.0),
                                                count);
      if (!(merging->eta > 0.0) || !std::isfinite(merging->eta) ||
          !std::isfinite(1.0 / merging->eta))
      {
        device_->set_error("VCM disk normalization exceeds floating-point range.");
        return false;
      }
    }
    uint first = 0;
    while (first < count) {
      if (is_cancel_requested()) {
        return false;
      }
      /* Bounded window. Workers may finish in any order; only the contiguous, validated
       * prefix is written, and the capture is published only after every round completes. */
      vector<VCMTransfer> transfers;
      for (size_t i = 0; i < window && first < count; ++i) {
        const uint n = producers[simulate_two ? 0 : i]->vcm_batch_size(
            std::min(batch_size, count - first));
        if (n == 0) {
          device_->set_error("VCM producer has no available path slots.");
          return false;
        }
        transfers.emplace_back();
        VCMTransfer &part = transfers.back();
        part.batch = {first, round, uint(data.integrator.seed), depth};
        part.batch.native_bsdf = native_bsdf;
        if (guide_light >= 0) {
          part.batch.guide_light = guide_light;
          part.batch.guide_target = make_float3(vcm_point_targets_[guide_light]);
          part.batch.guide_sigma = vcm_point_targets_[guide_light].w;
          part.batch.guide_fraction = vcm_point_fractions_[guide_light];
        }
        part.paths.resize(n);
        part.vertices.resize(size_t(n) * depth);
        first += n;
      }
      auto emit = [&](size_t i) {
        VCMTransfer &part = transfers[i];
        PathTraceWork *producer = producers[simulate_two ? 0 : i];
        part.ready = producer->vcm_emit_batch(
            part.batch, uint(part.paths.size()), part.paths.data(), part.vertices.data());
      };
      const double emission_start = connect ? time_dt() : 0.0;
      if (simulate_two) {
        for (size_t i = transfers.size(); i > 0; --i) {
          emit(i - 1);
        }
        if (transfers.front().batch.first == 0 && transfers.front().ready) {
          emit(0);
        }
      }
      else {
        parallel_for(size_t(0), transfers.size(), emit);
      }
      if (connect) {
        emission_seconds += time_dt() - emission_start;
      }
      const double build_start = time_dt();
      for (const auto &part : transfers) {
        if (!part.ready) {
          if (!is_cancel_requested() && !device_->have_error()) {
            device_->set_error("VCM device did not complete its assigned emission range.");
          }
          return false;
        }
        for (size_t i = 0; i < part.paths.size(); ++i) {
          const VCMLightPath &path = part.paths[i];
          if (path.index != part.batch.first + i || path.round != round || path.count > depth ||
              path.status < VCM_PATH_ESCAPE || path.status > VCM_PATH_ABSORBED)
          {
            device_->set_error("VCM capture rejected path " + to_string(part.batch.first + i) +
                               " in round " + to_string(round) + ", status " +
                               to_string(path.status) + ". Check emitter and surface support.");
            return false;
          }
          const VCMVertex *vertices = part.vertices.data() + i * depth;
          if (dump_paths) {
            file.write(reinterpret_cast<const char *>(&path), sizeof(path));
            file.write(reinterpret_cast<const char *>(vertices), path.count * sizeof(VCMVertex));
          }
          if (fit_points && path.count > 1 && path.light >= 0) {
            for (uint j = 1; j < path.count; ++j) {
              if (!(vertices[j].label & VCM_VERTEX_RECEIVER)) {
                continue;
              }
              const double w = double(average(float3(vertices[j].flux)));
              const float3 offset = float3(vertices[0].position) - float3(path.source);
              Moments &m = moments[path.light];
              m.weight += w;
              m.squared_weight += w*w;
              m.second += w * (double(offset.x)*offset.x + double(offset.y)*offset.y + double(offset.z)*offset.z);
              for (int axis = 0; axis < 3; ++axis) {
                m.first[axis] += w * offset[axis];
              }
              break;
            }
          }
          if (merging && !merging->append(path, vertices)) {
            device_->set_error("VCM merge position exceeds the precision of the grid.");
            return false;
          }
          if (!probes.empty()) {
            for (uint j = 0; j < path.count; ++j) {
              if (!surface_map.insert(path.index, j, vertices[j])) {
                device_->set_error("VCM photon position exceeds the precision of the grid.");
                return false;
              }
            }
          }
        }
        if (connect && !merging &&
            !replay(part.batch,
                    uint(part.paths.size()),
                    part.paths.data(),
                    part.vertices.data(),
                    nullptr))
        {
          return false;
        }
      }
      build_seconds += time_dt() - build_start;
      if (dump_paths && !file) {
        device_->set_error("Could not write VCM light paths.");
        return false;
      }
    }
    for (size_t i = 0; i < num_lights; ++i) {
      const Moments &m = moments[i];
      if (!(m.squared_weight > 0)) {
        continue;
      }
      const double ess = m.weight*m.weight / m.squared_weight;
      if (ess < 16.0) {
        continue;
      }
      const float3 mean = make_float3(float(m.first[0]/m.weight), float(m.first[1]/m.weight),
                                      float(m.first[2]/m.weight));
      const double variance = (m.second/m.weight - double(len_squared(mean)))/3.0;
      const float sigma = float(std::sqrt(std::max(variance, double(len_squared(mean))*1e-4)));
      if (isfinite_safe(mean) && std::isfinite(sigma) && sigma > 0) {
        vcm_point_targets_[i] = make_float4(mean, sigma);
        vcm_point_fractions_[i] = float(0.5 * ess/(ess + 64.0));
      }
    }
    if (merging) {
      for (uint offset = 0; offset < count; offset += batch_size) {
        VCMBatch batch = {offset, round, uint(data.integrator.seed), depth};
        batch.native_bsdf = native_bsdf;
        if (is_cancel_requested() || !replay(batch,
                                             std::min(batch_size, count - offset),
                                             &merging->paths[offset],
                                             nullptr,
                                             merging.get()))
        {
          return false;
        }
      }
    }
    for (size_t i = 0; i < probes.size(); ++i) {
      std::array<double, 3> rgb;
      uint64_t hits;
      std::vector<uint64_t> ids;
      if (!surface_map.radiance(
              probes[i].p, probes[i].n, rgb, hits, simulate_two ? &ids : nullptr))
      {
        device_->set_error("VCM receiver exceeds the precision of the grid.");
        return false;
      }
      probe_file << round << '\t' << i << '\t' << hits << '\t' << rgb[0] << '\t' << rgb[1] << '\t'
                 << rgb[2];
      if (simulate_two) {
        probe_file << '\t';
        for (const uint64_t id : ids) {
          probe_file << id << ' ';
        }
      }
      probe_file << '\n';
    }
  }
  if (!probes.empty()) {
    probe_file.close();
    if (!probe_file) {
      device_->set_error("Could not write VCM probe results.");
      return false;
    }
  }
  if (final_render) {
    /* PassAccessor normalizes the sum by the number of completed global rounds. */
    const double scale = double(pixels) / double(count);
    if (vcm_film_.empty()) {
      vcm_film_.assign(pixels * (1 + size_t(planes)), zero_float3());
      if (adaptive) {
        vcm_adaptive_film_.assign(pixels, zero_float3());
      }
    }
    for (size_t i = 0; i < combined.size(); ++i) {
      if (adaptive && i < pixels) {
        const auto &a = half_sum[i];
        vcm_adaptive_film_[i] += make_float3(float(a[0]*scale), float(a[1]*scale), float(a[2]*scale));
      }
      const auto &rgb = combined[i];
      vcm_film_[i] += make_float3(float(rgb[0] * scale),
                                  float(rgb[1] * scale),
                                  float(rgb[2] * scale));
    }
    LOG_INFO << "VCM rounds [" << first_round << ", " << end_round << ") emission/readback "
             << emission_seconds << " s, validation/index " << build_seconds
             << " s, camera/VC/VM " << connection_seconds << " s";
    return !is_cancel_requested();
  }
  if (connect) {
    std::ofstream table(connection_path), stats(stats_path),
        image(dump_paths ? image_path : temporary, std::ios::binary);
    table << std::setprecision(17) << "x\ty";
    const int techniques = initial_radius > 0 ? VCM_TECHNIQUES : VCM_SURFACE_MERGE;
    const char *names[VCM_TECHNIQUES] = {"camera_hit", "nee", "paired", "splat", "merge"};
    for (int t = 0; t < techniques; ++t) {
      const char *name = names[t];
      for (char channel : {'r', 'g', 'b'}) {
        table << '\t' << name << '_' << channel;
      }
    }
    table << '\n';
    image << "PF\n"
          << int(data.cam.width) << ' ' << int(data.cam.height) << '\n'
          << (std::endian::native == std::endian::little ? "-1.0\n" : "1.0\n");
    const double scale = double(pixels) / (double(count) * rounds);
    for (size_t i = 0; i < pixels; ++i) {
      if (is_cancel_requested()) {
        return false;
      }
      table << i % size_t(data.cam.width) << '\t' << i / size_t(data.cam.width);
      std::array<double, 3> sum = {};
      for (int t = 0; t < techniques; ++t) {
        const auto &strategy = film[i][t];
        for (int c = 0; c < 3; ++c) {
          const double value = strategy[c] * scale;
          table << '\t' << value;
          sum[c] += value;
        }
      }
      const float rgb[3] = {float(sum[0]), float(sum[1]), float(sum[2])};
      image.write(reinterpret_cast<const char *>(rgb), sizeof(rgb));
      table << '\n';
    }
    const size_t optix_count = std::count_if(producers.begin(), producers.end(), [](auto *p) {
      return p->get_device()->info.type == DEVICE_OPTIX;
    });
    stats << "pairs\tshadow_segments\tblocked\tdelta_group_skips\tmax_weight_error"
             "\toptix_producers\tcpu_producers\temission_seconds\tconnection_seconds\n"
          << uint64_t(rounds) * count << '\t' << shadows << '\t' << blocked << '\t'
          << singular_skipped << '\t' << std::setprecision(17) << weight_error << '\t'
          << optix_count << '\t' << producers.size() - optix_count << '\t' << emission_seconds
          << '\t' << connection_seconds << '\n';
    table.close();
    stats.close();
    image.close();
    if (!table || !stats || !image) {
      device_->set_error("Could not write VCM connection results.");
      return false;
    }
  }
  if (dump_paths) {
    file.close();
  }
  if ((!dump_paths || file) && !is_cancel_requested()) {
    fs::rename(temporary, final, error);
    published = !error;
  }
  if (!published && !is_cancel_requested()) {
    device_->set_error("Could not publish the completed VCM light paths.");
  }
  return published;
}

CCL_NAMESPACE_END
