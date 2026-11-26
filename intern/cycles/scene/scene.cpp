/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#include <cstdlib>

#include "bvh/bvh.h"

#include "device/device.h"
#include "device/photon.h"

#include "scene/background.h"
#include "scene/bake.h"
#include "scene/camera.h"
#include "scene/curves.h"
#include "scene/devicescene.h"
#include "scene/film.h"
#include "scene/hair.h"
#include "scene/integrator.h"
#include "scene/light.h"
#include "scene/mesh.h"
#include "scene/object.h"
#include "scene/osl.h"
#include "scene/particles.h"
#include "scene/pointcloud.h"
#include "scene/procedural.h"
#include "scene/scene.h"
#include "scene/scene_attributes.h"
#include "scene/shader.h"
#include "scene/shader_graph.h"
#include "scene/svm.h"
#include "scene/tables.h"
#include "scene/volume.h"

#include "session/session.h"

#include "util/guarded_allocator.h"
#include "util/log.h"
#include "util/lpe_query.h"
#include "util/progress.h"

CCL_NAMESPACE_BEGIN

Scene ::Scene(const SceneParams &params_, Device *device)
    : name("Scene"),
      default_surface(nullptr),
      default_volume(nullptr),
      default_light(nullptr),
      default_background(nullptr),
      default_empty(nullptr),
      device(device),
      dscene(device),
      params(params_),
      update_stats(nullptr),
      kernels_loaded(false),
      /* TODO(sergey): Check if it's indeed optimal value for the split kernel.
       */
      max_closure_global(1)
{
  memset((void *)&dscene.data, 0, sizeof(dscene.data));

  osl_manager = make_unique<OSLManager>(device);
  shader_manager = ShaderManager::create(device->info.has_osl ? params.shadingsystem :
                                                                SHADINGSYSTEM_SVM);

  light_manager = make_unique<LightManager>();
  geometry_manager = make_unique<GeometryManager>();
  object_manager = make_unique<ObjectManager>();
  image_manager = make_unique<ImageManager>(device->info, params);
  particle_system_manager = make_unique<ParticleSystemManager>();
  bake_manager = make_unique<BakeManager>();
  procedural_manager = make_unique<ProceduralManager>();
  volume_manager = make_unique<VolumeManager>();

  /* Create nodes after managers, since create_node() can tag the managers. */
  camera = create_node<Camera>();
  dicing_camera = create_node<Camera>();
  lookup_tables = make_unique<LookupTables>();
  film = create_node<Film>();
  background = create_node<Background>();
  integrator = create_node<Integrator>();
  scene_attribute = create_node<SceneAttributes>();

  ccl::Film::add_default(this);
  ccl::ShaderManager::add_default(this);
}

Scene::~Scene()
{
  free_memory(true);
}

void Scene::free_memory(bool final)
{
  bvh.reset();

  /* The order of deletion is important to make sure data is freed based on
   * possible dependencies as the Nodes' reference counts are decremented in the
   * destructors:
   *
   * - Procedurals can create and hold pointers to any other types.
   * - Objects can hold pointers to Geometries and ParticleSystems
   * - Lights and Geometries can hold pointers to Shaders.
   *
   * Similarly, we first delete all nodes and their associated device data, and
   * then the managers and their associated device data.
   */
  procedurals.clear();
  objects.clear();
  geometry.clear();
  particle_systems.clear();
  passes.clear();

  if (device) {
    camera->device_free(device, &dscene, this);
    film->device_free(device, &dscene, this);
    background->device_free(device, &dscene);
    integrator->device_free(device, &dscene, true);
    scene_attribute->device_free(device, &dscene, true);
  }

  if (final) {
    cameras.clear();
    integrators.clear();
    scene_attributes.clear();
    films.clear();
    backgrounds.clear();

    camera = nullptr;
    dicing_camera = nullptr;
    integrator = nullptr;
    scene_attribute = nullptr;
    film = nullptr;
    background = nullptr;
  }

  /* Delete Shaders after every other nodes to ensure that we do not try to
   * decrement the reference count on some dangling pointer. */
  shaders.clear();

  /* Now that all nodes have been deleted, we can safely delete managers and
   * device data. */
  if (device) {
    dscene.photon_caustics_casters.free();
    dscene.data.integrator.photon_caustics_caster_count = 0;
    object_manager->device_free(device, &dscene, true);
    geometry_manager->device_free(device, &dscene, true);
    shader_manager->device_free(device, &dscene, this);
    osl_manager->device_free(device, &dscene, this);
    light_manager->device_free(device, &dscene);
    particle_system_manager->device_free(device, &dscene);
    bake_manager->device_free(device, &dscene);
    volume_manager->device_free(&dscene);

    if (final) {
      image_manager->device_free(this);
    }
    else {
      image_manager->device_free_builtin(this);
    }

    lookup_tables->device_free(device, &dscene);
  }

  if (final) {
    lookup_tables.reset();
    object_manager.reset();
    geometry_manager.reset();
    shader_manager.reset();
    osl_manager.reset();
    light_manager.reset();
    particle_system_manager.reset();
    image_manager.reset();
    bake_manager.reset();
    update_stats.reset();
    procedural_manager.reset();
    volume_manager.reset();
  }
}

/* Missing participants are an inactive feature, not an invalid render. Inspect the
 * evaluated scene after light enable flags and mesh bounds have been updated. */
static bool photon_caustics_caster_events_enabled(const Scene *scene, const Geometry *geometry)
{
  for (const Node *node : geometry->get_used_shaders()) {
    const Shader *shader = static_cast<const Shader *>(node);
    if ((scene->integrator->get_caustics_reflective() && shader->get_caustics_reflective()) ||
        (scene->integrator->get_caustics_refractive() && shader->get_caustics_refractive()))
    {
      return true;
    }
  }
  return false;
}

static bool photon_light_enabled(const Scene *scene, const Object *emitter)
{
  const Light *light = static_cast<const Light *>(emitter->get_geometry());
  return emitter->get_visible_photon() && light->get_use_caustics() &&
         light->get_photon_strength() > 0.0f && light->enabled_for_object(emitter) &&
         (!light->is_background_light() || scene->dscene.data.background.photon_sun.w > 0.0f);
}

static bool photon_caustics_caster_is_linked(const Scene *scene, const Object *caster)
{
  const uint64_t receiver_mask = uint64_t(1) << caster->get_receiver_light_set();
  for (const Object *emitter : scene->objects) {
    if (!emitter->get_geometry()->is_light()) continue;
    if (photon_light_enabled(scene, emitter) &&
        (emitter->get_light_set_membership() & receiver_mask))
    {
      return true;
    }
  }
  return false;
}

static bool photon_caustics_has_participants(Scene *scene, string &notices, bool &has_area_spread)
{
  bool has_light = false, has_caster = false;
  for (Object *object : scene->objects) {
    Geometry *geom = object->get_geometry();
    if (geom->is_light()) {
      if (photon_light_enabled(scene, object)) {
        has_light = true;
        if (static_cast<const Light *>(geom)->is_area_light()) {
          has_area_spread |= static_cast<const AreaLight *>(geom)->get_spread() > 0.0f;
        }
      }
    }
    else if (object->get_is_caustics_caster() && !geom->is_mesh()) {
      if (!notices.empty()) {
        notices += "\n";
      }
      notices += string(reinterpret_cast<const char *>(u8"已跳过非网格焦散投射体：")) +
                 object->name.c_str();
    }
    else if (object->is_traceable() && object->bounds.valid() && geom->is_mesh() &&
             (geom->primitive_type() & PRIMITIVE_TRIANGLE) &&
             static_cast<const Mesh *>(geom)->num_triangles() > 0)
    {
      has_caster |= object->get_is_caustics_caster() &&
                    photon_caustics_caster_events_enabled(scene, geom) &&
                    photon_caustics_caster_is_linked(scene, object);
    }
  }
  return has_light && has_caster;
}

/* A requested photon method must never silently render with a different estimator.
 * Reject unsupported configurations before uploading effective integrator constants. */
static const char *photon_caustics_unavailable_reason(Scene *scene,
                                                      const uint64_t /*kernel_features*/)
{
  if (!photon_device_supported(scene->device->info)) {
    return "requires CPU, one CUDA/OptiX GPU, or identical OptiX GPUs (no CPU mixing)";
  }
  if (photon_sample_parallel(scene->device->info) && scene->film->get_use_deep_output()) {
    return "does not yet support multi-GPU photon rendering in Deep EXR";
  }
  if (scene->need_motion() == Scene::MOTION_BLUR &&
      scene->dscene.data.cam.rolling_shutter_type != 0)
  {
    return "does not support rolling shutter with time-coherent photon maps";
  }
  if (scene->bake_manager->get_baking()) {
    return "does not support baking";
  }
  if (scene->shader_manager->use_osl()) {
    return "does not support OSL";
  }
  for (Object *object : scene->objects) {
    const Geometry *geom = object->get_geometry();
    /* Camera visibility belongs to the native camera prefix, which the photon branch
     * also traces. It does not change the light-to-receiver subpath. Unselected lamps
     * remain entirely native and must not disable unrelated photon lights. */
    if (geom->is_light()) {
      if (!photon_light_enabled(scene, object)) {
        continue;
      }
    }
    const uint required_visibility = PATH_RAY_VISIBILITY_ALL & ~PATH_RAY_VISIBILITY_CAMERA;
    if (!geom->is_light() && (object->get_visibility() & required_visibility) != required_visibility) {
      return "requires default object ray visibility";
    }
    if (object->get_is_shadow_catcher() && !geom->is_light()) {
      return "does not support shadow catchers";
    }
    if (!geom->is_light()) {
      continue;
    }

    /* Light linking is applied to the light-adjacent caster, matching native
     * light connections. Blocker linking needs a different traversal. */
    if (object->get_shadow_set_membership() != LIGHT_LINK_MASK_ALL)
    {
      return "does not support shadow linking on caustic lights";
    }

    if (scene->integrator->get_use_photon_motion_blur_correction() &&
        scene->need_motion() == Scene::MOTION_BLUR && object->use_motion() &&
        !Light::motion_is_rigid(object))
    {
      return "supports moving lights with rigid transforms only (no animated scale or shear)";
    }
    Light *light = static_cast<Light *>(object->get_geometry());
    /* A finite emission disk cannot represent propagation from infinity through
     * an unbounded World medium. Mesh and VDB media have explicit boundaries. */
    if (light->is_distant_light() && scene->dscene.data.background.volume_shader != SHADER_NONE) {
      return "requires bounded mesh or VDB volumes for Sun photon caustics";
    }
    if (!light->get_cast_shadow()) {
      return "requires shadow casting on caustic lights";
    }
    if (light->is_point_light() || light->is_spot_light()) {
      const PointLight *point_light = static_cast<const PointLight *>(light);
      if (!isfinite_safe(point_light->get_radius()) || point_light->get_radius() < 0.0f) {
        return "requires finite, nonnegative Point and Spot radii";
      }
    }
    else if (!light->is_sun_light() && !light->is_area_light() && !light->is_background_light()) {
      return "supports only Point, Spot, Sun and Area caustic lights";
    }
    const Shader *shader = light->get_shader() ? light->get_shader() : scene->default_light;
    if (shader->has_surface_raytrace) {
      return "does not support ray-tracing shader nodes on caustic lights";
    }
    if (!is_zero(light->get_shadow_color())) {
      return "does not support colored shadows on caustic lights";
    }
  }
  return nullptr;
}

/* Volume events already carry the currently active object's identity. Disjoint media
 * can therefore be selected independently, even in a scene with many volumes. Validate
 * potential overlap after displacement, instances and motion bounds have been updated,
 * rather than rejecting all multi-medium scenes during Blender synchronization. */
static void validate_lpe_volume_selections(Scene *scene, Progress &progress)
{
  if (scene->film->lpe_volume_selections.empty()) {
    return;
  }
  const Shader *world = scene->background->get_shader(scene);
  const bool world_volume = world && world->has_volume;
  for (const Object *object : scene->objects) {
    const Geometry *geometry = object->get_geometry();
    if (!geometry->has_volume) {
      continue;
    }
    bool selected = false;
    for (const LPESelection &selection : scene->film->lpe_volume_selections) {
      selected |= selection.object == object;
      if (selection.shader) {
        for (const Node *shader : geometry->get_used_shaders()) {
          selected |= shader == selection.shader;
        }
      }
    }
    if (!selected) {
      continue;
    }
    bool overlaps = world_volume || !object->bounds.valid();
    for (const Object *other : scene->objects) {
      if (other == object || !other->get_geometry()->has_volume) {
        continue;
      }
      const float3 size = intersect(object->bounds, other->bounds).size();
      overlaps |= size.x > 0.0f && size.y > 0.0f && size.z > 0.0f;
    }
    if (overlaps) {
      progress.set_error("LPE: Object/Material volume selection for '" + object->name.string() +
                         "' has overlapping volume bounds or a World medium; use Light Volume "
                         "for combined media");
      return;
    }
  }
}

/* Advisory only. Folded shader constants can be checked exactly; spatial inputs
 * cannot be bounded here, so identify them explicitly instead of reporting a
 * fabricated roughness value. Never change eligibility or clamp the material. */
static string photon_roughness_notice(Scene *scene, const Object *object)
{
  if (object->get_photon_caustics_zero_roughness()) {
    return {};
  }
  string notices;
  set<const Node *> seen;
  for (const Node *base : object->get_geometry()->get_used_shaders()) {
    if (!seen.insert(base).second)
      continue;
    const Shader *shader = static_cast<const Shader *>(base);
    if (!shader->graph || shader->get_photon_caustics_zero_roughness())
      continue;
    float roughness = 0.0f;
    bool varying = false;
    for (const ShaderNode *node : shader->graph->nodes) {
      /* Custom OSL closures can carry roughness not exposed as a graph socket. */
      varying |= node->special_type == SHADER_SPECIAL_TYPE_OSL;
      const string type = node->type->name.string();
      if (type != "glossy_bsdf" && type != "glass_bsdf" && type != "refraction_bsdf" &&
          type != "principled_bsdf")
        continue;
      for (const ShaderInput *input : node->inputs) {
        const string name = input->name().string();
        if (name != "Roughness" && name != "Coat Roughness")
          continue;
        if (name == "Coat Roughness") {
          const ShaderInput *weight = nullptr;
          for (const ShaderInput *candidate : node->inputs) {
            if (candidate->name() == "Coat Weight")
              weight = candidate;
          }
          if (weight && !weight->link && node->get_float(weight->socket_type) <= 0.0f)
            continue;
        }
        varying |= input->link != nullptr;
        if (!input->link)
          roughness = max(roughness, node->get_float(input->socket_type));
      }
    }
    scene->dscene.data.integrator.photon_caustics_has_rough |= varying || roughness > 0.0037f;
    if (roughness <= 0.3f && !varying)
      continue;
    if (!notices.empty())
      notices += "\n";
    notices += reinterpret_cast<const char *>(u8"光子焦散提示：物体「") + object->name.string() +
               reinterpret_cast<const char *>(u8"」，材质「") + shader->name.string() +
               reinterpret_cast<const char *>(u8"」：");
    if (roughness > 0.3f) {
      notices += string_printf(reinterpret_cast<const char *>(
                                   u8"粗糙度 %.3f 高于建议参考值 "
                                   u8"0.3；局部光束展宽可能出现模糊或边界误差，仍继续计算。"),
                               roughness);
    }
    else {
      notices += reinterpret_cast<const char *>(
          u8"粗糙度由纹理或节点驱动，无法静态确认范围；较粗糙区域可能出现展宽误差，仍继续计算。");
    }
  }
  return notices;
}

/* Update the camera guide's world-space grid after geometry and object updates. */
static void update_gpu_guiding_bounds(Scene *scene)
{
  KernelIntegrator &integrator = scene->dscene.data.integrator;
  integrator.use_gpu_guiding = scene->integrator->get_use_gpu_guiding();
  if (!integrator.use_gpu_guiding) {
    return;
  }
  BoundBox bounds = BoundBox::empty;
  for (const Object *object : scene->objects) {
    if (object->is_traceable() && object->bounds.valid()) {
      bounds.grow(object->bounds);
    }
  }
  if (!bounds.valid()) {
    integrator.use_gpu_guiding = false;
    return;
  }
  const float3 size = max(bounds.size(), make_float3(1e-4f));
  const float3 origin = bounds.min - size * 0.001f;
  const float3 scale = make_float3(float(GPU_GUIDE_GRID)) / (size * 1.002f);
  integrator.gpu_guiding_min = make_float4(origin.x, origin.y, origin.z, 0.0f);
  integrator.gpu_guiding_scale = make_float4(scale.x, scale.y, scale.z, 0.0f);
}

/* Photon emission targets caster bounds in world space. This must run after geometry and
 * object updates, including displacement, even when the integrator itself was not modified. */
static void update_photon_caustics(Scene *scene, Progress &progress)
{
  KernelIntegrator &kintegrator = scene->dscene.data.integrator;
  progress.set_warning("");
  kintegrator.photon_caustics_has_rough = false;
  kintegrator.photon_caustics_caster_count = 0;
  scene->dscene.photon_caustics_casters.free();
  kintegrator.photon_caustics_target = zero_float4();
  kintegrator.photon_caustics_sun_distance = 0.0f;
  kintegrator.use_photon_caustics = false;
  kintegrator.use_photon_field_guiding = false;
  if (!scene->integrator->get_use_photon_caustics()) {
    return;
  }
  /* An inactive PHOTON selection must not activate native MNEE. */
  kintegrator.use_caustics = false;
  if (!scene->integrator->get_caustics_reflective() &&
      !scene->integrator->get_caustics_refractive())
  {
    progress.set_warning(reinterpret_cast<const char *>(
        u8"光子焦散已跳过：反射和折射焦散均已关闭。继续普通路径追踪。"));
    return;
  }
  string roughness_notice;
  bool has_area_spread = false;
  if (!photon_caustics_has_participants(scene, roughness_notice, has_area_spread)) {
    if (!roughness_notice.empty()) {
      roughness_notice += "\n";
    }
    progress.set_warning(roughness_notice + reinterpret_cast<const char *>(
        u8"光子焦散已跳过：缺少可用的焦散光源或投射物体。继续普通路径追踪。"));
    return;
  }
  /* Emission estimates and effective light enable flags are only current after shader and
   * light device updates. Checking them in update_kernel_features would read the last frame. */
  if (const char *reason = photon_caustics_unavailable_reason(scene,
                                                              scene->dscene.data.kernel_features))
  {
    progress.set_error(string("Photon caustics ") + reason + ".");
    return;
  }

  BoundBox caster_bounds = BoundBox::empty;
  BoundBox scene_bounds = BoundBox::empty;
  vector<int2> caster_indices;
  for (const Object *object : scene->objects) {
    if (!object->is_traceable() || !object->bounds.valid()) {
      continue;
    }
    scene_bounds.grow(object->bounds);
    const Geometry *geom = object->get_geometry();
    if (object->get_is_caustics_caster() && photon_caustics_caster_events_enabled(scene, geom) &&
        photon_caustics_caster_is_linked(scene, object) && geom->is_mesh() &&
        (geom->primitive_type() & PRIMITIVE_TRIANGLE) &&
        static_cast<const Mesh *>(geom)->num_triangles() > 0)
    {
      caster_bounds.grow(object->bounds);
      caster_indices.push_back(make_int2(object->get_device_index(), int(geom->prim_offset)));
      const string notice = photon_roughness_notice(scene, object);
      if (!notice.empty()) {
        if (!roughness_notice.empty())
          roughness_notice += "\n";
        roughness_notice += notice;
      }
    }
  }

  if (caster_bounds.valid() && scene_bounds.valid()) {
    const float3 center = caster_bounds.center();
    const float radius = 0.5f * len(caster_bounds.size());
    /* Keep the emission disk fitted to the casters, but start Sun rays before every scene
     * surface. Starting just outside the caster sphere would skip upstream occluders. The
     * farthest scene-bound corner from the caster center bounds the projection in any sampled
     * Sun direction, including finite angular Sun lights. */
    const float3 extent = max(fabs(scene_bounds.min - center), fabs(scene_bounds.max - center));
    const float max_distance = len(extent);
    const float sun_distance = max_distance + max(1e-4f, max_distance * 1e-4f);
    if (isfinite_safe(center) && isfinite_safe(radius) && radius > 0.0f &&
        isfinite_safe(sun_distance))
    {
      /* Device indices and primitive offsets are final only after the object/BVH upload.
       * Store both explicitly: the normal object_prim_offset array may not be allocated
       * when the light tree is disabled. Finite-light proposals only visit marked objects. */
      int2 *device_casters = scene->dscene.photon_caustics_casters.alloc(caster_indices.size());
      for (size_t i = 0; i < caster_indices.size(); ++i) {
        device_casters[i] = caster_indices[i];
      }
      scene->dscene.photon_caustics_casters.copy_to_device();
      kintegrator.photon_caustics_caster_count = int(caster_indices.size());
      kintegrator.photon_caustics_target = make_float4(center.x, center.y, center.z, radius);
      kintegrator.photon_caustics_sun_distance = sun_distance;
      kintegrator.use_photon_caustics = true;
      kintegrator.use_photon_field_guiding = has_area_spread && kintegrator.use_photon_guiding &&
                                            scene->integrator->get_use_photon_field_guiding();
      kintegrator.use_caustics = true;
      progress.set_warning(roughness_notice);
      if (scene->film->get_use_deep_output() &&
          scene->integrator->get_photon_caustics_separate_sampling())
      {
        progress.set_warning(roughness_notice + (roughness_notice.empty() ? "" : "\n") +
                             reinterpret_cast<const char *>(
                                 u8"Deep 输出暂不支持独立光子收敛：沿用联合收敛和原生采样预算。"));
      }
      return;
    }
  }

  progress.set_error("Photon caustics require finite, non-empty scene and caster bounds.");
}

void Scene::device_update(Device *device_, Progress &progress)
{
  /* Scene::update only enters here for modified scene data. Invalidate cached light
   * paths before the upload so an interrupted update cannot reuse the previous map. */
  ++dscene.photon_caustics_revision;

  if (!device) {
    device = device_;
  }

  const bool print_stats = need_data_update();
  bool kernels_reloaded = false;

  while (true) {
    if (update_stats) {
      update_stats->clear();
    }

    const scoped_callback_timer timer([this, print_stats](double time) {
      if (update_stats) {
        update_stats->scene.times.add_entry({"device_update", time});

        if (print_stats) {
          printf("Update statistics:\n%s\n", update_stats->full_report().c_str());
        }
      }
    });

    /* The order of updates is important, because there's dependencies between
     * the different managers, using data computed by previous managers. */

    if (film->update_lightgroups(this)) {
      light_manager->tag_update(this, ccl::LightManager::LIGHT_MODIFIED);
      object_manager->tag_update(this, ccl::ObjectManager::OBJECT_MODIFIED);
      background->tag_modified();
    }
    if (film->exposure_is_modified()) {
      integrator->tag_modified();
    }

    /* Compile shaders and get information about features they used. */
    progress.set_status("Updating Shaders");
    osl_manager->device_update_pre(device, this);
    shader_manager->device_update_pre(device, &dscene, this, progress);

    if (progress.get_cancel() || device->have_error()) {
      return;
    }

    /* Passes. After shader manager as this depends on the shaders. */
    film->update_passes(this);

    /* Update kernel features. After shaders and passes since those affect features. */
    update_kernel_features();

    /* Load render kernels, before uploading most data to the GPU, and before displacement and
     * background light need to run kernels.
     *
     * Do it outside of the scene mutex since the heavy part of the loading (i.e. kernel
     * compilation) does not depend on the scene and some other functionality (like display
     * driver) might be waiting on the scene mutex to synchronize display pass.
     *
     * This does mean the scene might have gotten updated in the meantime, in which case
     * we have to redo the first part of the scene update. */
    const uint64_t kernel_features = dscene.data.kernel_features;
    scene_updated_while_loading_kernels = false;
    if (!kernels_loaded || loaded_kernel_features != kernel_features) {
      mutex.unlock();
      kernels_reloaded |= load_kernels(progress);
      mutex.lock();
    }

    if (progress.get_cancel() || device->have_error()) {
      return;
    }

    if (!scene_updated_while_loading_kernels) {
      break;
    }
  }

  /* Upload shaders to GPU and compile OSL kernels, after kernels have been loaded. */
  shader_manager->device_update_post(device, &dscene, this, progress);
  osl_manager->device_update_post(device, this, progress, kernels_reloaded);

  if (progress.get_cancel() || device->have_error()) {
    return;
  }

  procedural_manager->update(this, progress);

  if (progress.get_cancel()) {
    return;
  }

  progress.set_status("Updating Background");
  background->device_update(device, &dscene, this);

  if (progress.get_cancel() || device->have_error()) {
    return;
  }

  progress.set_status("Updating Scene Attribute");
  scene_attribute->device_update(device, &dscene, this);

  if (progress.get_cancel() || device->have_error()) {
    return;
  }

  /* Camera will be used by adaptive subdivision, so do early. */
  progress.set_status("Updating Camera");
  camera->device_update(device, &dscene, this);

  if (progress.get_cancel() || device->have_error()) {
    return;
  }

  geometry_manager->device_update_preprocess(device, this, progress);
  if (progress.get_cancel() || device->have_error()) {
    return;
  }

  /* Update objects after geometry preprocessing. */
  progress.set_status("Updating Objects");
  object_manager->device_update(device, &dscene, this, progress);

  if (progress.get_cancel() || device->have_error()) {
    return;
  }

  progress.set_status("Updating Particle Systems");
  particle_system_manager->device_update(device, &dscene, this, progress);

  if (progress.get_cancel() || device->have_error()) {
    return;
  }

  /* Camera and shaders must be ready here for adaptive subdivision and displacement. */
  progress.set_status("Updating Meshes");
  geometry_manager->device_update(device, &dscene, this, progress);

  if (progress.get_cancel() || device->have_error()) {
    return;
  }

  /* Update object flags with final geometry. */
  progress.set_status("Updating Objects Flags");
  object_manager->device_update_flags(device, &dscene, this, progress);

  if (progress.get_cancel() || device->have_error()) {
    return;
  }

  /* Update BVH primitive objects with final geometry. */
  progress.set_status("Updating Primitive Offsets");
  object_manager->device_update_prim_offsets(device, &dscene, this);

  if (progress.get_cancel() || device->have_error()) {
    return;
  }

  /* Images last, as they should be more likely to use host memory fallback than geometry.
   * Some images may have been uploaded early for displacement already at this point. */
  progress.set_status("Updating Images");
  image_manager->device_update(device, this, progress);

  if (progress.get_cancel() || device->have_error()) {
    return;
  }

  /* Evaluate volume shader to build volume octrees. */
  progress.set_status("Updating Volume");
  volume_manager->device_update(device, &dscene, this, progress);

  if (progress.get_cancel() || device->have_error()) {
    return;
  }

  progress.set_status("Updating Camera Volume");
  camera->device_update_volume(device, &dscene, this);

  if (progress.get_cancel() || device->have_error()) {
    return;
  }

  progress.set_status("Updating Lookup Tables");
  lookup_tables->device_update(device, &dscene, this);

  if (progress.get_cancel() || device->have_error()) {
    return;
  }

  /* Light manager needs shaders and final meshes for triangles in light tree. */
  progress.set_status("Updating Lights");
  light_manager->device_update(device, &dscene, this, progress);

  if (progress.get_cancel() || device->have_error()) {
    return;
  }

  progress.set_status("Updating Integrator");
  integrator->device_update(device, &dscene, this);

  if (progress.get_cancel() || device->have_error()) {
    return;
  }

  progress.set_status("Updating Film");
  film->device_update(device, &dscene, this);

  if (progress.get_cancel() || device->have_error()) {
    return;
  }

  /* Update lookup tables a second time for film tables. */
  progress.set_status("Updating Lookup Tables");
  lookup_tables->device_update(device, &dscene, this);

  if (progress.get_cancel() || device->have_error()) {
    return;
  }

  progress.set_status("Updating Baking");
  bake_manager->device_update(device, &dscene, this, progress);

  if (progress.get_cancel() || device->have_error()) {
    return;
  }

  if (device->have_error() == false) {
    dscene.data.volume_stack_size = get_volume_stack_size();
    if (dscene.data.kernel_features & KERNEL_FEATURE_NESTED_DIELECTRICS) {
      int medium_count = 0;
      for (const Object *object : objects) {
        medium_count += object->num_nested_dielectric_materials();
      }
      if (medium_count > MAX_VOLUME_STACK_SIZE - 2) {
        progress.set_error("Nested dielectrics: too many enabled object/material regions for the current medium stack; disable unused regions");
        return;
      }
      if (dscene.data.integrator.use_caustics && !integrator->get_use_photon_caustics()) {
        progress.set_error("Nested dielectrics: native MNEE does not yet support explicit medium priorities; use Photon or disable shadow caustics");
        return;
      }
      if (dscene.data.kernel_features & (KERNEL_FEATURE_OSL_SHADING | KERNEL_FEATURE_SUBSURFACE)) {
        progress.set_error("Nested dielectrics: this implementation currently requires SVM surfaces without subsurface scattering");
        return;
      }
    }
    validate_lpe_volume_selections(this, progress);
    if (progress.get_error()) {
      return;
    }
    if (lpe_query_active()) {
      std::map<int, std::string> object_names, shader_names;
      for (const Object *ob : objects) {
        object_names[ob->get_device_index()] = ob->name.c_str();
      }
      for (const Shader *shader : shaders) {
        shader_names[shader->id] = shader->name.c_str();
      }
      lpe_query_names(std::move(object_names), std::move(shader_names), dscene.data.background.object_index);
    }
    update_photon_caustics(this, progress);
    update_gpu_guiding_bounds(this);
    KernelFilm &kfilm = dscene.data.film;
    kfilm.deep_photon_ptr = 0;
    kfilm.deep_photon_bins = 0;
    if (kfilm.use_deep_output && !kfilm.deep_physical && dscene.data.integrator.use_photon_caustics &&
        dscene.data.integrator.use_volumes)
    {
      float lo = FLT_MAX, hi = -FLT_MAX;
      for (const Object *object : objects) {
        if (!object->is_traceable() || !object->bounds.valid() ||
            !object->get_geometry()->has_volume)
        {
          continue;
        }
        for (int corner = 0; corner < 8; ++corner) {
          const BoundBox &b = object->bounds;
          const float3 P = make_float3((corner & 1) ? b.max.x : b.min.x,
                                       (corner & 2) ? b.max.y : b.min.y,
                                       (corner & 4) ? b.max.z : b.min.z);
          const float z = camera->get_camera_type() == CAMERA_PANORAMA ?
                              len(P - transform_get_column(&camera->cameratoworld, 3)) :
                              transform_point(&camera->worldtocamera, P).z;
          lo = min(lo, z);
          hi = max(hi, z);
        }
      }
      /* Unbounded world volumes use the camera clipping range. */
      const Shader *world = background->get_shader(this);
      if ((world && world->has_volume) || !(hi > max(lo, 0.0f))) {
        lo = camera->get_nearclip();
        hi = camera->get_farclip();
      }
      kfilm.deep_photon_z_min = camera->get_camera_type() == CAMERA_PANORAMA ? 0.0f :
                                                                               max(0.0f, lo);
      kfilm.deep_photon_z_max = max(kfilm.deep_photon_z_min + 1e-4f, hi);
      kfilm.deep_photon_bins = clamp(film->get_deep_photon_samples(), 8, 256);
    }
    if (progress.get_error()) {
      return;
    }

    progress.set_status("Updating Device", "Writing constant memory");
    device->const_copy_to("data", &dscene.data, sizeof(dscene.data));
  }

  device->optimize_for_scene(this);

  if (need_motion() == MOTION_PASS_INTERACTIVE) {
    /* Swap current camera/object/vertex positions to previous positions for next frame. */
    camera->update_interactive_motion();
    object_manager->update_interactive_motion(this);
    geometry_manager->update_interactive_motion(this);
  }

  if (print_stats) {
    const size_t mem_used = util_guarded_get_mem_used();
    const size_t mem_peak = util_guarded_get_mem_peak();

    LOG_INFO << "System memory statistics after full device sync:\n"
             << "  Usage: " << string_human_readable_number(mem_used) << " ("
             << string_human_readable_size(mem_used) << ")\n"
             << "  Peak: " << string_human_readable_number(mem_peak) << " ("
             << string_human_readable_size(mem_peak) << ")";
  }
}

void Scene::set_motion_blur_correction(const bool enabled)
{
  if (bool(dscene.data.integrator.use_motion_blur_correction) == enabled) {
    return;
  }
  dscene.data.integrator.use_motion_blur_correction = enabled;
  camera->set_motion_pivot(enabled ? camera->shutter_mount_pivot : zero_float3());
  /* The active estimator's shutter table can change even for a stationary camera. */
  if (need_motion() == MOTION_BLUR) {
    camera->tag_modified();
  }
  for (Object *object : objects) {
    const float3 pivot = enabled ? object->shutter_mount_pivot : zero_float3();
    if (object->get_motion_pivot() != pivot) {
      object->set_motion_pivot(pivot);
      object->tag_update(this);
    }
  }
  light_manager->tag_update(this, LightManager::UPDATE_ALL);
}

bool Scene::needs_separate_motion() const
{
  if (need_motion() != MOTION_BLUR || !integrator->get_use_photon_caustics() ||
      integrator->get_use_relative_motion_blur() == integrator->get_use_photon_motion_blur_correction())
  {
    return false;
  }
  const auto has_motion = [](const array<Transform> &keys, const Transform center) {
    return keys.size() > 1 && std::any_of(keys.begin(), keys.end(), [&](const Transform &key) {
      return key != transform_empty() && key != center;
    });
  };
  /* A shaped shutter also changes time sampling for deformation with static object transforms. */
  const array<float> &curve = camera->get_shutter_curve();
  if (curve.size() > 1 &&
      std::any_of(curve.begin() + 1, curve.end(), [&](float value) { return value != curve[0]; }))
  {
    return true;
  }
  if (has_motion(camera->get_motion(), camera->get_matrix())) {
    return true;
  }
  return std::any_of(objects.begin(), objects.end(), [&](const Object *object) {
    return has_motion(object->get_motion(), object->get_tfm());
  });
}

Scene::MotionType Scene::need_motion() const
{
  if (integrator->get_motion_blur()) {
    return MOTION_BLUR;
  }
  const DenoiserPassMask denoiser_motion_passes = DENOISER_PASS_BACKWARD_MOTION |
                                                  DENOISER_PASS_SPECULAR_MOTION;
  const bool denoiser_motion = (integrator->get_use_denoise()) &&
                               (integrator->get_denoiser_passes() & denoiser_motion_passes) != 0;
  if (denoiser_motion || (Pass::contains(passes, PASS_MOTION) ||
                          Pass::contains(passes, PASS_DENOISING_BACKWARD_MOTION) ||
                          Pass::contains(passes, PASS_DENOISING_SPECULAR_MOTION)))
  {
    return params.background ? MOTION_PASS : MOTION_PASS_INTERACTIVE;
  }
  return MOTION_NONE;
}

float Scene::motion_shutter_time()
{
  if (need_motion() == Scene::MOTION_PASS || need_motion() == Scene::MOTION_PASS_INTERACTIVE) {
    return 2.0f;
  }
  return camera->get_shuttertime();
}

bool Scene::need_global_attribute(AttributeStandard std) const
{
  if (std == ATTR_STD_UV) {
    return Pass::contains(passes, PASS_UV);
  }
  if (std == ATTR_STD_VOLUME_VELOCITY || std == ATTR_STD_VOLUME_VELOCITY_X ||
      std == ATTR_STD_VOLUME_VELOCITY_Y || std == ATTR_STD_VOLUME_VELOCITY_Z)
  {
    return need_motion() != MOTION_NONE;
  }

  return false;
}

void Scene::need_global_attributes(AttributeRequestSet &attributes)
{
  for (int std = ATTR_STD_NONE; std < ATTR_STD_NUM; std++) {
    if (need_global_attribute((AttributeStandard)std)) {
      attributes.add((AttributeStandard)std);
    }
  }

  for (const Shader *shader : shaders) {
    attributes.add(shader->global_attributes);
  }
}

bool Scene::need_update()
{
  return (need_reset() || film->is_modified());
}

bool Scene::need_data_update()
{
  return (
      background->is_modified() || image_manager->need_update() || object_manager->need_update() ||
      geometry_manager->need_update() || light_manager->need_update() ||
      lookup_tables->need_update() || integrator->is_modified() || shader_manager->need_update() ||
      particle_system_manager->need_update() || bake_manager->need_update() ||
      film->is_modified() || procedural_manager->need_update() || scene_attribute->is_modified());
}

bool Scene::need_reset(const bool check_camera)
{
  return need_data_update() || (check_camera && camera->is_modified());
}

void Scene::reset()
{
  osl_manager->reset(this);
  ShaderManager::add_default(this);

  /* ensure all objects are updated */
  camera->tag_modified();
  dicing_camera->tag_modified();
  film->tag_modified();
  background->tag_modified();

  background->tag_update(this);
  integrator->tag_update(this, Integrator::UPDATE_ALL);
  scene_attribute->tag_update(this, SceneAttributes::UPDATE_ALL);
  object_manager->tag_update(this, ObjectManager::UPDATE_ALL);
  geometry_manager->tag_update(this, GeometryManager::UPDATE_ALL);
  light_manager->tag_update(this, LightManager::UPDATE_ALL);
  particle_system_manager->tag_update(this);
  procedural_manager->tag_update();
}

void Scene::device_free()
{
  free_memory(false);
}

void Scene::collect_statistics(RenderStats *stats)
{
  geometry_manager->collect_statistics(this, stats);
  image_manager->collect_statistics(stats, this);
}

void Scene::enable_update_stats()
{
  if (!update_stats) {
    update_stats = make_unique<SceneUpdateStats>();
  }
}

void Scene::update_kernel_features()
{
  if (!need_update()) {
    return;
  }

  /* These features are not being tweaked as often as shaders,
   * so could be done selective magic for the viewport as well. */
  uint64_t kernel_features = shader_manager->get_kernel_features(this);

  const bool use_motion = need_motion() == Scene::MotionType::MOTION_BLUR;
  kernel_features |= KERNEL_FEATURE_PATH_TRACING;

  /* Track the max prim count in case the backend needs to rebuild BVHs or
   * kernels to support different limits. */
  size_t kernel_max_prim_count = 0;

  /* Figure out whether the scene will use shader ray-trace we need at least
   * one caustic light, one caustic caster and one caustic receiver to use
   * and enable the MNEE code path. */
  bool has_caustics_receiver = false;
  bool has_caustics_caster = false;
  bool has_caustics_light = false;

  for (Object *object : objects) {
    if (object->has_nested_dielectric()) {
      kernel_features |= KERNEL_FEATURE_NESTED_DIELECTRICS | KERNEL_FEATURE_VOLUME |
                         KERNEL_FEATURE_TRANSPARENT;
    }
    if (object->get_is_caustics_caster()) {
      has_caustics_caster = true;
    }
    else if (object->get_is_caustics_receiver()) {
      has_caustics_receiver = true;
    }
    Geometry *geom = object->get_geometry();
    if (use_motion) {
      if (object->use_motion() || geom->get_use_motion_blur()) {
        kernel_features |= KERNEL_FEATURE_OBJECT_MOTION;
      }
    }
    if (object->get_is_shadow_catcher() && !geom->is_light()) {
      kernel_features |= KERNEL_FEATURE_SHADOW_CATCHER;
    }
    if (geom->is_hair()) {
      const Hair *hair = static_cast<const Hair *>(geom);
      kernel_features |= (hair->curve_shape == CURVE_RIBBON) ? KERNEL_FEATURE_HAIR_RIBBON :
                                                               KERNEL_FEATURE_HAIR_THICK;
      kernel_max_prim_count = max(kernel_max_prim_count, hair->num_segments());
    }
    else if (geom->is_pointcloud()) {
      const PointCloud *pointcloud = static_cast<const PointCloud *>(geom);
      kernel_features |= KERNEL_FEATURE_POINTCLOUD;
      if (pointcloud->primitive_type() & PRIMITIVE_GSPLAT) {
        kernel_features |= KERNEL_FEATURE_GSPLATS;
      }
      kernel_max_prim_count = max(kernel_max_prim_count, pointcloud->num_points());
    }
    else if (geom->is_mesh()) {
      kernel_max_prim_count = max(kernel_max_prim_count,
                                  static_cast<Mesh *>(geom)->num_triangles());
    }
    else if (geom->is_light()) {
      const Light *light = static_cast<const Light *>(object->get_geometry());
      if (light->get_use_caustics()) {
        has_caustics_light = true;
      }
    }
    if (object->has_light_linking()) {
      kernel_features |= KERNEL_FEATURE_LIGHT_LINKING;
    }
    if (object->has_shadow_linking()) {
      kernel_features |= KERNEL_FEATURE_SHADOW_LINKING;
    }
  }

  dscene.data.integrator.use_caustics = false;
  if (!integrator->get_use_photon_caustics() && device->info.has_mnee() && has_caustics_caster &&
      has_caustics_receiver && has_caustics_light)
  {
    dscene.data.integrator.use_caustics = true;
    kernel_features |= KERNEL_FEATURE_MNEE;
  }

  dscene.data.integrator.use_photon_caustics = false;
  /* Reserve the shared caustics kernels before shader/device uploads. Runtime eligibility is
   * resolved by update_photon_caustics after shaders, lights and object bounds are current. */
  if (integrator->get_use_photon_caustics() && has_caustics_caster && has_caustics_light &&
      photon_device_supported(device->info))
  {
    kernel_features |= KERNEL_FEATURE_MNEE;
  }

  if (integrator->get_guiding_params(device).use) {
    kernel_features |= KERNEL_FEATURE_PATH_GUIDING;
  }
  bool nvidia_device = device->info.type == DEVICE_CUDA || device->info.type == DEVICE_OPTIX;
  for (const DeviceInfo &subdevice : device->info.multi_devices) {
    nvidia_device |= subdevice.type == DEVICE_CUDA || subdevice.type == DEVICE_OPTIX;
  }
  if (integrator->get_use_gpu_guiding() && nvidia_device && !bake_manager->get_baking()) {
    kernel_features |= KERNEL_FEATURE_GPU_PATH_GUIDING;
  }

  if (bake_manager->get_baking()) {
    kernel_features |= KERNEL_FEATURE_BAKING;
  }

  kernel_features |= film->get_kernel_features(this);
  kernel_features |= integrator->get_kernel_features();
  kernel_features |= camera->get_kernel_features();

  dscene.data.kernel_features = kernel_features;

  /* Currently viewport render is faster with higher max_closures, needs
   * investigating. */
  const uint max_closures = (params.background) ? get_max_closure_count() : MAX_CLOSURE;
  dscene.data.max_closures = max_closures;
  dscene.data.max_shaders = shaders.size();

  /* Inform the device of the BVH limits. If this returns true, all BVHs
   * and kernels need to be rebuilt. */
  if (device->set_bvh_limits(objects.size(), kernel_max_prim_count)) {
    kernels_loaded = false;
    for (Geometry *geom : geometry) {
      geom->need_update_rebuild = true;
      geom->tag_modified();
    }
  }
}

bool Scene::update(Progress &progress)
{
  if (!need_update()) {
    return false;
  }

  /* Upload scene data to the GPU. */
  progress.set_status("Updating Scene");
  MEM_GUARDED_CALL(&progress, device_update, device, progress);

  return true;
}

bool Scene::update_camera_resolution(Progress &progress, int width, int height)
{
  bool update_data = false;

  if (camera->set_screen_size(width, height)) {
    camera->device_update(device, &dscene, this);
    update_data = true;
  }

  if (integrator->get_use_pixel_jitter()) {
    integrator->tag_use_pixel_jitter_modified();

    integrator->device_update(device, &dscene, this);
    update_data = true;
  }

  if (update_data) {
    progress.set_status("Updating Device", "Writing constant memory");
    device->const_copy_to("data", &dscene.data, sizeof(dscene.data));
  }
  return update_data;
}

static void log_kernel_features(const uint64_t features)
{
  LOG_INFO << "Requested features:";
  LOG_INFO << "Use BSDF " << string_from_bool(features & KERNEL_FEATURE_NODE_BSDF);
  LOG_INFO << "Use Emission " << string_from_bool(features & KERNEL_FEATURE_NODE_EMISSION);
  LOG_INFO << "Use Volume " << string_from_bool(features & KERNEL_FEATURE_NODE_VOLUME);
  LOG_INFO << "Use Bump " << string_from_bool(features & KERNEL_FEATURE_NODE_BUMP);
  LOG_INFO << "Use Voronoi " << string_from_bool(features & KERNEL_FEATURE_NODE_VORONOI_EXTRA);
  LOG_INFO << "Use Shader Raytrace " << string_from_bool(features & KERNEL_FEATURE_NODE_RAYTRACE);
  LOG_INFO << "Use MNEE " << string_from_bool(features & KERNEL_FEATURE_MNEE);
  LOG_INFO << "Use Transparent " << string_from_bool(features & KERNEL_FEATURE_TRANSPARENT);
  LOG_INFO << "Use Denoising " << string_from_bool(features & KERNEL_FEATURE_DENOISING);
  LOG_INFO << "Use Path Tracing " << string_from_bool(features & KERNEL_FEATURE_PATH_TRACING);
  LOG_INFO << "Use Hair " << string_from_bool(features & KERNEL_FEATURE_HAIR);
  LOG_INFO << "Use Pointclouds " << string_from_bool(features & KERNEL_FEATURE_POINTCLOUD);
  LOG_INFO << "Use Object Motion " << string_from_bool(features & KERNEL_FEATURE_OBJECT_MOTION);
  LOG_INFO << "Use Baking " << string_from_bool(features & KERNEL_FEATURE_BAKING);
  LOG_INFO << "Use Subsurface " << string_from_bool(features & KERNEL_FEATURE_SUBSURFACE);
  LOG_INFO << "Use Volume " << string_from_bool(features & KERNEL_FEATURE_VOLUME);
  LOG_INFO << "Use Shadow Catcher " << string_from_bool(features & KERNEL_FEATURE_SHADOW_CATCHER);
  LOG_INFO << "Use Portal Node " << string_from_bool(features & KERNEL_FEATURE_NODE_PORTAL);
  LOG_INFO << "Use Light Linking " << string_from_bool(features & KERNEL_FEATURE_LIGHT_LINKING);
  LOG_INFO << "Use Shadow Linking " << string_from_bool(features & KERNEL_FEATURE_SHADOW_LINKING);
  LOG_INFO << "Use Gaussian Splats " << string_from_bool(features & KERNEL_FEATURE_GSPLATS);
}

bool Scene::load_kernels(Progress &progress)
{
  progress.set_status("Loading render kernels (may take a few minutes the first time)");

  const scoped_timer timer;

  const uint64_t kernel_features = dscene.data.kernel_features;
  log_kernel_features(kernel_features);
  if (!device->load_kernels(kernel_features)) {
    string message = device->error_message();
    if (message.empty()) {
      message = "Failed loading render kernel, see console for errors";
    }

    progress.set_error(message);
    progress.set_status(message);
    progress.set_update();
    return false;
  }

  kernels_loaded = true;
  loaded_kernel_features = kernel_features;
  return true;
}

int Scene::get_max_closure_count()
{
  if (shader_manager->use_osl()) {
    /* OSL always needs the maximum as we can't predict the
     * number of closures a shader might generate. */
    return MAX_CLOSURE;
  }

  int max_closures = 0;
  for (int i = 0; i < shaders.size(); i++) {
    Shader *shader = shaders[i];
    if (shader->reference_count()) {
      const int num_closures = shader->graph->get_num_closures();
      max_closures = max(max_closures, num_closures);
    }
  }
  max_closure_global = max(max_closure_global, max_closures);

  if (max_closure_global > MAX_CLOSURE) {
    /* This is usually harmless as more complex shader tend to get many
     * closures discarded due to mixing or low weights. We need to limit
     * to MAX_CLOSURE as this is hardcoded in CPU/mega kernels, and it
     * avoids excessive memory usage for split kernels. */
    LOG_WARNING << "Maximum number of closures exceeded: " << max_closure_global << " > "
                << MAX_CLOSURE;

    max_closure_global = MAX_CLOSURE;
  }

  return max_closure_global;
}

int Scene::get_volume_stack_size() const
{
  int volume_stack_size = 0;

  /* Space for background volume and terminator.
   * Don't do optional here because camera ray initialization expects that there
   * is space for at least those elements (avoiding extra condition to check if
   * there is actual volume or not).
   */
  volume_stack_size += 2;

  /* Quick non-expensive check. Can over-estimate maximum possible nested level,
   * but does not require expensive calculation during pre-processing. */
  bool has_volume_object = false;
  for (const Object *object : objects) {
    if (object->has_nested_dielectric()) {
      volume_stack_size += object->num_nested_dielectric_materials();
      /* Reserve an ordinary volume slot as well: priority-zero material regions
       * on a joined object still follow the native volume stack. */
      volume_stack_size += object->get_geometry()->has_volume;
      continue;
    }
    if (!object->get_geometry()->has_volume) {
      continue;
    }

    if (object->intersects_volume) {
      /* Object intersects another volume, assume it's possible to go deeper in
       * the stack. */
      /* TODO(sergey): This might count nesting twice (A intersects B and B
       * intersects A), but can't think of a computationally cheap algorithm.
       * Dividing my 2 doesn't work because of Venn diagram example with 3
       * circles. */
      ++volume_stack_size;
    }
    else if (!has_volume_object) {
      /* Allocate space for at least one volume object. */
      ++volume_stack_size;
    }

    has_volume_object = true;

    if (volume_stack_size == MAX_VOLUME_STACK_SIZE) {
      break;
    }
  }

  volume_stack_size = min(volume_stack_size, MAX_VOLUME_STACK_SIZE);

  LOG_DEBUG << "Detected required volume stack size " << volume_stack_size;

  return volume_stack_size;
}

bool Scene::has_shadow_catcher()
{
  if (shadow_catcher_modified_) {
    has_shadow_catcher_ = false;
    for (Object *object : objects) {
      /* Shadow catcher flags on lights only controls effect on other objects, it's
       * not catching shadows itself. This is on by default, so ignore to avoid
       * performance impact when there is no actual shadow catcher. */
      if (object->get_is_shadow_catcher() && !object->get_geometry()->is_light()) {
        has_shadow_catcher_ = true;
        break;
      }
    }

    shadow_catcher_modified_ = false;
  }

  return has_shadow_catcher_;
}

void Scene::tag_shadow_catcher_modified()
{
  shadow_catcher_modified_ = true;
}

bool Scene::has_volume()
{
  has_volume_modified_ = false;
  return dscene.data.integrator.use_volumes;
}

bool Scene::has_volume_modified() const
{
  return has_volume_modified_;
}

void Scene::tag_has_volume_modified()
{
  has_volume_modified_ = true;
}

bool Scene::use_light_mis() const
{
  for (const Object *object : objects) {
    if (!object->get_geometry()->is_light()) {
      continue;
    }

    const Light *light = static_cast<const Light *>(object->get_geometry());
    if (light->enabled_for_object(object) && light->get_use_mis() && light->is_traceable()) {
      return true;
    }
  }

  return false;
}

template<class T> T *Scene::create_light_node()
{
  unique_ptr<T> node = make_unique<T>();
  T *node_ptr = node.get();
  node->set_owner(this);
  geometry.push_back(std::move(node));
  light_manager->tag_update(this, LightManager::LIGHT_ADDED);
  return node_ptr;
}

template<> PointLight *Scene::create_node<PointLight>()
{
  return create_light_node<PointLight>();
}

template<> SpotLight *Scene::create_node<SpotLight>()
{
  return create_light_node<SpotLight>();
}

template<> AreaLight *Scene::create_node<AreaLight>()
{
  return create_light_node<AreaLight>();
}

template<> SunLight *Scene::create_node<SunLight>()
{
  return create_light_node<SunLight>();
}

template<> BackgroundLight *Scene::create_node<BackgroundLight>()
{
  return create_light_node<BackgroundLight>();
}

template<> Mesh *Scene::create_node<Mesh>()
{
  unique_ptr<Mesh> node = make_unique<Mesh>();
  Mesh *node_ptr = node.get();
  node->set_owner(this);
  geometry.push_back(std::move(node));
  geometry_manager->tag_update(this, GeometryManager::MESH_ADDED);
  return node_ptr;
}

template<> Hair *Scene::create_node<Hair>()
{
  unique_ptr<Hair> node = make_unique<Hair>();
  Hair *node_ptr = node.get();
  node->set_owner(this);
  geometry.push_back(std::move(node));
  geometry_manager->tag_update(this, GeometryManager::HAIR_ADDED);
  return node_ptr;
}

template<> Volume *Scene::create_node<Volume>()
{
  unique_ptr<Volume> node = make_unique<Volume>();
  Volume *node_ptr = node.get();
  node->set_owner(this);
  geometry.push_back(std::move(node));
  geometry_manager->tag_update(this, GeometryManager::MESH_ADDED);
  return node_ptr;
}

template<> PointCloud *Scene::create_node<PointCloud>()
{
  unique_ptr<PointCloud> node = make_unique<PointCloud>();
  PointCloud *node_ptr = node.get();
  node->set_owner(this);
  geometry.push_back(std::move(node));
  geometry_manager->tag_update(this, GeometryManager::POINT_ADDED);
  return node_ptr;
}

template<> Object *Scene::create_node<Object>()
{
  unique_ptr<Object> node = make_unique<Object>();
  Object *node_ptr = node.get();
  node->set_owner(this);
  objects.push_back(std::move(node));
  object_manager->tag_update(this, ObjectManager::OBJECT_ADDED);
  return node_ptr;
}

template<> ParticleSystem *Scene::create_node<ParticleSystem>()
{
  unique_ptr<ParticleSystem> node = make_unique<ParticleSystem>();
  ParticleSystem *node_ptr = node.get();
  node->set_owner(this);
  particle_systems.push_back(std::move(node));
  particle_system_manager->tag_update(this);
  return node_ptr;
}

template<> Shader *Scene::create_node<Shader>()
{
  unique_ptr<Shader> node = make_unique<Shader>();
  Shader *node_ptr = node.get();
  node->set_owner(this);
  shaders.push_back(std::move(node));
  shader_manager->tag_update(this, ShaderManager::SHADER_ADDED);
  return node_ptr;
}

template<> Pass *Scene::create_node<Pass>()
{
  unique_ptr<Pass> node = make_unique<Pass>();
  Pass *node_ptr = node.get();
  node->set_owner(this);
  passes.push_back(std::move(node));
  film->tag_modified();
  return node_ptr;
}

template<> Camera *Scene::create_node<Camera>()
{
  unique_ptr<Camera> node = make_unique<Camera>();
  Camera *node_ptr = node.get();
  node->set_owner(this);
  cameras.push_back(std::move(node));
  return node_ptr;
}

template<> Integrator *Scene::create_node<Integrator>()
{
  unique_ptr<Integrator> node = make_unique<Integrator>();
  Integrator *node_ptr = node.get();
  node->set_owner(this);
  integrators.push_back(std::move(node));
  return node_ptr;
}

template<> SceneAttributes *Scene::create_node<SceneAttributes>()
{
  unique_ptr<SceneAttributes> node = make_unique<SceneAttributes>();
  SceneAttributes *node_ptr = node.get();
  node->set_owner(this);
  scene_attributes.push_back(std::move(node));
  return node_ptr;
}

template<> Background *Scene::create_node<Background>()
{
  unique_ptr<Background> node = make_unique<Background>();
  Background *node_ptr = node.get();
  node->set_owner(this);
  backgrounds.push_back(std::move(node));
  return node_ptr;
}

template<> Film *Scene::create_node<Film>()
{
  unique_ptr<Film> node = make_unique<Film>();
  Film *node_ptr = node.get();
  node->set_owner(this);
  films.push_back(std::move(node));
  return node_ptr;
}

template<> void Scene::delete_node(Light *node)
{
  assert(node->get_owner() == this);
  geometry.erase_by_swap(node);
  light_manager->tag_update(this, LightManager::LIGHT_REMOVED);
}

template<> void Scene::delete_node(Mesh *node)
{
  assert(node->get_owner() == this);
  geometry.erase_by_swap(node);
  geometry_manager->tag_update(this, GeometryManager::MESH_REMOVED);
}

template<> void Scene::delete_node(Hair *node)
{
  assert(node->get_owner() == this);
  geometry.erase_by_swap(node);
  geometry_manager->tag_update(this, GeometryManager::HAIR_REMOVED);
}

template<> void Scene::delete_node(Volume *node)
{
  assert(node->get_owner() == this);
  geometry.erase_by_swap(node);
  geometry_manager->tag_update(this, GeometryManager::MESH_REMOVED);
}

template<> void Scene::delete_node(PointCloud *node)
{
  assert(node->get_owner() == this);
  geometry.erase_by_swap(node);
  geometry_manager->tag_update(this, GeometryManager::POINT_REMOVED);
}

template<> void Scene::delete_node(Geometry *node)
{
  assert(node->get_owner() == this);

  uint flag;
  if (node->is_hair()) {
    flag = GeometryManager::HAIR_REMOVED;
  }
  else {
    flag = GeometryManager::MESH_REMOVED;
    if (node->has_volume) {
      volume_manager->tag_update({node});
    }
  }

  geometry.erase_by_swap(node);
  geometry_manager->tag_update(this, flag);
}

template<> void Scene::delete_node(Object *node)
{
  assert(node->get_owner() == this);

  uint flag = ObjectManager::OBJECT_REMOVED;
  if (node->get_geometry()->has_volume) {
    volume_manager->tag_update({node}, flag);
  }

  objects.erase_by_swap(node);
  object_manager->tag_update(this, flag);
}

template<> void Scene::delete_node(ParticleSystem *node)
{
  assert(node->get_owner() == this);
  particle_systems.erase_by_swap(node);
  particle_system_manager->tag_update(this);
}

template<> void Scene::delete_node(Shader *node)
{
  assert(node->get_owner() == this);
  /* don't delete unused shaders, not supported */
  node->clear_reference_count();
}

template<> void Scene::delete_node(Procedural *node)
{
  assert(node->get_owner() == this);
  procedurals.erase_by_swap(node);
  procedural_manager->tag_update();
}

template<> void Scene::delete_node(Pass *node)
{
  assert(node->get_owner() == this);
  passes.erase_by_swap(node);
  film->tag_modified();
}

template<typename T> static void assert_same_owner(const set<T *> &nodes, const NodeOwner *owner)
{
#ifdef NDEBUG
  (void)nodes;
  (void)owner;
#else
  for (const T *node : nodes) {
    assert(node->get_owner() == owner);
  }
#endif
}

template<> void Scene::delete_nodes(const set<Geometry *> &nodes, const NodeOwner *owner)
{
  assert_same_owner(nodes, owner);
  volume_manager->tag_update(nodes);
  geometry.erase_in_set(nodes);
  geometry_manager->tag_update(this, GeometryManager::GEOMETRY_REMOVED);
  light_manager->tag_update(this, LightManager::LIGHT_REMOVED);
}

template<> void Scene::delete_nodes(const set<Object *> &nodes, const NodeOwner *owner)
{
  assert_same_owner(nodes, owner);
  volume_manager->tag_update(nodes, ObjectManager::OBJECT_REMOVED);
  objects.erase_in_set(nodes);
  object_manager->tag_update(this, ObjectManager::OBJECT_REMOVED);
}

template<> void Scene::delete_nodes(const set<ParticleSystem *> &nodes, const NodeOwner *owner)
{
  assert_same_owner(nodes, owner);
  particle_systems.erase_in_set(nodes);
  particle_system_manager->tag_update(this);
}

template<> void Scene::delete_nodes(const set<Shader *> &nodes, const NodeOwner * /*owner*/)
{
  /* don't delete unused shaders, not supported */
  for (Shader *shader : nodes) {
    shader->clear_reference_count();
  }
}

template<> void Scene::delete_nodes(const set<Procedural *> &nodes, const NodeOwner *owner)
{
  assert_same_owner(nodes, owner);
  procedurals.erase_in_set(nodes);
  procedural_manager->tag_update();
}

template<> void Scene::delete_nodes(const set<Pass *> &nodes, const NodeOwner *owner)
{
  assert_same_owner(nodes, owner);
  passes.erase_in_set(nodes);
  film->tag_modified();
}

/* Template instantiations so we don't have to inline functions. */
template PointLight *Scene::create_light_node<PointLight>();
template SpotLight *Scene::create_light_node<SpotLight>();
template AreaLight *Scene::create_light_node<AreaLight>();
template SunLight *Scene::create_light_node<SunLight>();
template BackgroundLight *Scene::create_light_node<BackgroundLight>();

CCL_NAMESPACE_END
