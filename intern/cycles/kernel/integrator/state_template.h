/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

/************************************ Path State *****************************/

KERNEL_STRUCT_BEGIN(path)
/* Index of a pixel within the device render buffer where this path will write its result.
 * To get an actual offset within the buffer the value needs to be multiplied by the
 * `kernel_data.film.pass_stride`.
 *
 * The multiplication is delayed for later, so that state can use 32bit integer. */
KERNEL_STRUCT_MEMBER(path, uint32_t, render_pixel_index, KERNEL_FEATURE_PATH_TRACING)
/* Current sample number. */
KERNEL_STRUCT_MEMBER(path, uint32_t, sample, KERNEL_FEATURE_PATH_TRACING)
/* Active deep hard-surface sample for primary visibility contribution accumulation. */
KERNEL_STRUCT_MEMBER(path, uint32_t, deep_surface_sample_idx, KERNEL_FEATURE_PATH_TRACING)
KERNEL_STRUCT_MEMBER(path, float, deep_volume_z, KERNEL_FEATURE_PATH_TRACING)
/* Current ray bounce depth. */
KERNEL_STRUCT_MEMBER(path, uint16_t, bounce, KERNEL_FEATURE_PATH_TRACING)
/* Current transparent ray bounce depth. */
KERNEL_STRUCT_MEMBER(path, uint16_t, transparent_bounce, KERNEL_FEATURE_PATH_TRACING)
/* Current diffuse ray bounce depth. */
KERNEL_STRUCT_MEMBER(path, uint16_t, diffuse_bounce, KERNEL_FEATURE_PATH_TRACING)
/* Current glossy ray bounce depth. */
KERNEL_STRUCT_MEMBER(path, uint16_t, glossy_bounce, KERNEL_FEATURE_PATH_TRACING)
/* Current transmission ray bounce depth. */
KERNEL_STRUCT_MEMBER(path, uint16_t, transmission_bounce, KERNEL_FEATURE_PATH_TRACING)
/* Current volume ray bounce depth. */
KERNEL_STRUCT_MEMBER(path, uint16_t, volume_bounce, KERNEL_FEATURE_PATH_TRACING)
/* Current volume bounds ray bounce depth. */
KERNEL_STRUCT_MEMBER(path, uint16_t, volume_bounds_bounce, KERNEL_FEATURE_PATH_TRACING)
/* Current portal ray bounce depth. */
KERNEL_STRUCT_MEMBER(path, uint16_t, portal_bounce, KERNEL_FEATURE_NODE_PORTAL)
/* DeviceKernel bit indicating queued kernels. */
KERNEL_STRUCT_MEMBER(path, uint16_t, queued_kernel, KERNEL_FEATURE_PATH_TRACING)
/* Random number generator per-pixel info. */
KERNEL_STRUCT_MEMBER(path, uint32_t, rng_pixel, KERNEL_FEATURE_PATH_TRACING)
/* Random number dimension offset. */
KERNEL_STRUCT_MEMBER(path, uint16_t, rng_offset, KERNEL_FEATURE_PATH_TRACING)
/* enum PathRayVisibilityFlag, limited to bits from PATH_RAY_VISIBILITY_ALL */
KERNEL_STRUCT_MEMBER(path, uint8_t, visibility, KERNEL_FEATURE_PATH_TRACING)
/* enum PathRayFlag */
KERNEL_STRUCT_MEMBER(path, uint32_t, flag, KERNEL_FEATURE_PATH_TRACING)
/* enum PathRayMNEE */
KERNEL_STRUCT_MEMBER(path, uint8_t, mnee, KERNEL_FEATURE_PATH_TRACING)
/* Index of shadow state path used for storing MNEE state. */
KERNEL_STRUCT_MEMBER(path, int, mnee_shadow_state, KERNEL_FEATURE_MNEE)
/* Independent photon estimator: trace camera prefixes without native radiance/NEE. */
KERNEL_STRUCT_MEMBER(path, uint8_t, photon_only, KERNEL_FEATURE_PATH_TRACING)
/* Actual supported receiver -> delta caster suffix, separate from MNEE's heuristic flags. */
KERNEL_STRUCT_MEMBER(path, uint32_t, photon_caustics, KERNEL_FEATURE_MNEE)
/* Global pixel ordering for photon blue noise, independent of the scene sampler. */
KERNEL_STRUCT_MEMBER(path, uint32_t, photon_noise_pixel, KERNEL_FEATURE_MNEE)
/* Majorant volume optical depth. */
KERNEL_STRUCT_MEMBER(path, float, optical_depth, KERNEL_FEATURE_PATH_TRACING)
/* Multiple importance sampling
 * The PDF of BSDF sampling at the last scatter point, which is at ray distance
 * zero and distance. Note that transparency and volume attenuation increase
 * the ray tmin but keep P unmodified so that this works. */
KERNEL_STRUCT_MEMBER(path, float, mis_ray_pdf, KERNEL_FEATURE_PATH_TRACING)
/* Object at last scatter point for light linking. */
KERNEL_STRUCT_MEMBER(path, int, mis_ray_object, KERNEL_FEATURE_LIGHT_LINKING)
/* Normal at last scatter point for light tree. */
KERNEL_STRUCT_MEMBER(path, packed_float3, mis_origin_n, KERNEL_FEATURE_PATH_TRACING)
/* Filter glossy. */
KERNEL_STRUCT_MEMBER(path, float, min_ray_pdf, KERNEL_FEATURE_PATH_TRACING)
/* Continuation probability for path termination. */
KERNEL_STRUCT_MEMBER(path,
                     FloatCompressedOnGPU,
                     continuation_probability,
                     KERNEL_FEATURE_PATH_TRACING)
/* Throughput. */
KERNEL_STRUCT_MEMBER(path, PackedSpectrum, throughput, KERNEL_FEATURE_PATH_TRACING)
/* Factor to multiple with throughput to get remove any guiding PDFS.
 * Such throughput without guiding PDFS is used for Russian roulette termination. */
KERNEL_STRUCT_MEMBER(path, float, unguided_throughput, KERNEL_FEATURE_PATH_GUIDING)
/* Ratio of throughput to distinguish diffuse / glossy / transmission render passes. */
KERNEL_STRUCT_MEMBER(path, SpectrumCompressedOnGPU, pass_diffuse_weight, KERNEL_FEATURE_LIGHT_PASSES)
KERNEL_STRUCT_MEMBER(path, SpectrumCompressedOnGPU, pass_glossy_weight, KERNEL_FEATURE_LIGHT_PASSES)
/* LPE uses the AOV feature allocation gate; no new global feature bit is required. */
KERNEL_STRUCT_MEMBER(path, uint64_t, lpe_states, KERNEL_FEATURE_NODE_AOV)
KERNEL_STRUCT_MEMBER(path, PackedSpectrum, lpe_weight, KERNEL_FEATURE_NODE_AOV)
KERNEL_STRUCT_MEMBER(path, uint8_t, lpe_skip_surface, KERNEL_FEATURE_NODE_AOV)
/* Denoising. */
KERNEL_STRUCT_MEMBER(path,
                     SpectrumCompressedOnGPU,
                     denoising_feature_throughput,
                     KERNEL_FEATURE_DENOISING)
/* Shader sorting. */
/* TODO: compress as uint16? or leave out entirely and recompute key in sorting code? */
KERNEL_STRUCT_MEMBER(path, uint32_t, shader_sort_key, KERNEL_FEATURE_PATH_TRACING)
KERNEL_STRUCT_END(path)

/************************************** Ray ***********************************/

KERNEL_STRUCT_BEGIN_PACKED(ray, KERNEL_FEATURE_PATH_TRACING)
KERNEL_STRUCT_MEMBER_PACKED(ray, packed_float3, P, KERNEL_FEATURE_PATH_TRACING)
KERNEL_STRUCT_MEMBER_PACKED(ray, packed_float3, D, KERNEL_FEATURE_PATH_TRACING)
KERNEL_STRUCT_MEMBER_PACKED(ray, float, tmin, KERNEL_FEATURE_PATH_TRACING)
KERNEL_STRUCT_MEMBER_PACKED(ray, float, tmax, KERNEL_FEATURE_PATH_TRACING)
KERNEL_STRUCT_MEMBER_PACKED(ray, float, dP, KERNEL_FEATURE_PATH_TRACING)
KERNEL_STRUCT_MEMBER_PACKED(ray, FloatCompressedOnGPU, dD, KERNEL_FEATURE_PATH_TRACING)
KERNEL_STRUCT_MEMBER_PACKED(ray, FloatCompressedOnGPU, time, KERNEL_FEATURE_PATH_TRACING)
KERNEL_STRUCT_MEMBER(ray, float, previous_dt, KERNEL_FEATURE_LIGHT_TREE)
KERNEL_STRUCT_END(ray)

/*************************** Intersection result ******************************/

/* Result from scene intersection. */
KERNEL_STRUCT_BEGIN_PACKED(isect, KERNEL_FEATURE_PATH_TRACING)
KERNEL_STRUCT_MEMBER_PACKED(isect, float, t, KERNEL_FEATURE_PATH_TRACING)
KERNEL_STRUCT_MEMBER_PACKED(isect, float, u, KERNEL_FEATURE_PATH_TRACING)
KERNEL_STRUCT_MEMBER_PACKED(isect, float, v, KERNEL_FEATURE_PATH_TRACING)
KERNEL_STRUCT_MEMBER_PACKED(isect, int, prim, KERNEL_FEATURE_PATH_TRACING)
KERNEL_STRUCT_MEMBER_PACKED(isect, int, object, KERNEL_FEATURE_PATH_TRACING)
KERNEL_STRUCT_MEMBER_PACKED(isect, int, type, KERNEL_FEATURE_PATH_TRACING)
KERNEL_STRUCT_END(isect)

/*************** Subsurface closure state for subsurface kernel ***************/

KERNEL_STRUCT_BEGIN_PACKED(subsurface, KERNEL_FEATURE_SUBSURFACE)
KERNEL_STRUCT_MEMBER_PACKED(subsurface, PackedSpectrum, radius, KERNEL_FEATURE_SUBSURFACE)
KERNEL_STRUCT_MEMBER_PACKED(subsurface, packed_float3, N, KERNEL_FEATURE_SUBSURFACE)
KERNEL_STRUCT_MEMBER_PACKED(subsurface, SpectrumCompressedOnGPU, albedo, KERNEL_FEATURE_SUBSURFACE)
KERNEL_STRUCT_MEMBER_PACKED(subsurface,
                            FloatCompressedOnGPU,
                            anisotropy,
                            KERNEL_FEATURE_SUBSURFACE)
KERNEL_STRUCT_END(subsurface)

/********************************** Volume Stack ******************************/

KERNEL_STRUCT_BEGIN(volume_stack)
KERNEL_STRUCT_ARRAY_MEMBER(volume_stack, int, object, KERNEL_FEATURE_VOLUME)
KERNEL_STRUCT_ARRAY_MEMBER(volume_stack, int, shader, KERNEL_FEATURE_VOLUME)
KERNEL_STRUCT_END_ARRAY(volume_stack,
                        KERNEL_STRUCT_VOLUME_STACK_SIZE,
                        KERNEL_STRUCT_VOLUME_STACK_SIZE)

/* Allocated only when a scene opts into explicit nested dielectrics. The shared
 * capacity includes all selected objects and volume entries plus a sentinel. */
KERNEL_STRUCT_BEGIN(medium_stack)
KERNEL_STRUCT_ARRAY_MEMBER(medium_stack, int, object, KERNEL_FEATURE_NESTED_DIELECTRICS)
KERNEL_STRUCT_ARRAY_MEMBER(medium_stack, int, shader, KERNEL_FEATURE_NESTED_DIELECTRICS)
KERNEL_STRUCT_ARRAY_MEMBER(medium_stack, int, count, KERNEL_FEATURE_NESTED_DIELECTRICS)
KERNEL_STRUCT_ARRAY_MEMBER(medium_stack, float, ior, KERNEL_FEATURE_NESTED_DIELECTRICS)
KERNEL_STRUCT_ARRAY_MEMBER(medium_stack, float, inv_abbe, KERNEL_FEATURE_NESTED_DIELECTRICS)
KERNEL_STRUCT_ARRAY_MEMBER(medium_stack, PackedSpectrum, transmission_tint, KERNEL_FEATURE_NESTED_DIELECTRICS)
KERNEL_STRUCT_END_ARRAY(medium_stack,
                        KERNEL_STRUCT_VOLUME_STACK_SIZE,
                        KERNEL_STRUCT_VOLUME_STACK_SIZE)

/************************************ Path Guiding *****************************/
KERNEL_STRUCT_BEGIN(gpu_guiding)
KERNEL_STRUCT_MEMBER(gpu_guiding, int, bin, KERNEL_FEATURE_GPU_PATH_GUIDING)
KERNEL_STRUCT_MEMBER(gpu_guiding, PackedSpectrum, inverse_weight, KERNEL_FEATURE_GPU_PATH_GUIDING)
KERNEL_STRUCT_END(gpu_guiding)

KERNEL_STRUCT_BEGIN(guiding)
#if defined(__PATH_GUIDING__)
/* Current path segment of the random walk/path. */
KERNEL_STRUCT_MEMBER(guiding,
                     openpgl::cpp::PathSegment *,
                     path_segment,
                     KERNEL_FEATURE_PATH_GUIDING)
#else
/* Current path segment of the random walk/path. */
KERNEL_STRUCT_MEMBER(guiding, uint64_t, path_segment, KERNEL_FEATURE_PATH_GUIDING)
#endif
/* If surface guiding is enabled */
KERNEL_STRUCT_MEMBER(guiding, bool, use_surface_guiding, KERNEL_FEATURE_PATH_GUIDING)
/* Random number used for additional guiding decisions (e.g., cache query, selection to use guiding
 * or BSDF sampling) */
KERNEL_STRUCT_MEMBER(guiding,
                     FloatCompressedOnGPU,
                     sample_surface_guiding_rand,
                     KERNEL_FEATURE_PATH_GUIDING)
/* The probability to use surface guiding (i.e., diffuse sampling prob * guiding prob). */
KERNEL_STRUCT_MEMBER(guiding,
                     FloatCompressedOnGPU,
                     surface_guiding_sampling_prob,
                     KERNEL_FEATURE_PATH_GUIDING)
/* Probability of sampling a BSSRDF closure instead of a BSDF closure. */
KERNEL_STRUCT_MEMBER(guiding,
                     FloatCompressedOnGPU,
                     bssrdf_sampling_prob,
                     KERNEL_FEATURE_PATH_GUIDING)
/* If volume guiding is enabled */
KERNEL_STRUCT_MEMBER(guiding, bool, use_volume_guiding, KERNEL_FEATURE_PATH_GUIDING)
/* Random number used for additional guiding decisions (e.g., cache query, selection to use guiding
 * or BSDF sampling) */
KERNEL_STRUCT_MEMBER(guiding,
                     FloatCompressedOnGPU,
                     sample_volume_guiding_rand,
                     KERNEL_FEATURE_PATH_GUIDING)
/* The probability to use surface guiding (i.e., diffuse sampling prob * guiding prob). */
KERNEL_STRUCT_MEMBER(guiding,
                     FloatCompressedOnGPU,
                     volume_guiding_sampling_prob,
                     KERNEL_FEATURE_PATH_GUIDING)
KERNEL_STRUCT_END(guiding)

/******************************* Shadow linking *******************************/

KERNEL_STRUCT_BEGIN(shadow_link)
/* Number of dedicated light hits along the shadow ray. */
KERNEL_STRUCT_MEMBER(shadow_link, uint16_t, dedicated_light_weight, KERNEL_FEATURE_SHADOW_LINKING)
/* Copy of primitive and object from the last main path intersection. */
KERNEL_STRUCT_MEMBER(shadow_link, int, last_isect_prim, KERNEL_FEATURE_SHADOW_LINKING)
KERNEL_STRUCT_MEMBER(shadow_link, int, last_isect_object, KERNEL_FEATURE_SHADOW_LINKING)
KERNEL_STRUCT_END(shadow_link)
