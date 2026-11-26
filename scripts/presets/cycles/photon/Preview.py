# SPDX-FileCopyrightText: 2026 Blender Authors
# SPDX-License-Identifier: GPL-2.0-or-later
# Photon parameters from Untitled.blend; source SHA256: b1bf92c013e9b1f318f19078cf33178614f300893aa52974e05304487031e0c0
import bpy

cycles = bpy.context.scene.cycles
cycles.photon_caustics_count = 65536
cycles.photon_caustics_update_interval = 1
cycles.use_photon_query_index = True
cycles.photon_caustics_separate_sampling = True
cycles.photon_caustics_sampling_budget = 'INDEPENDENT'
cycles.photon_caustics_samples = 256
cycles.photon_caustics_footprint = 'DIFFERENTIAL'
cycles.photon_caustics_radius = 0.019999999552965164
cycles.photon_caustics_min_radius = 0.004999999888241291
cycles.photon_caustics_gather_max = 0
cycles.photon_caustics_roughness_spread = 1.0
cycles.photon_caustics_scatter = 0.0
cycles.photon_caustics_differential_scatter = 0.0
cycles.photon_caustics_noise_strength = 0.0
cycles.photon_caustics_ellipse_noise_strength = 0.0
cycles.use_photon_adaptive_radius = False
cycles.photon_adaptive_neighbors = 16
cycles.photon_adaptive_max_scale = 2.0
cycles.photon_reconstruction_method = 'ELLIPSE_COVERAGE'
cycles.photon_reconstruction_pilot_scale = 4.0
cycles.use_photon_cell_statistics = True
cycles.use_photon_neural_bandwidth = False
cycles.photon_reconstruction_filter = 'LEGACY'
cycles.photon_reconstruction_normal_angle = 0.5235987901687622
cycles.use_photon_beam_continuity = True
cycles.use_photon_gradient_guard = True
cycles.photon_gradient_strength = 1.0
cycles.use_photon_guiding = True
cycles.use_photon_area_guiding = False
