# Module reference

Every package under `engine/cosmic_engine/`. Each section lists the
key public symbols and one-line responsibilities. Detailed behaviour
lives in docstrings; this page is the index.

## core

The bedrock dataclasses every other layer builds on.

| Symbol | Role |
|---|---|
| `Vector3` | 3-component float vector with arithmetic + dot/cross. |
| `Vector3Batch` | Batched `(N, 3)` numpy wrapper with the same ops. |
| `UniverseObject` | The universal record type: id, name, type, position, velocity, mass, radius, luminosity, spectral class, redshift, **truth_level**, **source**, metadata. |
| `UniverseRegistry` | O(1) id → object dict. |
| `CosmicObjectType` | Enum: STAR / PLANET / GALAXY / BLACK_HOLE / NEUTRON_STAR / etc. |
| `TruthLevel` | Provenance enum (see [`PROVENANCE.md`](PROVENANCE.md)). |
| `SimulationClock` | Julian-date clock with `tick(seconds)`. |
| `units` | `SPEED_OF_LIGHT_M_S`, `AU_IN_METERS`, `LIGHTYEAR_IN_METERS`, `PARSEC_IN_METERS` + converters. |
| `coordinates` | RA/Dec ↔ Cartesian. |

## data

Catalog ingestion. Every loader returns `list[UniverseObject]` with
correct `truth_level` / `source`.

| Symbol | Source |
|---|---|
| `load_gaia_like_catalog(path)` | `OBSERVED` Gaia-DR3-style CSV. |
| `load_sdss_like_catalog(path)` | `CATALOG_IMPORTED` SDSS galaxies. |
| `load_desi_catalog(path)` | `CATALOG_IMPORTED` DESI redshift survey. |
| `load_jpl_ephemeris_placeholder(jd)` | `EPHEMERIS_REAL` solar system. |
| `load_star_catalog_into_registry(...)` | Generic CSV → registry. |
| `generate_synthetic_galaxy_catalog(n, ...)` | `PROCEDURAL_APPROXIMATION`. |

## physics

Deterministic gravity + GR.

- `nbody.NBodySimulator` — leapfrog integrator over `(positions, velocities, masses)`.
- `barnes_hut.BarnesHutSimulator` — O(N log N) tree-coded integrator.
- `gr.SchwarzschildModel` — analytical Schwarzschild metric helpers.
- `solar_system.create_solar_system_objects(jd)` — JPL placeholder.
- `GRAVITATIONAL_CONSTANT` constant.

## cosmos

ΛCDM + redshift utilities.

- `LambdaCDM` — Hubble flow, comoving distance, redshift ↔ distance.
- `redshift_to_velocity`, `velocity_to_redshift`.

## perception

Per-observer photon-field warp.

- `ObserverState(position, velocity, forward, up, warp_factor)`.
- `transform_photon_field(samples, observer, ai_model=None)` —
  aberration + beaming + Doppler tint, optionally swapped for an
  `AIWarpModel`.
- `vectorized_transform.transform_photon_field_batch` — numpy batch.
- `vectorized_ai_transform` — same, with an ONNX `AIWarpModel`.

## rendering

Photon field → 2D image + Gaussian splatting.

- `SimpleCamera` — pinhole pose + FOV.
- `PhotonSample` / `PhotonFieldBatch` / `build_star_photon_field`.
- `GalaxyFieldBatch` / `build_galaxy_field_batch`.
- `render_photon_field_to_ppm` / `render_galaxy_batch_to_ppm`.
- `density_field.galaxy_batch_to_density_grid`.

The `apps/ai_viewer/neural_field/` tree adds CPU + WebGPU
Gaussian-splat renderers.

## ai

Neural augmentation + ONNX runtime + training.

- `AIWarpModel` (abstract) — predict_direction / brightness / color.
- `ONNXPhotonWarp`, `ONNXBatchPhotonWarp` — runtime inference.
- `ONNXDensityModel` — density-field reconstruction.
- `SpacetimeFieldModel` (abstract) — `query_acceleration(position)`.
- `ONNXSpacetimeField` — runtime inference, falls back analytically.
- `cosmic_engine.ai.training` — opt-in PyTorch training pipeline
  (only this submodule imports torch).

## streaming

Out-of-core data + LOD.

- `SpatialIndex(cell_size_m).query_radius(observer, radius)`.
- `TileStore` — partition + load tiles from disk.
- `select_lod_objects` — capped, spatially-aware downsampling.

## multiscale

Scale-zone-aware representation switching.

- `ScaleZone(name, min_scale_m, max_scale_m, representation_type)`.
- `ScaleManager(zones)` + `DEFAULT_ZONES`
  (microscale → solar_system → interstellar → intergalactic).
- `compute_transition_alpha`, `blend_representations` —
  smooth zone-boundary blends.
- Hooks into `CosmicRuntime.enable_multiscale` /
  `get_multiscale_scene`.

## observer

Multi-observer reality.

- `Observer(id, position_m, velocity_m_s, forward, up, warp_factor,
  spacetime_model, ai_warp_model, config, proper_time_tau,
  coordinate_time_t)`. `advance_time(dt, masses_for_potential=None)`
  steps SR + optional weak-field GR.
- `ObserverManager` — id-keyed registry.
- `RealityView` — the per-observer rendered output. Top-level fields:
  - `observer_id`, `scene_state`, `representation_type`, `frame_data`
  - `metadata`
  - `proper_time_tau`, `coordinate_time_t`, `visible_event_count`
  - `active_rule_ids`, `reality_metadata`
  - `provenance_summary`, `audit_warnings`
  - `feedback_summary`, `adaptive_suggestions`

## time

Subjective time + light-cone causality.

- `gamma_from_beta(beta)` — clamped Lorentz factor.
- `advance_proper_time(tau, dt, beta, gravitational_potential=None)`.
- `gravitational_potential_weak(position, masses)`.
- `Event(id, position_m, time_t, payload, source)`.
- `EventStore.add_event / list_events / query_time_window`.
- `is_event_visible(event, observer_pos, observer_t)`.
- `compute_retarded_time(observer_pos, observer_t, source_pos)`.

## reality

Emergent rule layer.

- Enums: `RealityRuleDomain`, `RealityRuleKind`.
- `RealityRule` (kw-only dataclass) + 4 built-in subclasses:
  `WarpAmplificationRule`, `RedshiftSymbolicColorRule`,
  `CausalityRelaxationRule`, `NeuralRealityRule`.
- `RuleContext` (cloneable; deep copy semantics).
- `RealityRuleEngine.add_rule / remove_rule / list_rules / evaluate`.
- Presets: `create_scientific_reality_preset`,
  `create_hyperwarp_reality_preset`,
  `create_blackhole_perception_preset`.

## provenance

Truth-integrity audit.

- `ProvenanceRecord(entity_id, source, truth_level, transformations,
  timestamp, observer_id)`.
- `TruthTracker.register_entity / add_transformation /
  add_transformation_to_all / get_record / list_records`.
- `audit_reality_view(view) → dict` — observer_id, truth_distribution,
  transformations, active_rules, warnings.
- `detect_truth_mixing(view) → list[str]`.

## adaptive

Self-improving feedback (opt-in).

- `FeedbackRecord` — discrepancy event with deviation + source.
- `compute_acceleration_error / direction_error / brightness_error`.
- `AdaptivePolicy(error_threshold, window_size, sustained_fraction,
  max_updates_per_run)`.
- `AdaptiveEngine.record_feedback / evaluate / summary / reset_run`.
  Suggestions never auto-applied.

## synthesis

Reality synthesis from declarative specs.

- `UniverseSpec(id, seed, scale_limits, initial_conditions,
  physics_model, spacetime_model, rule_ids, constraints)`.
- `Constraint` + `MaxMassConstraint` / `MaxVelocityConstraint` /
  `apply_constraints`.
- `UniverseGenerator(spec).generate_initial_state /
  generate_runtime`. `create_runtime_from_spec(spec)` shorthand.
- Presets: `create_standard_physics_universe`,
  `create_hyperwarp_universe`, `create_symbolic_universe`.

## runtime

The orchestrator everything else plugs into.

- `RuntimeConfig` — physics / perception / AI flags + active radius.
- `CosmicRuntime` — registry, clock, observers, multiscale,
  streaming, reality rules, adaptive, provenance.
  Methods: `step`, `render_for_observer`, `step_all_observers`,
  `enable_multiscale`, `enable_streaming`, `build_spatial_index`,
  `get_multiscale_scene`, `get_visible_events`.
- `RuntimeServer` — TCP server broadcasting `scene_state` /
  `reality_view` JSON messages.
- `SceneState` — the shared-truth half of every render.

## distributed

Lambda Cloud / Ray helpers.

- `DistributedConfig.from_env()` — `COSMIC_*` env-driven config.
- `init_ray / shutdown_ray / is_ray_available`.
- 5 Ray remote tasks (with non-Ray fallbacks):
  `process_catalog_chunk`, `run_physics_step`, `run_ai_warp_batch`,
  `run_density_reconstruction`, `render_gaussian_frame`.
- `health_report(config)` — CUDA / Ray / dir probes.
- `entrypoints` — `run_head_node`, `run_worker_node`,
  `run_runtime_service`, `run_viewer_service`,
  `run_training_service`, plus a `python -m
  cosmic_engine.distributed.entrypoints <role>` CLI.

## apps/ai_viewer

The standalone client. Reads `scene_state` / `reality_view`
messages from `RuntimeServer`, post-processes frames (gamma,
brightness, contrast, neural ONNX postprocess), saves PPMs, and
optionally pops a Tk window.

- `AIViewerConfig` — every knob the viewer exposes (dozens of
  fields covering server, rendering, GR, multiscale, observer
  filtering, audit verbosity, adaptive feedback display).
- `AIViewer` — main loop + per-message dispatch.
- `RuntimeClient` — TCP client.
- `apps/ai_viewer/neural_field/` — Gaussian splatter (CPU +
  WebGPU), neural-warp viewer, GR / lensing experiments.
