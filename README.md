# CosmicEngine

CosmicEngine is a data-driven, physics-grounded, AI-assisted cosmic perception engine.

## Future modules

- `engine` — deterministic universe state
- `renderer` — photon-field and warp viewer
- `ai` — neural renderer experiments
- `data_pipeline` — catalog ingestion

## Current status

- Phase 1 core object model implemented (`TruthLevel`, `CosmicObjectType`, `Vector3`, `UniverseObject`, `UniverseRegistry`).
- Time system (`SimulationClock`, Julian-Date helpers), units, and RA/Dec ↔ Cartesian coordinates.
- No simulation, rendering, or AI yet.

## Phase 2: Data ingestion

- Basic CSV star catalog loader implemented (`cosmic_engine.data`).
- Supports RA/Dec → Cartesian conversion and distance in light-years.
- Registry integration available via `load_star_catalog_into_registry`.
- Sample catalog at `data/sample_stars.csv`; demo at `examples/load_star_catalog_demo.py`.

## Phase 3: Photon field projection

- CosmicEngine can now convert catalog stars into photon samples
  (`SimpleCamera`, `PhotonSample`, `build_star_photon_field`).
- Basic plain-PPM starfield export implemented
  (`render_photon_field_to_ppm`); demo at `examples/render_starfield_demo.py`
  (writes `examples/output_starfield.ppm`).
- Still no AI, GPU, relativity, or advanced rendering.

## Phase 4: Perception transform

- Deterministic, relativistic-inspired perception layer implemented
  (`cosmic_engine.perception`): `ObserverState` carries pose, velocity,
  and a `warp_factor` knob; `transform_photon_field` applies
  aberration-like direction compression, beaming-like brightness
  scaling, and a simple Doppler-flavored blue/red color shift.
- `warp_factor >= 1.0` extends perception beyond the physical regime
  for visualization; superluminal velocities are rejected.
- Demo at `examples/perception_warp_demo.py` writes
  `output_warp_{1,5,50}.ppm`.
- Still deterministic and AI-free.

## Phase 5: AI-assisted perception

- AI warp layer introduced (`cosmic_engine.ai`): `AIWarpModel` base
  class defines the pluggable interface (`predict_direction`,
  `predict_brightness`, `predict_color`, `confidence`); future
  PyTorch / ONNX / JAX models will implement it.
- Currently ships with a deterministic placeholder
  (`SimpleNeuralWarp`) that stands in for a real model so the rest of
  the pipeline is exercisable today.
- `transform_photon_field(samples, observer, ai_model=None)` routes to
  the deterministic path when `ai_model` is `None`. The deterministic
  fallback is always available and remains the default.
- Demo at `examples/ai_warp_demo.py` writes
  `output_ai_{off,on,extreme}.ppm`.

## Phase 6: Real AI integration

- ONNX Runtime (CPU-only) is now a dependency. Real AI models can be
  loaded externally via `ONNXModelWrapper` (file-level) and consumed
  through `ONNXWarpModel` (AIWarpModel interface).
- The pipeline supports real inference end-to-end — the per-sample AI
  path in `transform_photon_sample` is wrapped in try/except so any
  AI failure falls back to the deterministic transform for that field;
  the pipeline never crashes on a bad model.
- A tiny mock model is bundled at `data/mock_warp_model.onnx` (9→7
  linear graph) so the demo and tests run out of the box. Rebuild it
  with `python scripts/build_mock_onnx_model.py` (requires the `onnx`
  authoring library, not needed at runtime).
- Demo at `examples/onnx_warp_demo.py` writes
  `output_onnx_{off,simple,real}.ppm`.

## Phase 7: Vectorized pipeline

- NumPy batch photon-field path added (`PhotonFieldBatch`,
  `build_star_photon_field_batch`, `transform_photon_field_batch`,
  `render_photon_batch_to_ppm`), plus `cosmic_engine.core.vector_batch`
  helpers for `(N, 3)` array conversion, normalization, distances, and
  clamping.
- Prepares the engine for large catalogs and future GPU acceleration;
  on a 10k-star synthetic catalog the deterministic transform runs
  ~30× faster than the scalar path.
- The scalar `PhotonSample` pipeline is untouched and remains the
  reference implementation; AI / ONNX support remains optional and
  lives only on the scalar path for now.
- Demo at `examples/vectorized_warp_demo.py` writes
  `output_vectorized_warp.ppm`.

## Phase 8: Galaxy point clouds and density fields

- CosmicEngine now supports DESI-like galaxy point clouds.
  `cosmic_engine.cosmos.galaxy` adds `GalaxyProperties` +
  `create_galaxy_object` (galaxies carry morphology, redshift,
  magnitude, color index, stellar/halo mass in metadata).
- CSV ingestion via `cosmic_engine.data.galaxy_catalog.load_galaxy_catalog`
  (expects `id,name,x_m,y_m,z_m,redshift_z,apparent_magnitude,morphology`);
  `generate_synthetic_galaxy_catalog(count, radius_m, seed)` is a
  deterministic filamentary placeholder for future DESI ingestion —
  **not scientifically accurate**.
- `GalaxyFieldBatch` + `build_galaxy_field_batch` project galaxies
  into a vectorized rendering frame with redshift-derived colors.
  `render_galaxy_batch_to_ppm` dots them into a PPM.
- `galaxy_batch_to_density_grid` bins positions into a
  `(G, G, G)` cube normalized to `[0, 1]` — a scaffold for future AI
  super-resolution / reconstruction.
- Real DESI ingestion is **not** implemented yet; no real data is
  downloaded, no cosmological distance calculation is performed.
- Sample data at `data/sample_galaxies.csv` (7 Local-Group / nearby
  galaxies); demo at `examples/synthetic_galaxy_cloud_demo.py` writes
  `output_synthetic_galaxies.ppm`.

## Phase 9: AI density reconstruction

- Density grids can now be enhanced by AI models. The
  `cosmic_engine.ai.density_*` modules add a `DensityFieldModel` base
  class, a deterministic NumPy-only `SimpleDensityEnhancer`
  (2× nearest upsample + 3³ box blur + normalize), and an
  `ONNXDensityModel` that runs a real ONNX session.
- `enhance_density_field(grid, model)` is the high-level entry point:
  `model=None` uses the simple enhancer; if a real model raises, the
  pipeline silently falls back to the simple enhancer so callers
  never crash.
- A bundled mock ONNX upscaler at `data/density_upscaler.onnx`
  (Resize op, 64³ → 128³, 272 bytes) makes the demo and tests run
  out of the box. Rebuild with
  `python scripts/build_density_onnx_model.py`.
- This is the first step toward neural universe reconstruction
  (no PyTorch / TensorFlow / GPU; NumPy + ONNX Runtime only).
- Demo at `examples/density_ai_demo.py` writes
  `output_density_{original,simple,ai}.ppm`.

## Phase 10: Cosmology ingestion

- Basic redshift → distance mapping implemented in
  `cosmic_engine.physics.cosmology` via the linear Hubble law
  (`H0 = 70 km/s/Mpc`): `redshift_to_velocity`,
  `redshift_to_distance_m`, `redshift_to_distance_lightyears`.
  Documented as **valid only at low z (≲ 0.3)** and explicitly a
  placeholder for a future ΛCDM integrator.
- DESI-like catalog ingestion at
  `cosmic_engine.data.desi_like_catalog.load_desi_like_catalog` /
  `load_desi_into_registry` accepts `id, ra_deg, dec_deg,
  redshift_z, magnitude`, runs each row through the Hubble-law
  distance and the existing RA/Dec → Cartesian conversion, and emits
  `UniverseObject` galaxies tagged `truth_level=CATALOG_IMPORTED`,
  `source="desi_like_csv"`. Invalid / negative-z rows are skipped.
- Real DESI APIs / FITS / network access are **not** wired up; full
  ΛCDM cosmology and Astropy are still out of scope.
- Sample data at `data/sample_desi_like.csv`; demo at
  `examples/desi_like_demo.py` writes `output_desi_like.ppm`.

## Phase 11: ΛCDM cosmology

- Flat-ΛCDM model implemented in
  `cosmic_engine.physics.cosmology_lcdm` (Ω_m = 0.3, Ω_Λ = 0.7,
  H₀ = 70 km/s/Mpc): dimensionless expansion rate `E(z)`, in-house
  Simpson integrator (`integrate_simpson`), comoving distance
  `D_C(z) = (c/H₀)·∫₀ᶻ dz'/E(z')`, plus `luminosity_distance_m`
  (`D_L = (1+z)·D_C`), `angular_diameter_distance_m`
  (`D_A = D_C/(1+z)`), and `distance_modulus` (`μ = 5·log₁₀(D_L/10 pc)`).
- `cosmology.py` now switches between modes via
  `set_cosmology_mode("lcdm" | "hubble")`; **default is `"lcdm"`**.
  `redshift_to_distance_m` and `redshift_to_distance_lightyears`
  delegate accordingly; the Hubble-law path is preserved for
  backwards compatibility and quick checks.
- DESI-like loader records `metadata["distance_model"]` (the active
  mode) and, when `magnitude` is provided, the
  `metadata["absolute_magnitude"]` from the distance modulus.
- Still simplified: flat geometry only; no radiation, neutrinos, or
  evolving dark energy equation of state. For sub-percent precision
  beyond `z ≈ 3` swap in a proper integrator (Astropy / camb) once
  those become acceptable dependencies.
- Demo at `examples/lcdm_vs_hubble_demo.py` shows the divergence
  between Hubble and ΛCDM distances over `z ∈ [0.01, 1.0]` and
  reports the `data/sample_desi_like.csv` distance ranges in both
  modes.

## Phase 12: Real survey ingestion

- Multi-source ingestion at `cosmic_engine.data.sources`: a
  `DataSource` enum (`DESI`, `SDSS`, `GAIA`, `NASA`, `JPL`,
  `SYNTHETIC`), `tag_source(obj, source)` that stamps the source
  field and a `metadata["data_source"]` key, and `load_catalog(path,
  source)` / `load_catalog_into_registry` that dispatch to the
  per-survey loader.
- Per-survey loaders (offline CSV only):
  - `data/gaia_catalog.load_gaia_like_catalog` — Gaia-style stars
    (`source_id, ra, dec, parallax_mas, phot_g_mean_mag, bp_rp`).
    Distance via `d_pc = 1000 / ϖ_mas`; non-positive parallaxes
    skipped.
  - `data/sdss_catalog.load_sdss_like_catalog` — SDSS-style galaxies
    (`objid, ra, dec, z, modelMag_r`).
  - `data/desi_catalog.load_desi_catalog` — DESI-style galaxies
    (`targetid, ra, dec, z, mag`). The legacy
    `desi_like_catalog.load_desi_like_catalog` (with
    `id, ra_deg, …` columns) remains available for backwards
    compatibility.
- JPL placeholder at `data/jpl_ephemeris.load_jpl_ephemeris_placeholder`
  / `load_jpl_into_registry` returns a Sun + 8-planet snapshot from
  the Phase 13 Keplerian model (see below). Not a real SPICE ingest.
- Real APIs / FITS / SPICE kernels are **not** wired up; ingestion
  is offline CSV today.
- Sample data: `data/sample_gaia_like.csv`, `data/sample_sdss_like.csv`,
  `data/sample_desi.csv`. Demo at `examples/multi_catalog_demo.py`
  combines all four sources and writes `output_multi_catalog.ppm`.

## Phase 13: Orbital mechanics

- Simplified Keplerian orbital mechanics in
  `cosmic_engine.physics.orbital`: `OrbitalElements` dataclass,
  `mean_motion(period)`, `solve_kepler_equation(M, e)` (Newton's
  method, valid for `0 ≤ e < 1`), and
  `orbital_position_from_elements(elements, jd)` returning a
  heliocentric ecliptic Cartesian `Vector3` in meters.
- Solar-system snapshot at
  `cosmic_engine.physics.solar_system.create_solar_system_objects(jd)`
  returns the Sun + eight major planets at the given Julian Date,
  positioned via the Keplerian solver from approximate J2000.0 mean
  elements. Source `"approx_solar_system"`,
  `truth_level=PHYSICS_SIMULATED`,
  `metadata["model"]="simplified_keplerian"`.
- The JPL placeholder (`data/jpl_ephemeris.load_jpl_ephemeris_placeholder`)
  now delegates to this Keplerian model and re-tags the source as
  `"jpl"`. Positions move consistently with time but are still
  **not** real JPL ephemerides.
- No N-body integrator, no perturbations, no relativity, no galaxy
  dynamics — those are deliberately out of scope.
- Demo at `examples/solar_system_physics_demo.py` advances a
  `SimulationClock` by 30 days and prints Earth/Mars position deltas;
  also writes an `output_solar_system.json` snapshot.

## Phase 14: N-body physics

- Newtonian local N-body backend in
  `cosmic_engine.physics.nbody`: `NBodyState` dataclass with
  `validate()`, `objects_to_nbody_state` /
  `apply_nbody_state_to_objects` for round-tripping with
  `UniverseObject`, `compute_accelerations(positions, masses, softening)`
  vectorized direct-summation gravity, `GRAVITATIONAL_CONSTANT`
  (6.67430e−11 m³/kg/s²).
- Two integrators: `euler_step` (testing only, not symplectic) and
  `leapfrog_step` (kick-drift-kick velocity-Verlet, **preferred**
  for orbital stability).
- `NBodySimulator` wraps state + integrator selection with
  `step(dt)` / `run(steps, dt)` / `get_state()`. Validates dt > 0,
  unsupported integrator, negative softening.
- Suitable for local gravitational systems and experiments — **not**
  for galaxy-scale dynamics. No tree-code, FMM, mesh, relativity, or
  GPU acceleration. Keplerian orbital module remains the analytic
  reference.
- Demo at `examples/nbody_solar_demo.py` simulates Sun + Earth + Mars
  for 90 days at 1-hour leapfrog steps; Earth-Sun distance stays
  stable to within a few microns of an AU.

## Phase 15: Barnes-Hut N-body

- Octree-based Barnes-Hut force approximation in
  `cosmic_engine.physics.barnes_hut`: `OctreeNode` dataclass
  (`is_leaf`), `build_octree(positions, masses)`,
  `compute_acceleration_bh(index, node, ...)`, and the batched
  `compute_accelerations_bh(positions, masses, theta=0.5, softening_m=0.0)`.
- Reduces the asymptotic cost of the all-pairs gravitational sum
  from `O(N²)` to roughly `O(N log N)` by treating distant subtrees
  as a single point mass when `s/d < theta`.
- `theta` controls the accuracy/speed tradeoff (smaller = more
  accurate, larger = faster). Default 0.5; observed average relative
  errors versus the exact direct sum on a 1k-body cluster:
  `theta=0.3 → ~8e-4`, `theta=0.5 → ~4e-3`, `theta=0.8 → ~2e-2`.
- `NBodySimulator` accepts `integrator="barnes_hut"` and a `theta`
  parameter; under the hood it uses leapfrog stepping with the BH
  acceleration evaluator. The exact `compute_accelerations` remains
  the reference solver and is unchanged.
- Pure-Python recursive walk; the asymptotic win shows up at
  catalogue scales where `O(N²)` becomes infeasible. At small N the
  vectorized NumPy direct sum still wins on wall-clock time. No
  GPU, parallelism, or Cython.
- Demo at `examples/barnes_hut_demo.py` benchmarks both solvers from
  N=200 to N=5000 across three `theta` values and prints relative
  errors plus timings.

## Phase 16: Unified runtime

- `cosmic_engine.runtime.CosmicRuntime` orchestrates registry, time,
  physics, perception, and rendering preparation through a single
  object. `RuntimeConfig` (validated) selects backends and
  rendering parameters; `SceneState` is a serializable per-frame
  summary with `to_dict` / `to_json`.
- `CosmicRuntime` exposes `add_objects`, `load_sample_data`
  (best-effort load of Gaia/SDSS/DESI sample CSVs and the JPL
  placeholder), `select_active_objects(observer_position)` (sorted
  by id, capped by `max_active_objects`, optionally filtered by
  `active_radius_m`), `step(dt)` (advances the
  `SimulationClock` and runs the configured physics backend on
  registered massive bodies), and `build_scene_state` /
  `last_scene_state`.
- `run_headless_frame(runtime, camera, observer, output_path=None)`
  selects active objects, builds the star photon field, builds a
  galaxy field batch, optionally applies the perception transform,
  and optionally writes a PPM. Always returns a `SceneState`; never
  crashes on empty scenes; non-renderable types (planets, etc.) are
  noted but skipped.
- Headless only — no GUI, GPU, Vulkan, Unreal, or real-time
  windowing yet. The runtime layer adds zero new domain logic; it
  only schedules calls into the modules built in earlier phases.
- Demo at `examples/runtime_demo.py` runs three frames (baseline,
  physics step, perception-enabled) and writes
  `output_runtime_demo.ppm` plus per-frame `SceneState` JSON.

## Phase 17: Live runtime server

- `cosmic_engine.runtime.RuntimeServer` exposes `CosmicRuntime` over a
  plain TCP socket as a stream of newline-delimited JSON messages.
  `start()` opens a listening socket on `host:port` (port 0 lets the
  OS pick a free port); a background tick thread runs
  `runtime.step(period)` at `tick_rate_hz` and a background accept
  thread enrolls new subscribers.
- `broadcast_state(scene_state)` sends a
  `{"type": "scene_state", "data": ...}` message to every connection
  every tick. `broadcast_frame(image_path)` is opt-in and emits a
  `{"type": "frame", "data": "<base64>"}` message on demand.
- `run_streaming_frame(runtime, camera, observer)` writes a PPM into
  `runtime.config.output_directory` and returns
  `(SceneState, frame_path)` so a server (or any caller) can decide
  whether to embed the frame in its broadcast.
- `scene_state_to_json` and `encode_frame_to_base64` live in
  `runtime/stream.py` for client-side use.
- Standard library only (`socket`, `threading`, `time`, `base64`,
  `json`); no GUI, GPU, or networking framework. No authentication —
  intended for trusted local consumers (AI viewer, Unreal bridge,
  web client).
- Demos at `examples/runtime_server_demo.py` (starts a server for
  three seconds) and `examples/runtime_client_demo.py` (connects and
  prints a few `SceneState` messages).

## Phase 18: AI Viewer client

- First standalone viewer client lives outside the engine package at
  `apps/ai_viewer/` (also discoverable as `import ai_viewer` after
  `pip install -e .`). `AIViewerConfig` (validated host / port /
  dimensions / output directory) configures the client; `RuntimeClient`
  speaks the same newline-delimited JSON protocol as the runtime
  server (`receive_message`, `receive_scene_state`, `receive_frame`).
- `FrameBuffer` parses plain (P3) PPM bytes, round-trips via
  `save_ppm`, and renders a `to_ascii_preview(max_width)` for
  terminal-friendly display. NumPy-only; no Pillow, no GUI.
- `AIViewer` orchestrates the loop: `run_once` processes one message;
  `run_loop(max_frames)` drains a bounded number of messages with
  connect/disconnect bookkeeping; `render_scene_state_text` formats
  a `SceneState` as a short readable block; `optional_ai_postprocess`
  applies a deterministic gamma-0.8 brightness curve as a placeholder
  for a future neural step (clearly labeled — **not real AI yet**).
- Frames are written into `config.output_directory` (default
  `outputs/viewer/`); `last_frame_path`, `frames_received`, and
  `scene_states_received` track progress.
- No GUI framework, no GPU, no Unreal — designed to be replaced by a
  future Vulkan / WebGPU / neural renderer without touching any other
  module.
- Demo at `examples/ai_viewer_demo.py` spins up a `RuntimeServer` in
  the same process, drains a few `SceneState` messages, broadcasts
  one PPM frame via `broadcast_frame`, saves it locally, and prints
  the ASCII preview.

## Phase 19: AI Viewer real-time display

- `apps/ai_viewer/window.py` adds a tkinter-based `ViewerWindow`
  (no extra dependency beyond Pillow, which we now ship). The window
  loads each frame as a `PhotoImage` and pumps events via
  `update_idletasks` / `update`; on a headless host it flips
  `available = False` and every later `show_frame` / `update` is a
  no-op so demos still run.
- `apps/ai_viewer/postprocess.py` adds a modular postprocessing
  pipeline: `FramePostProcessor` base (pass-through), plus
  `ContrastBoostProcessor`, `BrightnessProcessor`,
  `ColorShiftProcessor` (per-channel additive shift via a clamped
  256-entry LUT), and `CompositeProcessor` that chains a list of
  processors in order. Pure Pillow / `ImageEnhance`; no neural
  networks here yet.
- `AIViewer` is upgraded to: take a `ViewerWindow` and a
  `FramePostProcessor`; convert PPM bytes → `PIL.Image` (with
  fallback to the in-house parser); apply postprocess; save and
  display; track recent frame timestamps and report FPS every
  five frames; throttle to `config.max_fps`.
- `AIViewerConfig` adds `enable_window: bool = True`,
  `enable_postprocess: bool = True`, `max_fps: float = 30.0`
  (validated > 0).
- New dependency: `pillow>=10`. Still no OpenCV, no GPU, no
  threading complexity in the core loop.
- Demo at `examples/ai_viewer_window_demo.py` spins up a
  `RuntimeServer`, runs a background thread that broadcasts a fresh
  PPM each 100 ms, displays the live frames in a tkinter window,
  applies a `Contrast → Brightness → ColorShift` chain, and prints
  FPS every five frames. Falls back to ASCII preview on headless
  hosts.

## Phase 20: Neural postprocessing

- `apps/ai_viewer/neural_postprocess.py` adds `ONNXFrameProcessor`,
  a `FramePostProcessor` that loads an ONNX model on construction
  and runs it on each frame. Image is resized to the model's input
  size (or kept at its original size if `input_size=None`),
  optionally normalized to `[0, 1]`, reshaped to `NCHW`, run through
  `onnxruntime` (CPU provider), then converted back to PIL and
  resized to the original dimensions. Missing files, invalid models,
  and inference exceptions are caught and recorded in
  `last_error`; the original image is returned unchanged.
- `apps/ai_viewer/postprocess.py` adds `SafeProcessor` (wraps any
  processor; on exception returns the input image and surfaces the
  inner `last_error`) and `build_postprocessor_from_config` (selects
  `SafeProcessor(ONNXFrameProcessor)` when `neural_model_path` is
  set, a deterministic `Composite(Contrast + Brightness)` otherwise,
  or a pass-through if `enable_postprocess` is false).
- `AIViewer` auto-builds the postprocessor from config at
  construction, prints the active processor name, and emits one
  warning if the processor surfaces a `last_error` — not every frame.
- New config fields: `neural_model_path`, `neural_input_width`,
  `neural_input_height` (must be both set or both `None`),
  `neural_normalize` (default `True`).
- A bundled tiny identity model
  (`data/neural_postprocess_identity.onnx`, 159 bytes,
  `(1, 3, 32, 32)` Identity) makes the demo and tests exercise the
  inference path without a real trained model. Rebuild with
  `python scripts/build_neural_postprocess_model.py`.
- No model training, no PyTorch, no OpenCV, no GPU — CPU
  `onnxruntime` only, fully fallback-safe.
- Demo at `examples/ai_viewer_neural_demo.py` runs deterministic by
  default (`--model` omitted) or neural with `--model` and optional
  `--neural-input-width/height`.
