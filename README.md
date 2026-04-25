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

## Phase 21: Neural photon warp

- AI now operates **before** rendering, in photon space, not just on
  the final image. `cosmic_engine.ai.ONNXPhotonWarpModel` implements
  the existing `AIWarpModel` interface but with a documented
  9-float input and 7-float output schema:
  `[dir.x, dir.y, dir.z, brightness, color_r, color_g, color_b,
  observer_beta, warp_factor] → [new_dir.x, new_dir.y, new_dir.z,
  new_brightness, new_color_r, new_color_g, new_color_b]`.
- Each `predict_*` call runs one ONNX inference, normalizes the
  direction, clamps brightness to non-negative, and clamps colors to
  `[0, 255]`. Missing files, invalid models, and inference exceptions
  silently fall back to the deterministic `apply_*_warp` helpers and
  record `last_error`. `confidence()` returns `0.85` when the session
  is healthy and `0.3` after any failure.
- A bundled mock model (`data/photon_warp_model.onnx`, 448 bytes,
  9→7 linear MatMul + Add) lets the demo and tests exercise the
  inference path without a real trained model. Rebuild with
  `python scripts/build_photon_warp_model.py`.
- `apps/ai_viewer/neural_warp_viewer.NeuralWarpViewer` drives a frame
  end-to-end on the local side: select active objects, build the
  star photon field + galaxy field batch, run
  `transform_photon_field(..., ai_model=warp_model)`, then render
  to PPM. Prints warp mode, confidence, photon count, and frame
  time per call. `run_loop(max_frames, output_pattern)` is the
  batch driver.
- `AIViewerConfig.use_photon_warp: bool = False` and
  `photon_warp_model_path: str | None = None`. When
  `use_photon_warp=True`, `AIViewer` skips the image postprocess
  pipeline (the warp happens at the source) and emits a one-line
  warning if no model path is provided.
- No PyTorch, no GPU, no batching yet (per-sample inference). The
  deterministic perception transform remains the default; the
  scalar / vectorized / Barnes-Hut / runtime / runtime-server paths
  are all untouched.
- Demo at `examples/neural_photon_warp_demo.py` renders three
  PPMs into `outputs/viewer/`: deterministic, neural (warp_factor=2),
  and neural-extreme (warp_factor=50).

## Phase 22: Batch neural photon warp

- `cosmic_engine.ai.BatchONNXPhotonWarpModel` runs **one ONNX
  inference per frame** over a `(N, 9)` photon batch, where the
  Phase 21 model ran one inference per sample. Construction raises
  on missing/invalid models; `predict_batch` raises on bad input
  shape, inference failure, or unexpected output shape — fallback
  policy lives in the integration layer.
- `cosmic_engine.perception.vectorized_ai_transform` provides
  `build_photon_warp_input(batch, observer)` (packs to `(N, 9)`
  with the documented schema), `validate_warp_output(out, n)`
  (rejects non-`(N, 7)` arrays), and `apply_batch_ai_warp(batch,
  observer, model)`. The latter routes silently to the deterministic
  vectorized transform when the model is `None`, raises in any way,
  or returns the wrong shape — so the engine never crashes on a bad
  model.
- `data/photon_warp_model.onnx` is now declared with a symbolic
  `["batch", 9]` input dimension so the same file serves both the
  scalar (Phase 21) and batch (Phase 22) inference paths. Existing
  scalar tests still pass unchanged.
- `apps/ai_viewer/NeuralWarpViewer` accepts a new
  `batch_warp_model: BatchONNXPhotonWarpModel | None` argument; when
  set, stars go through the batch AI path and galaxies stay on the
  deterministic scalar path. Reports photon count, batch inference
  time, and fallback-used flag per frame; mode is `"batch_neural"`.
- Demo at `examples/batch_neural_photon_warp_demo.py` (50k
  synthetic stars; deterministic vs. AI-batch vs. forced-fallback
  timing) writes `outputs/viewer/output_batch_photon_warp.ppm`. On
  this machine: 50k photons → 29 ms with the bundled model, ~10 ms
  for deterministic, vs. ~4.5 s the per-sample Phase 21 path would
  take (≈150× speedup).
- Per-sample `ONNXPhotonWarpModel` is preserved for small fields and
  for cases where one needs per-photon control flow.

## Phase 23: Streaming and LOD

- New `cosmic_engine.streaming` package adds three primitives:
  - `SpatialIndex(cell_size_m)` — hashed uniform-grid index with
    `insert` / `build` / `query_cell` / `query_radius`. Pure
    Python, deterministic, no external libs.
  - `TileStore(data_directory)` — JSON-on-disk tile reader/writer
    with `save_tile` / `load_tile` / `list_tiles` /  `has_tile`,
    plus `partition_objects_into_tiles(objects, cell_size_m)` that
    buckets a registry into tile-shaped JSON files keyed by cell
    index.
  - `compute_lod_weight(distance)` and
    `select_lod_objects(objects, observer, max_objects, seed=42)`
    — inverse-square-distance weighted, sampling without replacement
    via NumPy's `default_rng` for deterministic frame-to-frame
    selection that preserves the spatial distribution.
- `CosmicRuntime` integrations:
  - New optional fields `spatial_index` and `tile_store` (both
    default `None` so existing behavior is unchanged).
  - `build_spatial_index(cell_size_m)` builds an in-memory grid over
    the current registry.
  - `enable_streaming(cell_size_m, data_directory, clear_registry=True,
    cache_size=16)` partitions the registry to disk, optionally
    empties the in-memory registry, and configures an LRU tile
    cache.
  - `select_active_objects(observer)` now picks candidates from the
    spatial index → or tile-store-with-cache → or the in-memory
    registry, then runs the LOD selector to cap at
    `config.max_active_objects`. `last_loaded_tiles` records which
    tiles a frame touched.
- File-based JSON tiles only — no databases, no multiprocessing,
  no GPU. The current registry-cleared path keeps memory bounded
  by the cache, not the dataset.
- Demo at `examples/streaming_lod_demo.py` partitions 200k synthetic
  galaxies into ~4,900 tiles (~102 MiB on disk), clears the in-memory
  registry, and walks an observer through the field reporting
  per-step tiles touched, cache size, and active-object count
  (LOD-bounded at 5,000 per frame).

## Phase 24: Neural field rendering

- New `apps/ai_viewer/neural_field/` package adds a CPU-only
  Gaussian-splatting renderer that turns discrete galaxy / density
  samples into a continuous visual field — a first step toward
  neural rendering.
  - `GaussianPoint` carries a 3D position, RGB color, scalar
    intensity, and a world-space `sigma`.
  - `build_gaussian_field_from_galaxy_batch(batch, sigma_scale=1.0)`
    builds one point per galaxy with sigma proportional to distance.
  - `build_gaussian_field_from_density(grid, threshold, cell_size_m)`
    samples a 3D density grid into points, skipping cells at or
    below the threshold.
  - `GaussianSplatRenderer(width, height, camera)` projects each
    point through the camera basis, computes a screen-space sigma
    from world sigma and depth, splats a 2D Gaussian
    (`I·exp(-r²/(2σ²))`) over a 3-σ bounding box into a float
    accumulator, and tone-maps by max-channel normalization.
- `AIViewerConfig` adds `render_mode: "ppm" | "gaussian"`,
  `gaussian_sigma_scale: float = 1.0`, and
  `max_gaussian_points: int = 50_000`. `render_mode="gaussian"`
  switches `AIViewer` to a passthrough postprocessor and prints a
  note that the gaussian path is driven through `NeuralWarpViewer`
  / `GaussianSplatRenderer` directly (the network client doesn't
  see 3D objects).
- Pure NumPy; loops over points, vectorizes each footprint. ~50k
  splats render to a 256×256 frame in ~0.8 s on CPU. GPU
  acceleration is the next phase.
- Demo at `examples/gaussian_splat_demo.py` builds 50k synthetic
  galaxies, projects them, and writes
  `outputs/viewer/output_gaussian.ppm`.

## Phase 25: Relativistic neural field fusion

- Phase 21/22 perception transforms now apply directly to Gaussian
  fields. `apps/ai_viewer/neural_field/field_warp.py` adds:
  - `warp_gaussian_point(point, observer, ai_model=None)` — converts
    each `GaussianPoint` to a `PhotonSample`, runs it through the
    existing deterministic / AI photon transform, then reconstructs
    the point with the warped direction expressed as a position
    rotation around the observer at the same distance. Always
    returns a fresh point; the input is never mutated. Sigma grows
    gently with `warp_factor**0.25`.
  - `warp_gaussian_field(points, observer, ai_model=None,
    max_points=None)` — the batch wrapper, deterministic, empty-list
    safe, with an optional truncation knob.
- `GaussianPoint` now carries `object_id`, `truth_level`, and
  `metadata`; the warp records `original_position`, `warp_factor`,
  and a `warped` flag in the metadata so callers can trace what
  moved where. The galaxy → field builder propagates the source
  galaxy's id and truth level.
- `GaussianSplatRenderer(width, height, camera, normalize_exposure=True)`
  now scrubs non-finite pixels with `np.nan_to_num` before
  tone-mapping, so an extreme warp can't poison the frame; the
  final image is always clamped to `[0, 255]`.
- `AIViewerConfig` adds `enable_field_warp: bool = False` and
  `field_warp_mode: str = "deterministic"` (`"none" | "deterministic"
  | "ai"`, validated). Existing `render_mode` and `gaussian_*`
  fields are unchanged.
- Demo at `examples/relativistic_gaussian_field_demo.py` warps a
  20k-galaxy Gaussian field at `warp_factor ∈ {1, 5, 50}` and writes
  `outputs/viewer/output_gaussian_warp_{1,5,50}.ppm`. Visible
  effect: as warp_factor grows, the field compresses toward the
  forward axis (455 → 152 → 105 visible splats) and the Doppler
  blueshift saturates (avg blue 18 → 23 → 45, avg red collapses to
  0), exactly the perception behavior Phase 4 introduced for
  point samples — now applied to a continuous field.

## Phase 26: GPU splatting foundation

- New `apps/ai_viewer/neural_field/gpu/` subpackage adds the
  scaffolding for a future Vulkan / WebGPU / CUDA backend without
  taking on a real GPU dependency:
  - `GPUDevice(backend="auto")` resolves to one of `"none"` /
    `"cpu"` / `"mock_gpu"`. Auto-detect currently returns `"cpu"`;
    real probes land when a backend is bound.
  - `GPUGaussianBuffer` is the upload-shaped data view: parallel
    `(N, 3)` positions / colors and `(N,)` intensities / sigmas with
    `from_gaussian_points`, `validate`, and `to_numpy` for
    serialization.
  - `CPUSplatFallback` re-uses the Phase 24 `GaussianSplatRenderer`
    so the fallback path is the existing reference renderer.
  - `GaussianSplatPipeline(device)` selects between the CPU
    fallback and a vectorized `mock_gpu` renderer that does
    whole-batch projection and frustum culling in NumPy and only
    walks visible points to splat.
- `AIViewerConfig` adds `use_gpu_pipeline: bool = False`. When
  `render_mode == "gaussian"`, `AIViewer` prints which path
  (CPU vs GPU mock) downstream callers should drive.
- All-NumPy implementation; CPU `GaussianSplatRenderer` and Phase 25
  field warp continue to work unchanged. The mock backend is a
  stand-in for a future real GPU implementation, not a substitute
  for one.
- Demo at `examples/gpu_splat_demo.py` renders 100k points through
  both backends. On this machine: CPU 1870 ms, `mock_gpu` 1766 ms
  (~1.06× — the mock is mostly architectural; real speedup arrives
  with a real GPU kernel).

## Phase 27: Real GPU rendering

- New WebGPU backend at `apps/ai_viewer/neural_field/gpu/`:
  - `WebGPUDevice(GPUDevice)` tries `wgpu.gpu.request_adapter_sync` +
    `request_device_sync` on construction. On failure (no GPU /
    drivers / display) it downgrades silently to `backend="cpu"`
    and records the cause in `last_error`. `is_available()` stays
    `True` so the pipeline still has a usable target.
  - `WebGPUGaussianBuffer` packs Gaussian points into a 32-byte
    storage layout (`pos.xyz, intensity, color.rgb, sigma`),
    uploads to a GPU `STORAGE | COPY_DST` buffer, and keeps a CPU
    shadow so tests can verify the layout without a GPU.
  - `apps/ai_viewer/neural_field/gpu/shaders/splat.wgsl` is the
    real WGSL shader. The vertex stage runs the relativistic
    direction warp + Doppler color shift **in-shader** so the GPU
    sees pre-warped Gaussians; the fragment stage evaluates
    `exp(-r²/(2σ²))` over a 3-σ quad with additive blending.
  - `WebGPUSplatRenderer(device, width, height)` initializes a
    pipeline, renders to an off-screen RGBA texture, and reads it
    back to a NumPy array. Raises `RuntimeError` when the device
    isn't real WebGPU; the pipeline catches that and falls back to
    CPU.
- `GaussianSplatPipeline.render(points, camera, observer=None)`
  now dispatches `webgpu` → real GPU path → CPU fallback on any
  exception, with `mock_gpu` and `cpu` paths preserved.
- `AIViewerConfig` adds `gpu_backend: str = "auto"` (`"auto" |
  "webgpu" | "cpu"`, validated) and `enable_shader_warp: bool =
  True`. `AIViewer` resolves the active backend and prints a note
  describing pipeline + warp mode.
- New runtime dependency: `wgpu>=0.20`. Importing the engine never
  requires a GPU — adapter probing is lazy and graceful.
- Demo at `examples/webgpu_splat_demo.py` renders 200k points
  through both paths. On a host without a GPU adapter the demo
  reports the resolved backend, the underlying error, and falls
  back to CPU for both runs (~3.7 s for 200k points / 256×256). On
  a host with a real WebGPU adapter the second run uses the GPU
  pipeline.

## Phase 28: Relativistic rendering

- New `apps/ai_viewer/neural_field/gr/` package implements
  weak-field gravitational lensing and a Schwarzschild-style black
  hole — physically-inspired distortion approximations, **not**
  full GR ray tracing:
  - `compute_deflection_angle(impact_parameter_m, mass_kg)` returns
    `α = 4GM/(c²b)` capped at 85° to keep the resulting 2D rotation
    well-defined.
  - `apply_lensing(direction, lens_position, observer_position,
    mass_kg)` rotates a unit ray toward the lens by α. Rays passing
    behind the observer or directly through the lens pass through
    unchanged.
  - `apply_lensing_to_points(points, lens, mass, observer)` lenses
    every Gaussian in a list, preserving each point's distance to
    the observer and recording `metadata["lensed"]`.
  - `BlackHole(position, mass_kg)` exposes
    `schwarzschild_radius()`, `is_inside_event_horizon(point)`, and
    `deflection_strength(distance)` (saturates at the horizon).
    `apply_black_hole_to_points` removes inside-horizon points and
    lenses the rest.
- WGSL shader at
  `apps/ai_viewer/neural_field/gpu/shaders/gr_splat.wgsl` is a
  superset of `splat.wgsl`: vertex stage applies the same
  weak-field bend toward a `bh_position` uniform; fragment stage
  outputs pure black for points inside the event horizon. With
  `enable_lensing=0` the shader behaves identically to
  `splat.wgsl`.
- `WebGPUSplatRenderer.enable_gr_effects(black_hole, enable_lensing)`
  swaps the shader on the next render and wires GR uniforms into
  the same uniform buffer (now 7 × vec4, still padded to 256 bytes).
- `AIViewerConfig` adds `enable_gr: bool = False`,
  `black_hole_mass_kg: float | None = None`, and
  `black_hole_position: tuple[float, float, float] | None = None`,
  validated for positive mass and 3-tuple position.
- All math is bounded to avoid NaNs at the horizon: deflection
  angle clamps at 85°, deflection strength saturates at `1e6`,
  lensing skips coincident-with-observer / behind-observer cases,
  inside-horizon points are removed before lensing.
- Demo at `examples/black_hole_lensing_demo.py` renders a 20k-galaxy
  field in three modes (no-GR / weak lensing / strong black hole)
  with deliberately oversized masses for visibility — physical
  supermassive black holes produce sub-pixel bends at galaxy-cluster
  scales. The resulting PPMs differ dramatically: 1.1k → 21k →
  20.9k visible pixels, with the black-hole mode redistributing
  the field around the lens.

## Phase 29: Geodesic ray marching

- Stepwise photon trajectory integration replaces Phase 28's single-
  impulse deflection with a per-step bend, producing curved paths
  rather than a one-shot rotation:
  - `apps/ai_viewer/neural_field/gr/geodesic.py` —
    `schwarzschild_acceleration` (toward-mass `2GM/r²`, with a
    velocity placeholder for a future GR upgrade) and
    `integrate_geodesic_step(position, direction, step_size, mass_kg)`
    that rotates direction by `dθ = 2GMs/(c²r²)` per step (capped at
    π/4) and advances the position by `step_size`.
  - `apps/ai_viewer/neural_field/gr/ray_marcher.py` —
    `GeodesicRayMarcher(black_hole, step_size, max_steps)` runs a
    fixed-iteration loop over `integrate_geodesic_step` and returns
    `(final_direction, absorbed)`. Validates positive `step_size`
    / `max_steps`; rays crossing the event horizon are absorbed;
    the final direction is renormalized to scrub float drift over
    many cos/sin updates.
  - `trace_points_through_geodesic(points, marcher, observer)`
    drops absorbed points and lenses the rest by replacing each
    point's apparent direction with the marched final direction
    (distance preserved exactly).
- New WGSL shader at
  `apps/ai_viewer/neural_field/gpu/shaders/geodesic_splat.wgsl`
  unrolls the same algorithm into a fixed `GEODESIC_STEPS = 8`
  loop in the vertex stage. Per-step length goes through
  `bh_params.w`. With `enable_geodesic = 0` the shader is a no-op.
- `WebGPUSplatRenderer.enable_gr_effects(...)` now takes
  `gr_mode: "none" | "lensing" | "geodesic"`, `geodesic_steps`,
  and `geodesic_step_size`. Switching to `"geodesic"` swaps the
  shader on the next render and feeds the step size through the
  same uniform block (still 7 × vec4, padded to 256 bytes).
- `AIViewerConfig` adds `gr_mode: str = "lensing"` (validated
  against the same three values), `geodesic_steps: int = 8`, and
  `geodesic_step_size: float = 1e9` (both validated positive).
- All math bounded: per-step deflection capped at π/4, zero
  direction passes through, zero mass / origin skip the bend, the
  marcher always terminates within `max_steps`, and every code
  path has explicit no-NaN tests.
- Demo at `examples/geodesic_blackhole_demo.py` renders a 20k
  galaxy field in three modes (no-GR / lensing / geodesic) with
  the same black hole. Visible-pixel counts: 1,077 → 33,689 →
  12,629; the geodesic mode absorbs 792 rays into the event
  horizon and produces a more concentrated bend pattern than the
  single-impulse lensing approximation.

## Phase 30: Neural spacetime field

- New pluggable interface: `cosmic_engine.ai.SpacetimeFieldModel`
  with `query_acceleration(position, direction=None)` and
  `confidence()`. Subclasses can swap in a learned curvature field
  in place of the analytical Schwarzschild formula.
- ONNX implementation: `cosmic_engine.ai.ONNXSpacetimeField` runs a
  CPU `onnxruntime` session over a `(1, 6) → (1, 3)` graph
  (`[pos.x, pos.y, pos.z, dir.x, dir.y, dir.z] → [a.x, a.y, a.z]`).
  Construction errors propagate so the caller decides on fallback;
  inference errors raise `RuntimeError` and store `last_error`.
  Output is NaN-scrubbed and clamped to a finite magnitude;
  `confidence()` is `0.9` after a clean run, `0.2` after a failure.
- New step function:
  `apps/ai_viewer/neural_field/gr/neural_geodesic.integrate_geodesic_step_neural`
  uses the model's acceleration when available and silently falls
  back to the analytical `schwarzschild_acceleration` if the model
  raises or returns NaNs. Per-step rotation capped at π/4 (same
  stability bound as Phase 29).
- `GeodesicRayMarcher` now accepts an optional
  `spacetime_model: SpacetimeFieldModel | None` parameter; when
  set, every step routes through `integrate_geodesic_step_neural`.
  The marcher's existing horizon-absorption + final renormalization
  contracts are unchanged.
- `geodesic_splat.wgsl` has a new `neural_params` vec4 with a
  `use_neural_field` flag. The GPU path currently stays on the
  analytical formula even when the flag is set — running ONNX
  inside the vertex stage requires a tensor-shader bridge that
  isn't part of the wgpu surface yet — but the uniform is plumbed
  through so a future GPU upgrade is purely additive.
- `WebGPUSplatRenderer.enable_gr_effects(...)` accepts
  `use_neural_field` and `neural_confidence`; the uniform block
  grew to 8 × vec4 (still padded to 256 bytes).
- `AIViewerConfig` adds `use_neural_spacetime: bool = False` and
  `spacetime_model_path: str | None = None`. A missing model under
  `use_neural_spacetime=True` is allowed — the runtime falls back
  to the analytical path with a warning.
- A bundled mock spacetime model lives at
  `data/spacetime_field_identity.onnx` (257 bytes, 6→3 linear
  MatMul + Add producing `a = -k·position`). Rebuild with
  `python scripts/build_spacetime_field_model.py`.
- Demo at `examples/neural_spacetime_demo.py` traces 20k galaxy
  rays through the marcher in two modes (analytical only / neural
  with fallback). With the bundled model the neural path absorbs
  more rays (12,486 surviving vs 19,208 analytical) because the
  model's pull-toward-origin acceleration nudges more rays into
  the event horizon. With `--model /missing.onnx` the fallback
  produces identical output to the analytical run.

## Phase 31: Training neural spacetime

- New training package at `cosmic_engine.ai.training` lets you
  generate a supervised dataset from analytical Schwarzschild
  acceleration, fit a small MLP on CPU, and export it to an ONNX
  file the runtime `ONNXSpacetimeField` reads natively. PyTorch
  is **only** imported by submodules in this package — runtime /
  inference paths still take no torch dependency.
  - `SpacetimeDataset(num_samples, mass_kg, radius_range, seed)`
    samples uniform-in-log radii on a 3D ball, draws random unit
    directions, and labels every row with the analytical
    acceleration (vectorized via `schwarzschild_acceleration_batch`).
    All construction args validated.
  - `SpacetimeMLP(r_char, a_char, hidden=64)` is a 6 → 64 → 64 →
    64 → 3 ReLU MLP (~9k params) with input position rescaled by
    `r_char` and output rescaled by `a_char` *inside the forward
    pass*. The exported ONNX therefore speaks raw SI units.
  - `train_model(dataset, epochs, batch_size, learning_rate)`
    derives `r_char` (geometric mean of the radius range) and
    `a_char` (= `2GM/r_char²`) from the dataset, runs Adam + MSE,
    and returns `(model, per_epoch_losses)`.
  - `export_to_onnx(model, output_path)` writes the trained model
    with `(N, 6) → (N, 3)` shape and dynamic batch axis.
- New runtime dependency: `torch>=2.0` (CPU wheels). Importing the
  engine never triggers a torch import; only `cosmic_engine.ai.training.*`
  pulls it in.
- Demo at `examples/train_spacetime_model.py` trains for 20
  epochs over 100k samples on a tight radius range
  (`2e10..5e10` m) in ~5 s and writes
  `models/spacetime_field.onnx` (gitignored). Loss reduces from
  `3.3e-3` → `4.7e-6` (~700× reduction).
- Demo at `examples/test_trained_spacetime_model.py` loads the
  trained ONNX into the runtime `ONNXSpacetimeField`, runs
  inference on 2k held-out samples, and reports avg / median
  relative error and cosine similarity. After the bundled training
  run: avg `|err/truth| ≈ 1.9%`, median `≈ 1.5%`, cosine similarity
  `≈ 0.9998`.

## Phase 32: Multi-scale universe

- New package `cosmic_engine.multiscale` introduces hierarchical
  scale zones so a single observer can navigate continuously from
  microscale to intergalactic distances without swapping engines
  or reloading data.
  - `ScaleZone(name, min_scale_m, max_scale_m, representation_type, metadata)`
    is a validated dataclass. `representation_type` must be one
    of `("galaxy_field", "star_field", "nbody", "density_field",
    "neural_field")`.
  - `ScaleManager(zones)` validates non-overlap, sorts by
    `min_scale_m`, and exposes `get_zone(distance_m)`,
    `get_neighbor_above/below(zone)`, and
    `get_representation(distance_m, runtime)`.
  - `DEFAULT_ZONES` spans ~27 orders of magnitude:
    `microscale` (0..1e9 m, density_field) →
    `solar_system` (1e9..1e16 m, nbody) →
    `interstellar` (1e16..1e20 m, star_field) →
    `intergalactic` (1e20..1e27 m, galaxy_field).
  - `get_representation_for_zone(zone, runtime)` filters the
    runtime registry to objects matching the zone's representation
    type and returns `{type, zone_name, objects, object_count, blended}`.
  - `compute_transition_alpha(distance_m, zone_a, zone_b, blend_width=0.1)`
    is a linear ramp around the boundary `zone_a.max_scale_m`
    spanning `blend_width × (zone_a.max - zone_a.min)`.
  - `blend_representations(rep_a, rep_b, alpha)` takes a
    `(1 - alpha)` deterministic prefix from `rep_a` and an
    `alpha` prefix from `rep_b`, marking the result `type =
    "blended"` with `primary_zone` / `secondary_zone` metadata.
- `CosmicRuntime` gained `scale_manager`, `multiscale_blend_width`,
  `enable_multiscale(scale_manager, blend_width=0.1)`, and
  `get_multiscale_scene(observer_position)`. The latter computes
  the distance from the observer to the **nearest** active object
  (most semantically meaningful proxy for "what scale of structure
  am I in?"), picks the matching zone, and blends with the upper
  or lower neighbor when the observer is past that zone's
  midpoint. Fully lazy-imports the multiscale package so existing
  pipelines pay zero cost.
- `AIViewerConfig` gained `enable_multiscale: bool = False` and
  `multiscale_blend_width: float = 0.1` (validated to lie in
  `[0.0, 1.0]`).
- Demo at `examples/multiscale_navigation_demo.py` loads bundled
  Gaia + SDSS + DESI + JPL data, plus 2,000 synthetic galaxies
  and a fresh solar-system snapshot (~2,039 objects total), then
  walks an observer through five distance scales (`5e25 → 5e21 →
  5e17 → 5e10 → 1e3` m). Prints the active zone and renders a
  PPM frame using the appropriate path for each
  `representation_type` (gaussian splatter for galaxy_field,
  photon field for star_field, blank placeholder for nbody /
  density_field). With the bundled dataset the demo crosses **3
  zone transitions** and writes 5 frames to
  `outputs/viewer/output_multiscale_*.ppm`.

## Phase 33: Observer-dependent reality

- New package `cosmic_engine.observer` introduces a multi-observer
  reality model: many observers can share the same universe state
  but each one perceives it differently through their own pose,
  velocity, perception knobs, and (optionally) different spacetime
  / AI warp models.
  - `Observer(id, position_m, velocity_m_s, forward, up,
    warp_factor=1.0, spacetime_model=None, ai_warp_model=None,
    config={})` is a dataclass with `beta()` and `validate()`
    (rejects empty id, warp_factor < 1.0, beta >= 1.0, zero
    forward / up).
  - `RealityView(observer_id, scene_state, representation_type,
    frame_data=None, metadata={})` packages everything one
    observer perceives, with `to_dict()` / `summary()` helpers.
  - `ObserverManager` is an O(1) dict-backed registry with
    `add_observer / remove_observer / get_observer /
    list_observers` (sorted by id for determinism).
- `CosmicRuntime` gained an `observer_manager` field (empty by
  default — single-observer pipelines are unaffected) plus two
  methods:
  - `render_for_observer(observer_id, camera) -> RealityView`:
    selects active objects from the observer's position (re-uses
    streaming + LOD + multiscale), applies the observer's
    perception transform (relativistic aberration + beaming +
    `warp_factor`, optionally swapped for an `AIWarpModel`),
    runs the photon-field renderer, and packages the resulting
    pixels + metadata. Spacetime / AI model failures fall back
    cleanly and are recorded in `RealityView.metadata`.
  - `step_all_observers(delta_seconds) -> list[RealityView]`:
    advances the clock once and renders one view per registered
    observer (camera taken from `observer.config['camera']` when
    set, else built from the observer's pose).
- `RuntimeServer` gained an observer-aware tick: when the runtime
  has any registered observers, each tick calls
  `step_all_observers` and broadcasts one
  `{"type": "reality_view", "observer_id": ..., "scene_state":
  ..., "representation_type": ..., "metadata": ..., "frame": ...}`
  per observer. Subscribers can call `set_observer_filter(conn,
  observer_id)` to receive only one observer's stream;
  unfiltered connections receive all of them. With zero
  observers the server falls back to the original single
  `scene_state` broadcast.
- `AIViewer` learned to consume `reality_view` messages: a new
  `AIViewerConfig.observer_id: str | None` field selects which
  observer the viewer follows (`None` = follow all). Each
  message prints a one-line `observer=... warp_factor=...
  spacetime=... rep=...` note.
- Demo at `examples/multi_observer_demo.py` creates 3 observers
  on the bundled Gaia + SDSS + DESI + JPL data:
  1. **baseline** — stationary, `warp_factor=1.0`, no models.
  2. **relativistic** — `beta=0.6`, no models (pure aberration
     + beaming).
  3. **neural** — `beta=0.1`, `warp_factor=8.0`, attached
     `SpacetimeFieldModel` (mock).
  Renders 2 frames per observer (6 PPMs total). Reports the
  mean absolute pixel delta of each observer's frame vs the
  baseline; with the bundled dataset both relativistic and
  neural show ~`0.08` mean delta — same data, different
  perceived realities — at ~`280 ms` for `3 observers ×
  2 frames`.
- 25 new tests in `tests/test_observer_system.py` cover Observer
  validation, ObserverManager add/remove/get/list/duplicate
  rejection, RealityView dict + summary, runtime integration
  (render returns RealityView, unknown observer raises,
  step_all_observers returns one per observer, two observers
  render to distinct files, broken spacetime model falls back,
  multiscale produces non-flat representations), and
  AIViewerConfig observer_id default + validation.

## Phase 34: Subjective time and causality

- New package `cosmic_engine.time` introduces per-observer proper
  time, an event log, and a light-cone-based visibility filter so
  every observer perceives a causally consistent slice of the
  universe.
  - `gamma_from_beta(beta)` — Lorentz factor with a hard clamp at
    β < 1 (handles negative β symmetrically; near-1 inputs produce
    a large but finite γ instead of an inf).
  - `advance_proper_time(tau, delta_t, beta, gravitational_potential=None)`
    advances proper time using `dτ = dt / γ`, optionally
    multiplied by the weak-field factor `(1 + Φ/c²)`. The factor
    is floored at `1.0e-3` so absurd inputs can't drive `dτ`
    negative.
  - `gravitational_potential_weak(position, [(mass_pos, mass_kg), …])`
    returns `Φ ≈ -Σ G·M / r`, with `r` clamped at 1 m so a
    coincident mass doesn't yield `-inf`.
  - `Event(id, position_m, time_t, payload, source)` — pointlike
    event record. `position_m` is canonicalized to a `(3,)`
    numpy float64 array on construction.
  - `EventStore` — list-backed registry with
    `add_event / list_events / query_time_window`.
  - `is_event_visible(event, observer_position, observer_time_t)`
    enforces the past-light-cone condition
    `c·(t_obs − t_event) ≥ |x_obs − x_event|` (and rejects future
    events). 1 ns slack on the cone absorbs floating-point drift.
  - `compute_retarded_time(observer_position, observer_time_t, source_position)`
    returns `t_emit ≈ t_obs − distance / c` (static-source
    approximation; no iteration on a moving emitter).
- `Observer` gained `proper_time_tau` and `coordinate_time_t`
  fields plus `advance_time(delta_t, masses_for_potential=None)`
  that steps both clocks (SR + optional weak-field GR).
- `CosmicRuntime` gained an `event_store` and a
  `coordinate_time_t`. `runtime.step(delta_t)` now also advances
  `coordinate_time_t` and every registered observer's clocks,
  using up to the 16 most-massive registry objects as
  gravitational sources for the weak-field correction.
  `get_visible_events(observer)` filters the event store
  through the observer's past light cone.
- `RealityView` gained `proper_time_tau`, `coordinate_time_t`,
  and `visible_event_count` (top-level fields, not just
  metadata). `to_dict()` and `summary()` surface them.
- `RuntimeServer.broadcast_observer_view` extends the wire
  protocol with `proper_time_tau`, `coordinate_time_t`, and
  `visible_event_count` so subscribers can render an observer's
  subjective timeline directly.
- `AIViewer` displays the new fields in its per-message print
  (`tau=… t=… events=…`).
- Demo at `examples/causality_demo.py` creates 3 observers
  (`near` 1 ls from origin, `far` 10 ls from origin, `moving` at
  1 ls but with β = 0.8) and 3 events (origin flash at t=0,
  far-side flash at t=0, future flash at t=6 s). Stepping 6× at
  Δt = 2 s shows:
  - light-cone gating: `near`/`moving` see `origin_flash`
    immediately (1 ls away, t > 1 s), but `far` only sees it
    after t ≥ 10 s; `far` sees `far_flash` instantly because
    they're collocated; `future_flash` (emitted at t=6 from the
    origin) becomes visible to `near`/`moving` at t ≥ 7 s and
    is still invisible to `far` at t = 12 s.
  - subjective time: at t = 12 s, `near` and `far` have τ = 12 s
    (no motion, weak GR), but `moving` has τ = 7.2 s
    (γ = 1.667, Δ = 4.8 s of dilation).
- 34 new tests in `tests/test_causality.py` cover gamma /
  proper-time math, the gravitational-potential clamp, Event /
  EventStore behavior, light-cone visibility (including future
  events and on-cone slack), retarded-time monotonicity,
  Observer.advance_time at rest / in motion / in a well, and
  runtime integration (event store presence, step advances
  global + per-observer time, get_visible_events filters
  correctly, render_for_observer attaches τ / t /
  visible_event_count).

## Phase 35: Emergent reality layer

- New package `cosmic_engine.reality` adds a deterministic,
  inspectable rule engine that can modify perception / rendering
  / interpretation per observer or region without touching the
  underlying universe state. Every rule declares its
  **domain** (`PHYSICS / PERCEPTION / RENDERING / CAUSALITY /
  DATA_INTERPRETATION / SYMBOLIC`), **kind** (`PHYSICAL /
  PERCEPTUAL / SYMBOLIC / EXPERIMENTAL`), priority, and
  parameters; effects show up in the produced
  `RealityView.active_rule_ids` / `reality_metadata`.
  - `RuleContext(observer_id, observer_position_m,
    observer_velocity_m_s, warp_factor, coordinate_time_t,
    proper_time_tau, scale_zone, truth_level_counts,
    active_rule_ids, metadata)`. `clone()` returns a deep,
    independent copy; `to_dict()` is JSON-friendly.
  - `RealityRule` (kw-only dataclass): base class. Default
    `applies_to_context` returns `self.enabled`; default
    `apply` clones the context and echoes
    `self.parameters` under `metadata['rules'][self.id]` so
    even base-class declarative markers are fully traceable.
  - Built-in subclasses:
    - `WarpAmplificationRule` (`PERCEPTION` /
      `PERCEPTUAL`): multiplies `warp_factor` by
      `parameters['factor']` (`> 0` enforced; result
      floored at 1.0).
    - `RedshiftSymbolicColorRule` (`DATA_INTERPRETATION` /
      `SYMBOLIC`): sets `metadata['color_mapping'] =
      'symbolic_redshift'`.
    - `CausalityRelaxationRule` (`CAUSALITY` /
      `EXPERIMENTAL`): sets `metadata['causality_mode'] =
      'relaxed'` (declarative; visibility filter
      unchanged).
    - `NeuralRealityRule` (`PERCEPTION` / `PERCEPTUAL`):
      sets `metadata['use_neural_perception'] = True` and
      optional `metadata['model_id']`.
  - `RealityRuleEngine.add_rule / remove_rule / list_rules /
    evaluate`. Duplicate ids are rejected; `list_rules` sorts by
    `(-priority, id)` for deterministic order; `evaluate` clones
    the input context, applies enabled rules in priority order,
    and appends each applied rule's id to
    `context.active_rule_ids` (without duplicates).
- Three presets in `cosmic_engine.reality.presets`:
  - `create_scientific_reality_preset()` → empty list
    (default); the absence of active rule ids is the
    inspectable signal that no reality modification ran.
  - `create_hypertravel_reality_preset()` → 4 rules
    (`warp_amplification × 4.0`, `redshift_symbolic_color`,
    `neural_reality` with `model_id = neural_perception_v1`,
    `causality_relaxation`).
  - `create_blackhole_perception_preset()` → 3 rules
    (`neural_blackhole_perception` with `model_id =
    neural_spacetime_blackhole`, `lensing_emphasis` with
    `emphasis = 2.0`, `spectral_shift_blackhole`).
- `CosmicRuntime.reality_rule_engine` (`None` by default —
  scientific behavior). When set, `render_for_observer` builds
  a fresh `RuleContext` from the observer + truth-level counts,
  runs the engine, uses the resulting `warp_factor` as the
  *effective* warp for the perception transform, and attaches
  `rule_context.active_rule_ids` / `rule_context.metadata` to
  the produced `RealityView`. The observer's stored
  `warp_factor` is **never** mutated (verified by
  `test_runtime_render_does_not_mutate_observer_warp` and the
  demo's tamper check).
- `RealityView` gained `active_rule_ids: list[str]` and
  `reality_metadata: dict` (top-level fields, surfaced in
  `to_dict()` / `summary()`).
- `RuntimeServer.broadcast_observer_view` extends the wire
  protocol with `active_rule_ids` and `reality_metadata`.
- `AIViewer` displays the effective post-rules `warp_factor`
  plus the active rules and the `mode` derived from
  `reality_metadata` on every `reality_view` message.
- Demo at `examples/emergent_reality_demo.py` renders the same
  observer (5 ly out, β = 0.05) under all three presets and
  reports active rules + effective warp + reality metadata per
  preset, then runs a tamper check verifying registry size,
  ids hash, truth-level counts, and the observer's stored
  `warp_factor` are unchanged after every render. Mean abs
  pixel delta vs scientific: hypertravel ≈ `0.085`, blackhole
  ≈ `0.0` (symbolic-only — no in-pipeline pixel effect yet).
- 25 new tests in `tests/test_reality_rules.py` cover
  RuleContext clone independence + dict round-trip, every
  built-in rule's apply behavior + parameter validation,
  engine duplicate-id rejection / unknown removal / priority
  ordering / enabled-only listing / non-mutation, all three
  presets (id sets / metadata propagation), and runtime
  integration (no rules → empty active_rule_ids, hypertravel
  preset attaches expected rules and effective_warp_factor,
  observer.warp_factor never mutated across multiple renders,
  to_dict surfaces the new fields, sample-data render leaves
  registry size + truth counts unchanged).

## Phase 36: Provenance and truth integrity

- New package `cosmic_engine.provenance` adds a lightweight,
  in-memory audit layer that tracks **how** every rendered output
  was produced. No external dependencies, no heavyweight logging
  — every storage primitive is a plain dict.
  - `ProvenanceRecord(entity_id, source, truth_level,
    transformations, timestamp, observer_id)`. Methods:
    `add_transformation(name)` (rejects empty), `to_dict()`.
  - `TruthTracker.register_entity(entity_id, source, truth_level,
    timestamp=0.0, observer_id=None)` is **idempotent**:
    re-registering an existing id never overwrites its
    transformation history (so callers can fire register-on-load
    safely). `add_transformation(entity_id, transformation)`
    returns a bool so hot paths can fire-and-forget without
    try/except. `add_transformation_to_all(name)` stamps every
    record at once for whole-pipeline steps. `list_records()`
    is sorted by `entity_id` for determinism.
  - `audit_reality_view(view) -> dict` returns
    `{observer_id, truth_distribution, transformations,
    active_rules, warnings}`. `transformations` deduplicates
    pipeline labels (e.g. `"perception_warp"`, `"ai_warp"`,
    `"physics_<backend>"`) followed by reality rules emitted as
    `"reality_rule:<id>"` so the whole audit grep is one prefix.
  - `detect_truth_mixing(view) -> list[str]` flags three
    deterministic conditions: (1) AI step touched observed-class
    data without a `use_neural_perception` marker;
    (2) symbolic rule active without a `color_mapping` marker;
    (3) `causality_mode == "relaxed"` while
    `visible_event_count > 0`.
- `CosmicRuntime.truth_tracker` (always present, empty by
  default). `add_objects` registers each new
  `UniverseObject`'s `id` / `source` / `truth_level.value`.
  Physics-step labels every massive object with
  `"physics_<backend>"`. `render_for_observer` records
  `"perception_warp"` on every rendered photon's source object,
  `"ai_warp"` when an `AIWarpModel` is attached and didn't
  fall back, and `"reality_rule:<id>"` for each active rule.
- `RealityView` gained two top-level fields:
  `provenance_summary: dict` (transformations, source_counts,
  truth_level_counts, tracked_entities, observed_entities,
  total_records) and `audit_warnings: list[str]`.
  `runtime.render_for_observer` runs `detect_truth_mixing`
  *after* the view is fully assembled and stashes the result
  on the view itself. `to_dict()` and `summary()` surface
  both fields.
- `RuntimeServer.broadcast_observer_view` extends the
  `reality_view` wire payload with `provenance_summary` and
  `audit_warnings`.
- `AIViewer` prints the audit-warning count on every
  reality_view message; `AIViewerConfig.verbose_audit = True`
  switches on per-message dumps of truth_level_counts +
  transformations + each warning text.
- Demo at `examples/provenance_audit_demo.py` loads the
  bundled mixed dataset (`gaia` + `sdss` + `desi` + `jpl` =
  `catalog_imported` / `ephemeris_real`, `physics_simulated`,
  plus 200 `procedural_approximation` synthetic galaxies) and
  attaches an identity AI warp model so every render exercises
  the AI-mixing path. Renders the same observer under all 3
  reality presets and prints the audit:
  - **scientific**: 1 audit warning ("observed data passed
    through an AI transformation without use_neural_perception
    declared") — exactly what the audit is for.
  - **hypertravel** / **blackhole**: 0 warnings, because the
    presets explicitly set `use_neural_perception = True` in
    `reality_metadata`. The pipeline transformation list grows
    to include each `reality_rule:<id>`.
  - 239 provenance records total; ~11 records carry pipeline
    transformations (only entities the renderer actually
    touched).
- 23 new tests in `tests/test_provenance.py` cover record
  to_dict / append-order / empty-transformation rejection,
  tracker register/get/idempotence/empty-id rejection/list
  ordering/per-id and global transformation propagation,
  audit return-keys / pipeline+rules merging, every
  `detect_truth_mixing` condition (AI-on-observed, symbolic
  without marker, relaxed-causality with events, clean view),
  and runtime integration (tracker present by default,
  add_objects registers, render attaches a populated
  provenance_summary, AI warp records `"ai_warp"` and
  triggers the warning, reality rules record their labels,
  empty-registry render doesn't crash, to_dict surfaces
  both new fields).

## Phase 37: Self-improving reality

- New package `cosmic_engine.adaptive` adds an opt-in,
  deterministic, fully-auditable feedback layer that measures
  discrepancies between analytical references and neural / AI
  approximations, **suggests** model or rule updates, but never
  mutates anything itself.
  - `FeedbackRecord(id, observer_id, timestamp_t, context,
    metric_name, metric_value, expected_value, deviation,
    source, notes)`. Validates non-empty id / metric_name /
    source and non-negative deviation; `to_dict()`.
  - `compute_acceleration_error(analytical, predicted)` →
    relative-L2 vector error. `compute_direction_error(a, p)`
    → `1 − cos(θ)` ∈ `[0, 2]` with degenerate inputs returning
    the maximum (no NaNs). `compute_brightness_error(a, p)`
    → relative scalar error. All three floor the denominator
    at `1e-30` so a zero reference doesn't divide by zero.
  - `AdaptivePolicy(error_threshold=0.10, window_size=32,
    sustained_fraction=0.5, max_updates_per_run=1)` exposes
    every threshold as a constructor parameter (rejects
    invalid values). `should_record` records anything above
    threshold. `should_update_model / should_update_rules`
    require the *full* most-recent window for that source AND
    `sustained_fraction` of it over threshold (so a few noisy
    spikes can't trigger drift suggestions).
  - `AdaptiveEngine(policy=None)`: `record_feedback(record)`
    returns a bool (kept or filtered out). `evaluate()`
    returns at most `policy.max_updates_per_run` suggestions
    *deterministically* — model first, rules second — each
    citing the exact `feedback_ids` that motivated it plus
    `mean_deviation` / `window_size`. `reset_run()` clears
    the per-run counter when the caller wants a fresh
    suggestion budget. `summary()` reports
    `record_count / by_source / max_deviation /
    mean_deviation`. Log capacity is bounded at `4 ×
    window_size` (FIFO) so long runs don't grow without
    bound.
- `CosmicRuntime.adaptive_engine` (`None` by default, opt-in).
  When set, `render_for_observer` probes the observer's
  attached models:
  - For `spacetime_model`: queries
    `query_acceleration(observer_position + probe_radius·x̂)`
    against analytical Schwarzschild for a configurable
    reference mass (`observer.config['adaptive_reference_mass_kg']`,
    default 1 M☉; `adaptive_probe_radius_m` default 1e9 m).
  - For `ai_warp_model`: queries `predict_direction(observer.forward)`
    against the deterministic `apply_direction_warp`.

  Both produce `FeedbackRecord`s, hand them to
  `adaptive_engine.record_feedback`, and call `evaluate()` so
  any matured suggestions land on the produced view.
- `RealityView` gained two top-level fields:
  `feedback_summary: dict` (record_count, by_source,
  max_deviation, mean_deviation) and
  `adaptive_suggestions: list[dict]`. Both surface in
  `to_dict()` / `summary()` and on the
  `RuntimeServer.broadcast_observer_view` wire payload.
- `AIViewerConfig.show_feedback = False` toggles a verbose
  per-message dump of the feedback summary plus every
  suggestion's action + reason; the always-on print adds
  `suggestions=N` to the standard line.
- Demo at `examples/adaptive_engine_demo.py` builds a
  `_DriftingSpacetime` model that returns analytical
  Schwarzschild scaled by 1.30 (a flat 30 % relative-L2
  deviation), attaches it to an observer with a tight
  `AdaptivePolicy(threshold=0.10, window_size=5,
  sustained_fraction=0.6, max_updates_per_run=4)`, and renders
  8 frames. Every probe records `deviation = 0.3000`; once the
  window fills (step 4) every subsequent frame emits the same
  `retrain_spacetime_model` suggestion citing the 5 feedback
  ids that motivated it. The no-mutation check confirms the
  observer's `spacetime_model` is the original instance,
  `bias = 1.30` is unchanged, `runtime.reality_rule_engine` is
  still `None`, and `observer.warp_factor = 1.0`.
- 31 new tests in `tests/test_adaptive_engine.py` cover all
  three metrics (zero-on-identical / relative-L2 /
  shape-mismatch / orthogonal / antiparallel / degenerate /
  zero-truth-with-eps), `FeedbackRecord` validation
  (non-negative deviation, non-empty id/metric/source,
  to_dict round-trip), `AdaptivePolicy` (threshold filter,
  invalid-input rejection, full-window requirement,
  sustained-fraction logic, source filtering),
  `AdaptiveEngine` (below-threshold drop, no suggestion under
  window, model retrain on sustained drift, max_updates_per_run
  budget, reset_run), and runtime integration (no engine →
  empty fields, drifted spacetime model emits retrain
  suggestion, no-mutation invariant across multiple frames,
  AI warp drift records ai_warp feedback, to_dict surfaces
  the new fields).
