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
