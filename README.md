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
  / `load_jpl_into_registry` returns a frozen Sun / Earth / Mars
  snapshot tagged `truth_level=EPHEMERIS_REAL`. Not a real SPICE
  ingest.
- Real APIs / FITS / SPICE kernels are **not** wired up; ingestion
  is offline CSV today.
- Sample data: `data/sample_gaia_like.csv`, `data/sample_sdss_like.csv`,
  `data/sample_desi.csv`. Demo at `examples/multi_catalog_demo.py`
  combines all four sources and writes `output_multi_catalog.ppm`.
