# Demo index

Every example in `examples/` is self-contained: `python
examples/<name>.py` from the repo root produces output (typically
PPM frames in `outputs/viewer/` and a printed summary). They're
roughly grouped by which layer they exercise.

## Data ingestion + basics

| Demo | What it does |
|---|---|
| `load_star_catalog_demo.py` | Read a CSV catalog into a `UniverseRegistry`; print truth-level breakdown. |
| `multi_catalog_demo.py` | Load Gaia + SDSS + DESI + JPL together. |
| `desi_like_demo.py` | DESI-style redshift survey ingestion. |
| `synthetic_galaxy_cloud_demo.py` | Generate a procedural galaxy field. |

## Rendering

| Demo | What it does |
|---|---|
| `render_starfield_demo.py` | CPU photon-field render to PPM. |
| `perception_warp_demo.py` | Aberration + beaming + Doppler tint. |
| `vectorized_warp_demo.py` | NumPy batched perception transform. |
| `gaussian_splat_demo.py` | CPU Gaussian splatting. |
| `gpu_splat_demo.py` / `webgpu_splat_demo.py` | WebGPU-backed splatter. |
| `relativistic_gaussian_field_demo.py` | Relativistic warp in field space. |

## Physics

| Demo | What it does |
|---|---|
| `nbody_solar_demo.py` | Leapfrog N-body of the solar system. |
| `barnes_hut_demo.py` | O(N log N) tree-coded N-body. |
| `solar_system_physics_demo.py` | Solar-system snapshot + step. |
| `lcdm_vs_hubble_demo.py` | ΛCDM cosmology vs. Hubble flow. |
| `black_hole_lensing_demo.py` | Schwarzschild lensing. |
| `geodesic_blackhole_demo.py` | Geodesic ray marching near a black hole. |

## AI

| Demo | What it does |
|---|---|
| `ai_warp_demo.py` | Identity AI warp model on photon samples. |
| `onnx_warp_demo.py` | ONNX photon-warp inference. |
| `neural_photon_warp_demo.py` / `batch_neural_photon_warp_demo.py` | Neural warp in two flavours. |
| `density_ai_demo.py` | ONNX density-field reconstruction. |
| `neural_spacetime_demo.py` | Neural spacetime field with analytical fallback. |
| `train_spacetime_model.py` | PyTorch training pipeline → ONNX. |
| `test_trained_spacetime_model.py` | Validate the trained ONNX model. |

## Runtime

| Demo | What it does |
|---|---|
| `runtime_demo.py` | Headless `CosmicRuntime` step loop. |
| `runtime_server_demo.py` | TCP server broadcasting scene state. |
| `runtime_client_demo.py` | TCP client reading the same. |
| `streaming_lod_demo.py` | Tile store + LOD selection. |

## Viewer client

| Demo | What it does |
|---|---|
| `ai_viewer_demo.py` | `AIViewer` reading from a server. |
| `ai_viewer_window_demo.py` | Same, with the optional Tk window. |
| `ai_viewer_neural_demo.py` | ONNX neural postprocess in the viewer. |

## Multi-scale / multi-observer

| Demo | What it does |
|---|---|
| `multiscale_navigation_demo.py` | Walk an observer across 4 scale zones. |
| `multi_observer_demo.py` | 3 observers (baseline, relativistic, neural) on the same dataset. |

## Subjective time + reality + audit

| Demo | What it does |
|---|---|
| `causality_demo.py` | Per-observer proper time + light-cone event filter. |
| `emergent_reality_demo.py` | Scientific / hypertravel / blackhole reality presets. |
| `provenance_audit_demo.py` | Truth-mixing audit warnings. |
| `adaptive_engine_demo.py` | Drift detection + retrain suggestion (no auto-apply). |
| `reality_synthesis_demo.py` | Generate 3 deterministic universes from specs. |

## Output convention

Every demo writes its frames into `outputs/viewer/` so a single
directory holds the full set after a "run them all" pass:

```bash
mkdir -p outputs/viewer
for f in examples/*demo.py; do python "$f" || true; done
ls outputs/viewer/
```
