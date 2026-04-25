# CosmicEngine

A data-driven, physics-grounded, AI-assisted cosmic perception engine.
CosmicEngine ingests real astronomical catalogs, simulates physics
deterministically, lets neural models *augment* perception (never
replace truth), and renders the result through a photon / Gaussian-splat
pipeline that supports multiple observers, subjective time, and
auditable reality rules.

> **Core rule:** AI may drive rendering and perception, but it never
> replaces truth metadata, coordinates, physics constraints, or source
> provenance. Every generated object carries `truth_level` and
> `source`; every transformation is recorded.

---

## Highlights

- **Real-data first.** Gaia / SDSS / DESI / JPL ingestion with explicit
  `TruthLevel` tagging on every object.
- **Deterministic physics.** Newtonian + Barnes-Hut N-body, ΛCDM
  cosmology, Schwarzschild geodesic ray marching for GR effects.
- **AI-augmented, not AI-replaced.** ONNX neural photon warp, density
  reconstruction, neural spacetime field — each produces a
  `RealityView` that's *layered on top of* deterministic state.
- **Multi-scale.** One observer can navigate continuously from
  intergalactic (`1e25 m`) to microscale (`1e3 m`); the runtime
  swaps representations (galaxy field / star field / N-body /
  density field) automatically and blends across boundaries.
- **Multi-observer reality.** Different observers see the same
  universe differently — pose, velocity, warp factor, optional
  per-observer spacetime / AI models. Each gets its own subjective
  time, light-cone-filtered events, and rule-shaped reality.
- **Audit + provenance built in.** Every render carries a
  `provenance_summary`, `audit_warnings`, and an `adaptive_suggestions`
  list — discrepancies between analytical references and neural
  approximations are detected, but **never** auto-applied.
- **Lambda-Cloud-ready.** Optional `[distributed]` extra adds Ray
  workers; per-role Dockerfiles + start scripts ship under
  `deploy/lambda/`.

## Architecture at a glance

```
                 ┌──────────────────────────────────────────────┐
                 │              CosmicRuntime                   │
                 │   registry · clock · physics · observers     │
                 │   provenance · adaptive · reality rules      │
                 └───┬─────────────────────────────────────┬────┘
                     │                                     │
       ┌─────────────▼─────────────┐         ┌─────────────▼─────────────┐
       │       data_pipeline       │         │       rendering           │
       │  Gaia · SDSS · DESI · JPL │         │  photon · Gaussian · GPU  │
       │  synthetic · synthesis    │         │  geodesic · neural warp   │
       └───────────────────────────┘         └───────────────────────────┘
                     │                                     │
       ┌─────────────▼─────────────┐         ┌─────────────▼─────────────┐
       │           ai              │         │       AI Viewer           │
       │  ONNX warp · density      │         │  PPM · GUI · post-process │
       │  neural spacetime · train │         │  reality_view consumer    │
       └───────────────────────────┘         └───────────────────────────┘
```

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the full module
map and data flow.

## Repository layout

```
cosmicengine/
├── engine/cosmic_engine/   # core engine (≈20 sub-packages, see docs/MODULES.md)
├── apps/ai_viewer/         # AI Viewer client + neural-field renderer
├── data/                   # bundled catalog samples (Gaia / SDSS / DESI / JPL)
├── models/                 # ONNX checkpoints (mock + trained)
├── examples/               # 40+ runnable demos
├── tests/                  # 880 pytest tests
├── scripts/                # helper scripts (model builders)
├── deploy/lambda/          # Lambda Cloud deployment scaffold
└── docs/                   # documentation (you are here)
```

## Quick start

### Install

```bash
git clone <this-repo> cosmicengine
cd cosmicengine
pip install -e .
```

Optional extras:

```bash
pip install -e .[distributed]   # adds Ray for multi-node deployment
```

### Render a starfield in 10 lines

```python
from cosmic_engine.runtime import CosmicRuntime, RuntimeConfig
from cosmic_engine.core.vector import Vector3
from cosmic_engine.observer import Observer
from cosmic_engine.rendering import SimpleCamera

runtime = CosmicRuntime(config=RuntimeConfig(active_radius_m=1.0e30))
runtime.load_sample_data()  # Gaia + SDSS + DESI + JPL samples
runtime.observer_manager.add_observer(Observer(
    id="cam",
    position_m=Vector3(0.0, -5.0e17, 0.0),
    velocity_m_s=Vector3.zero(),
    forward=Vector3(0.0, 1.0, 0.0),
    up=Vector3(0.0, 0.0, 1.0),
    config={"output_ppm_path": "outputs/quick.ppm"},
))
view = runtime.render_for_observer("cam", SimpleCamera(
    position_m=Vector3.zero(), forward=Vector3(0.0, 1.0, 0.0),
    up=Vector3(0.0, 0.0, 1.0), fov_degrees=120.0,
    image_width=256, image_height=256,
))
print(view.summary())
```

### Run the demos

Every major feature ships with a self-contained demo in `examples/`:

```bash
python examples/load_star_catalog_demo.py        # ingest a CSV
python examples/render_starfield_demo.py         # CPU render a frame
python examples/multiscale_navigation_demo.py    # walk across scales
python examples/multi_observer_demo.py           # 3 observers, same universe
python examples/causality_demo.py                # subjective time + light cone
python examples/emergent_reality_demo.py         # reality presets
python examples/provenance_audit_demo.py         # truth-mixing audit
python examples/adaptive_engine_demo.py          # drift detection
python examples/reality_synthesis_demo.py        # generate universes
```

The full list is in [`docs/DEMOS.md`](docs/DEMOS.md).

### Run the test suite

```bash
python -m pytest tests/
# 880 passed
```

## Documentation

- **[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)** — module map,
  data flow, package responsibilities.
- **[`docs/MODULES.md`](docs/MODULES.md)** — per-package reference:
  `core`, `data`, `physics`, `cosmos`, `perception`, `rendering`,
  `ai`, `streaming`, `multiscale`, `observer`, `time`, `reality`,
  `provenance`, `adaptive`, `synthesis`, `runtime`, `distributed`.
- **[`docs/DEMOS.md`](docs/DEMOS.md)** — every example with a
  one-line description and the modules it exercises.
- **[`docs/PROVENANCE.md`](docs/PROVENANCE.md)** — truth levels,
  audit warnings, how to interpret a `RealityView`.
- **[`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md)** — local dev,
  Docker, and Lambda Cloud (links to `deploy/lambda/README.md`).
- **[`docs/DEVELOPMENT_LOG.md`](docs/DEVELOPMENT_LOG.md)** — phase-by-
  phase changelog (≈40 phases, the original incremental README).
- **[`docs/ROADMAP.md`](docs/ROADMAP.md)** /
  **[`docs/TASKS.md`](docs/TASKS.md)** — planning notes.

## Lambda Cloud deployment (TL;DR)

Single node:

```bash
ssh ubuntu@<INSTANCE_IP>
cd /workspace/cosmicengine
pip install -e .[distributed]
ray start --head --port=6379 --dashboard-host=0.0.0.0
bash deploy/lambda/scripts/start_runtime.sh
```

Multi-node + viewer + training: see
[`deploy/lambda/README.md`](deploy/lambda/README.md) for the operator
guide.

## Contributing

CosmicEngine is built incrementally — every phase ships small,
testable units. The development conventions are encoded in
[`CLAUDE.md`](CLAUDE.md):

- Small modules. Tests for every subsystem.
- No fake scientific certainty. Every generated object must carry
  `truth_level` and `source`.
- Approximations stay explicitly marked.
- Prefer simple working prototypes over giant abstractions.
- Never rewrite the whole repo without permission.

## License

Source-available; see [`LICENSE`](LICENSE) if present, or contact the
maintainers for redistribution terms.
