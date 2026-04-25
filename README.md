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
  intergalactic (`1e25 m`) to microscale (`1e3 m`); the runtime swaps
  representations (galaxy field / star field / N-body / density field)
  automatically and blends across boundaries.
- **Multi-observer reality.** Different observers see the same
  universe differently — pose, velocity, warp factor, optional
  per-observer spacetime / AI models. Each gets its own subjective
  time, light-cone-filtered events, and rule-shaped reality.
- **Audit + provenance built in.** Every render carries a
  `provenance_summary`, `audit_warnings`, and an `adaptive_suggestions`
  list — discrepancies between analytical references and neural
  approximations are detected, but **never** auto-applied.
- **Cloud-ready.** Optional `[distributed]` extra adds Ray; per-role
  Dockerfiles + start scripts ship under
  [`deploy/lambda/`](deploy/lambda/) (Lambda Cloud) and
  [`deploy/runpod/`](deploy/runpod/) (RunPod).

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
├── tests/                  # 902 pytest tests
├── scripts/                # helper scripts (model builders)
├── deploy/lambda/          # Lambda Cloud deployment scaffold
├── deploy/runpod/          # RunPod deployment scaffold
└── docs/                   # documentation (you are here)
```

---

## Installation

### Requirements

| | |
|---|---|
| **Python** | 3.11 or newer |
| **OS** | Linux, macOS, Windows (WSL recommended) |
| **GPU** | Optional. CUDA 12.x for full acceleration; everything works on CPU. |
| **Disk** | ≈ 1 GB for the engine + bundled samples; trained models add a few hundred MB. |

### Local install

```bash
# 1. Clone
git clone https://github.com/<your-org>/cosmicengine.git
cd cosmicengine

# 2. (Recommended) virtual environment
python -m venv .venv
source .venv/bin/activate     # Windows: .venv\Scripts\activate

# 3. Install the engine + AI Viewer in editable mode
pip install -e .
```

### Optional extras

```bash
# Add Ray for multi-node / cloud deployments
pip install -e .[distributed]
```

### Verify the install

```bash
python -m pytest tests/
# 902 passed
```

If pytest passes, the engine, AI Viewer, data ingestion, ONNX runtime,
and the deployment scaffolding are all wired correctly.

---

## Running locally

### Render a starfield in 10 lines of Python

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
# observer=cam rep=flat objects=39 warp=1.0 spacetime=none ...
```

### Run the bundled demos

Every major feature ships with a self-contained demo in `examples/`.
Each writes its frames into `outputs/viewer/` and prints a summary:

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

The full list (40+ demos grouped by layer) is in
[`docs/DEMOS.md`](docs/DEMOS.md).

### Run the live runtime + viewer

In one terminal:

```bash
python examples/runtime_server_demo.py
# [server] listening on 127.0.0.1:8765
```

In another:

```bash
python examples/runtime_client_demo.py
# scene_state messages stream to stdout
```

The AI Viewer (with optional Tk window) consumes the same protocol:

```bash
python examples/ai_viewer_window_demo.py
```

---

## Distributed deployment

CosmicEngine ships **two parallel deployment scaffolds**. Both consume
the same cloud-agnostic `cosmic_engine.distributed` package; pick the
one that matches your provider:

| Provider | Files | Best for |
|---|---|---|
| **Lambda Cloud** | [`deploy/lambda/`](deploy/lambda/) | Bare GPU VMs, mounted filesystems, `ssh` workflows. |
| **RunPod** | [`deploy/runpod/`](deploy/runpod/) | Container pods, network volumes, `runpodctl` automation. |

Ray is **optional** in both paths — single-node and CPU-only modes
work without it. See [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md) for
the full operator reference and `COSMIC_*` environment-variable
table.

### Local emulation (works for both)

```bash
# Lambda topology
docker compose -f deploy/lambda/docker-compose.lambda.yml up --build

# RunPod topology
docker compose -f deploy/runpod/docker-compose.runpod.yml up --build
```

Each launches `ray-head`, `worker-ai`, `worker-render`, and a
`viewer`. GPU support is opt-in (uncomment the
`deploy.resources.reservations.devices` block on a service when the
NVIDIA Container Toolkit is installed).

---

## Lambda Cloud deployment

Full operator guide: [`deploy/lambda/README.md`](deploy/lambda/README.md).

### 1. Provision a Lambda instance

Pick an instance type — `gpu_1x_a10` is plenty for runtime + 1 worker
on the same box. Mount your shared filesystem at `/workspace`.

### 2. SSH in and install

```bash
ssh ubuntu@<INSTANCE_IP>
git clone https://github.com/<your-org>/cosmicengine.git /workspace/cosmicengine
cd /workspace/cosmicengine
pip install -e .[distributed]
```

### 3. Single-node deployment

```bash
# Start a local Ray head
ray start --head --port=6379 --dashboard-host=0.0.0.0

# Source the role-specific env file and start the runtime
set -a; . deploy/lambda/config/runtime.env.example; set +a
bash deploy/lambda/scripts/start_runtime.sh
# [runtime] listening on 0.0.0.0:8765
```

The runtime now serves on `0.0.0.0:8765`. Connect a viewer machine
with:

```bash
COSMIC_HEAD_IP=<INSTANCE_IP> COSMIC_RUNTIME_PORT=8765 \
    bash deploy/lambda/scripts/start_viewer.sh
```

### 4. Multi-node deployment

**Head node:**

```bash
ray start --head --port=6379 --dashboard-host=0.0.0.0 --dashboard-port=8265
bash deploy/lambda/scripts/start_runtime.sh
```

**Each worker** (replace `<HEAD_IP>` with the head's reachable IP):

```bash
ssh ubuntu@<WORKER_IP>
cd /workspace/cosmicengine && pip install -e .[distributed]

export COSMIC_HEAD_IP=<HEAD_IP>
export COSMIC_RAY_ADDRESS=<HEAD_IP>:6379
ray start --address=<HEAD_IP>:6379
bash deploy/lambda/scripts/start_worker.sh
```

**Optional training node:**

```bash
ssh ubuntu@<TRAINING_IP>
cd /workspace/cosmicengine && pip install -e .[distributed]
ray start --address=<HEAD_IP>:6379
bash deploy/lambda/scripts/start_training.sh
# Then SSH in separately to actually run a training script:
#   python examples/train_spacetime_model.py
```

### 5. Required ports (security group)

| Port | Service | Inbound on |
|---|---|---|
| 8765 | CosmicRuntime API | head / runtime node |
| 6379 | Ray GCS | head node |
| 10001 | Ray client | head node |
| 8265 | Ray dashboard (operator only) | head node |

### 6. Health check

On any node:

```bash
bash deploy/lambda/scripts/healthcheck.sh | jq .
# JSON report: hostname, CUDA availability, Ray status,
# data/model/output dir writability, resolved DistributedConfig.
```

---

## RunPod deployment

Full operator guide: [`deploy/runpod/README.md`](deploy/runpod/README.md).

### 1. Build and push images

```bash
# Build all four role images from the repo root.
docker build -f deploy/runpod/docker/Dockerfile.runtime  -t myorg/cosmic-engine-runtime:runpod  .
docker build -f deploy/runpod/docker/Dockerfile.worker   -t myorg/cosmic-engine-worker:runpod   .
docker build -f deploy/runpod/docker/Dockerfile.viewer   -t myorg/cosmic-engine-viewer:runpod   .
docker build -f deploy/runpod/docker/Dockerfile.training -t myorg/cosmic-engine-training:runpod .

# Push to a registry RunPod can pull from.
docker push myorg/cosmic-engine-runtime:runpod
docker push myorg/cosmic-engine-worker:runpod
docker push myorg/cosmic-engine-viewer:runpod
docker push myorg/cosmic-engine-training:runpod
```

### 2. Create a RunPod network volume

In the RunPod dashboard, create a volume in your target data center
(e.g. `cosmic-engine-shared`, 100 GB). Attach it to every pod at
`/workspace`.

### 3. Single-node deployment (Web Dashboard)

1. **Templates** → **New Template** → image
   `myorg/cosmic-engine-runtime:runpod`.
2. **Volume Mount Path**: `/workspace`.
3. **Expose Ports**: `8765/tcp`, `6379/tcp`, `8265/http`, `10001/tcp`.
4. **Environment Variables** — copy from
   [`deploy/runpod/config/runtime.env.example`](deploy/runpod/config/runtime.env.example):

   ```
   COSMIC_ROLE=runtime
   COSMIC_RAY_ADDRESS=auto
   COSMIC_RUNTIME_PORT=8765
   COSMIC_DATA_DIR=/workspace/data
   COSMIC_MODEL_DIR=/workspace/models
   COSMIC_OUTPUT_DIR=/workspace/outputs
   COSMIC_ENABLE_GPU=true
   ```
5. **Deploy** — pick a GPU type (e.g. `RTX A4000`).

The pod's container `CMD` runs `start_runtime.sh` automatically.
External clients reach it via the proxy URL on the pod's
**Connect** tab (TCP mapping for `8765`).

### 3'. Single-node deployment (`runpodctl`)

```bash
runpodctl create pod \
    --name cosmic-runtime \
    --imageName myorg/cosmic-engine-runtime:runpod \
    --gpuType "RTX A4000" \
    --containerDiskInGb 20 \
    --volumeInGb 100 \
    --volumeMountPath /workspace \
    --ports "8765/tcp,6379/tcp,8265/http,10001/tcp" \
    --env COSMIC_ROLE=runtime \
    --env COSMIC_RAY_ADDRESS=auto \
    --env COSMIC_RUNTIME_PORT=8765 \
    --env COSMIC_DATA_DIR=/workspace/data \
    --env COSMIC_MODEL_DIR=/workspace/models \
    --env COSMIC_OUTPUT_DIR=/workspace/outputs
```

### 4. Multi-node deployment

**Step A — start the head pod** as above.

**Step B — note the head host:port** the workers will use.

- Same project's pods → use the head pod's RunPod-internal hostname
  (visible on the dashboard's networking tab) and port `6379`.
- Cross-region or external → set up a TCP tunnel (Tailscale / SSH
  `-L`); RunPod's HTTP proxy can't carry Ray's GCS traffic.

**Step C — create one or more worker pods** (replace `<HEAD_HOST>`):

```bash
runpodctl create pod \
    --name cosmic-worker-ai \
    --imageName myorg/cosmic-engine-worker:runpod \
    --gpuType "RTX A4000" \
    --containerDiskInGb 20 \
    --volumeInGb 100 \
    --volumeMountPath /workspace \
    --env COSMIC_ROLE=worker \
    --env COSMIC_RAY_ADDRESS=<HEAD_HOST>:6379 \
    --env COSMIC_HEAD_IP=<HEAD_HOST> \
    --env COSMIC_DATA_DIR=/workspace/data \
    --env COSMIC_MODEL_DIR=/workspace/models \
    --env COSMIC_OUTPUT_DIR=/workspace/outputs
```

Each worker's container `CMD` connects to the head and idles as a
Ray worker.

### 5. Connect a viewer pod

```bash
runpodctl create pod \
    --name cosmic-viewer \
    --imageName myorg/cosmic-engine-viewer:runpod \
    --volumeInGb 50 \
    --volumeMountPath /workspace \
    --env COSMIC_ROLE=viewer \
    --env COSMIC_HEAD_IP=<HEAD_HOST> \
    --env COSMIC_RUNTIME_PORT=8765 \
    --env COSMIC_OUTPUT_DIR=/workspace/outputs
```

### 6. Health check

```bash
runpodctl exec <POD_ID> -- bash deploy/runpod/scripts/healthcheck.sh
```

Same JSON report shape as Lambda — convenient for verifying CUDA,
Ray, and storage from outside the pod.

---

## Documentation

- **[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)** — module map,
  data flow, package responsibilities.
- **[`docs/MODULES.md`](docs/MODULES.md)** — per-package reference
  for every sub-package: `core`, `data`, `physics`, `cosmos`,
  `perception`, `rendering`, `ai`, `streaming`, `multiscale`,
  `observer`, `time`, `reality`, `provenance`, `adaptive`,
  `synthesis`, `runtime`, `distributed`.
- **[`docs/DEMOS.md`](docs/DEMOS.md)** — every example with a
  one-line description and the modules it exercises.
- **[`docs/PROVENANCE.md`](docs/PROVENANCE.md)** — truth levels,
  audit warnings, how to interpret a `RealityView`.
- **[`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md)** — local dev,
  Docker, Lambda Cloud, and RunPod side-by-side reference.
- **[`docs/DEVELOPMENT_LOG.md`](docs/DEVELOPMENT_LOG.md)** — phase-
  by-phase changelog (≈40 phases, the original incremental
  README).
- **[`docs/ROADMAP.md`](docs/ROADMAP.md)** /
  **[`docs/TASKS.md`](docs/TASKS.md)** — planning notes.

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
