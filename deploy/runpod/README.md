# CosmicEngine on RunPod

This directory contains everything needed to deploy CosmicEngine on
[RunPod](https://www.runpod.io/) as a distributed multi-GPU system.
The local single-node workflow is **unchanged** — these files only
add new ways to run.

If you've already read [`deploy/lambda/README.md`](../lambda/README.md):
the engine layer is identical (`cosmic_engine.distributed` is
cloud-agnostic). The differences here are RunPod-specific:
container-first, network volumes, the RunPod proxy URL convention,
and the `runpodctl` CLI for pod management.

## Overview

CosmicEngine runs as four distinct pod roles. You can put them all
on one pod or scale each role independently:

| Role | Image | Default port | What it does |
|---|---|---|---|
| `runtime` | `Dockerfile.runtime` | `8765` | Hosts the `CosmicRuntime` TCP server: scene state, registry, observers, reality views. |
| `worker` | `Dockerfile.worker` | – | Joins the Ray cluster and runs photon warps, density reconstruction, Gaussian splatting. |
| `viewer` | `Dockerfile.viewer` | – | Pulls frames from the runtime and post-processes them. |
| `training`| `Dockerfile.training` | – | Optional. Runs training scripts for the neural spacetime field. |

## Architecture

```
                   ┌─────────────────────────┐
                   │   Head pod              │
                   │   ─ Ray head            │
                   │   ─ CosmicRuntime API   │
                   │   ─ Job coordinator     │
                   └────────┬────────────────┘
                            │ 6379 / 8765
            ┌───────────────┼───────────────┐
            │               │               │
     ┌──────┴──────┐  ┌─────┴─────┐  ┌──────┴──────┐
     │ AI workers  │  │ Render    │  │ Training    │
     │ ─ ONNX warp │  │  workers  │  │  (optional) │
     │ ─ density   │  │ ─ Gaussian│  │  ─ PyTorch  │
     │ ─ neural st.│  │   splat   │  │  ─ ONNX out │
     └─────────────┘  └───────────┘  └─────────────┘
                            │
                     ┌──────┴──────┐
                     │ Viewer pods │
                     │ ─ AIViewer  │
                     └─────────────┘

  RunPod network volume mounted on every pod
   /workspace/data    /workspace/models    /workspace/outputs
```

## RunPod deployment model

RunPod gives you GPU pods (containers, not VMs). There's no managed
Ray autoscaler, but pods spin up in seconds and share storage
through **network volumes**. Two main paths:

1. **Web Dashboard** — create pods from a template, paste the
   environment variables, attach the network volume.
2. **`runpodctl` / GraphQL API** — script the same thing from a
   shell or CI.

Either way, the per-pod artefacts in this directory stay the same.

### Recommended pod templates

| Role | Recommended GPU | Why |
|---|---|---|
| Runtime / head | `RTX A4000` or CPU pod | Modest CPU, no big GPU. |
| AI worker | `RTX A4000` / `RTX A6000` | ONNX inference + neural warps. |
| Render worker | `RTX A6000` / `L40S` | WebGPU / Gaussian splatting fits in one consumer GPU. |
| Training | `A100 80GB` | The spacetime model is small but training is faster on A100. |
| Viewer | CPU pod | Just receives frames + writes PPMs. |

### Pod template setup (Web Dashboard)

For each role:

1. **Templates** → **New Template**.
2. **Container Image**: push your built image (e.g.
   `myorg/cosmic-engine-runtime:runpod`) to a registry RunPod can
   pull from.
3. **Container Disk** / **Volume Disk**: configure the network
   volume to mount at `/workspace`.
4. **Environment Variables**: paste from
   [`config/runtime.env.example`](config/runtime.env.example) /
   [`config/worker.env.example`](config/worker.env.example).
5. **Expose HTTP/TCP Ports**: `8765` for the runtime template;
   `6379`, `8265`, `10001` if the same pod will run the Ray head.
6. **Save**.

### Pod template setup (runpodctl)

```bash
# Create a runtime pod (Ray head + CosmicRuntime).
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

# Create one or more worker pods (replace HEAD_HOST below).
runpodctl create pod \
    --name cosmic-worker-ai \
    --imageName myorg/cosmic-engine-worker:runpod \
    --gpuType "RTX A4000" \
    --containerDiskInGb 20 \
    --volumeInGb 100 \
    --volumeMountPath /workspace \
    --env COSMIC_ROLE=worker \
    --env COSMIC_RAY_ADDRESS=<HEAD_HOST>:6379 \
    --env COSMIC_HEAD_IP=<HEAD_HOST>
```

`<HEAD_HOST>` is whichever address the worker can use to reach
the head pod's `6379` port (see "Networking" below).

## Single-node deployment

Quickest path to a running cluster — useful for smoke tests.

1. Create one runtime pod from `Dockerfile.runtime`.
2. SSH in (RunPod gives you a per-pod SSH command on the dashboard).
3. The container's `CMD` already starts the runtime; if you SSH'd in
   manually you can re-run:

```bash
ray start --head --port=6379 --dashboard-host=0.0.0.0
bash deploy/runpod/scripts/start_runtime.sh
```

The runtime now listens on `0.0.0.0:8765` inside the pod. Outside
clients connect to it via the proxy URL printed on the pod's
**Connect** tab (something like
`https://<POD_ID>-8765.proxy.runpod.net`, but for raw TCP use the
TCP port mapping from the same tab).

## Multi-node Ray deployment

Pick one pod for the head, any number of pods for workers. Every
pod must mount the same network volume at `/workspace`.

### 1. Start the head

```bash
# Pod template env vars (already set by config/runtime.env.example):
#   COSMIC_ROLE=runtime
#   COSMIC_RAY_ADDRESS=auto
#   COSMIC_RUNTIME_PORT=8765
ray start --head --port=6379 --dashboard-host=0.0.0.0 --dashboard-port=8265
bash deploy/runpod/scripts/start_runtime.sh
```

Note the `<HEAD_HOST>` to use from worker pods. Two options:

- **Same project's pods**: use the head pod's internal hostname (on
  the dashboard's networking tab) and port `6379`.
- **Cross-project / external**: expose `6379/tcp` and use the
  proxy host:port. For Ray multi-node specifically, RunPod's
  HTTP proxy doesn't carry GCS traffic — you'll need either a
  TCP port-forward (via SSH `-L`) or a private network like
  Tailscale across the pods.

### 2. Attach worker pods

For each worker pod (template env vars from
[`config/worker.env.example`](config/worker.env.example)):

```bash
# COSMIC_HEAD_IP and COSMIC_RAY_ADDRESS are set on the pod template.
ray start --address=${COSMIC_RAY_ADDRESS}
bash deploy/runpod/scripts/start_worker.sh
```

### 3. Start a training pod (optional)

```bash
ray start --address=${COSMIC_RAY_ADDRESS}
bash deploy/runpod/scripts/start_training.sh
# Then SSH into the pod and run e.g.
#   python examples/train_spacetime_model.py
```

## Persistent volumes

RunPod **network volumes** are the cleanest way to share state
between pods. Create one volume in the same data center as your
pods, attach it at `/workspace`, and use the standard sub-paths:

| Path | Contents |
|---|---|
| `/workspace/data` | Catalog CSVs, JPL ephemeris snapshots, per-tile streaming data. |
| `/workspace/models` | ONNX / PyTorch checkpoints. |
| `/workspace/outputs` | Rendered PPMs, viewer captures, training artefacts. |

> **Tip**: for cross-region sharing, run a periodic `rsync` from
> the head pod's `/workspace` to an S3-compatible bucket.
> RunPod's network volumes are region-bound.

## Networking

| Direction | Port | Service | Notes |
|---|---|---|---|
| inbound on head pod | 6379 | Ray GCS | Workers connect here. Use TCP port mapping. |
| inbound on head pod | 8265 | Ray dashboard | Operator only. Use HTTP proxy. |
| inbound on head pod | 10001 | Ray client | Workers connect here for the client API. |
| inbound on head pod | 8765 | CosmicRuntime API | Viewer connects here via TCP proxy. |

Workers don't need any inbound ports open; they reach back to the
head over the same connection.

## Local docker-compose

For smoke-testing the topology without spinning up RunPod pods,
use [`docker-compose.runpod.yml`](docker-compose.runpod.yml):

```bash
cd deploy/runpod
docker compose -f docker-compose.runpod.yml up --build
```

The compose file launches `ray-head`, `worker-ai`, `worker-render`,
and a `viewer`. GPU support is opt-in (uncomment the
`deploy.resources.reservations.devices` block on a service when the
NVIDIA Container Toolkit is installed).

## Health checks

Every image bakes in a `HEALTHCHECK` that runs:

```bash
python -m cosmic_engine.distributed.entrypoints health
```

The command emits a JSON report on stdout summarising:

- Hostname, Python version, platform.
- `DistributedConfig` summary (role, ports, paths).
- CUDA availability (`available`, `device_count`, `device_names`).
- Ray availability (`available`, `initialized`, `cluster_resources`).
- Whether `data_dir` / `model_dir` / `output_dir` exist and are
  writable.

Run it ad-hoc on any pod:

```bash
runpodctl exec <POD_ID> -- bash deploy/runpod/scripts/healthcheck.sh
```

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `worker requires COSMIC_RAY_ADDRESS or COSMIC_HEAD_IP` | Worker pod template missing the env vars. | Edit the template (or `runpodctl update pod`) and re-deploy. |
| `ray is not installed` | Image was built without the extras. | Use the supplied Dockerfiles which install `[distributed]`. |
| Worker can't reach head's `6379` | RunPod HTTP proxy can't carry Ray TCP. | Map `6379/tcp` on the head pod, or set up a Tailscale / SSH `-L` tunnel. |
| `cuda.available=false` on a GPU pod | Container missing CUDA runtime. | Use `Dockerfile.worker` / `Dockerfile.training` (CUDA bases), not `Dockerfile.viewer`. |
| Viewer can't connect to runtime | `COSMIC_HEAD_IP` set to the internal IP from outside the project. | Use the public proxy host:port from the runtime pod's Connect tab. |
| Pod recycled mid-run | Spot pod preemption. | Move long jobs to On-Demand pods or add checkpointing. |

## File map

```
deploy/runpod/
├── README.md                       # this file
├── docker-compose.runpod.yml       # local emulation
├── docker/
│   ├── Dockerfile.runtime          # CUDA runtime, CosmicRuntime API
│   ├── Dockerfile.worker           # CUDA runtime, Ray worker
│   ├── Dockerfile.viewer           # slim Python, AI Viewer
│   └── Dockerfile.training         # CUDA devel, PyTorch training
├── scripts/
│   ├── start_runtime.sh
│   ├── start_worker.sh
│   ├── start_viewer.sh
│   ├── start_training.sh
│   └── healthcheck.sh
└── config/
    ├── runtime.env.example
    ├── worker.env.example
    └── ray_cluster.yaml.example
```
