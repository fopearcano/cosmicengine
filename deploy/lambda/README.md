# CosmicEngine on Lambda Cloud

This directory contains everything needed to deploy CosmicEngine on
[Lambda Cloud](https://lambdalabs.com/) as a distributed, multi-GPU
system. The local single-node workflow is **unchanged** — these
files only add new ways to run.

## Overview

CosmicEngine is split into four node roles. You can run them all on
one box or scale each role independently:

| Role | Image | Default port | What it does |
|---|---|---|---|
| `runtime` | `Dockerfile.runtime` | `8765` | Hosts the `CosmicRuntime` TCP server: scene state, registry, observers, reality views. |
| `worker` | `Dockerfile.worker` | – | Joins a Ray cluster and runs photon warps, density reconstruction, Gaussian splatting. |
| `viewer` | `Dockerfile.viewer` | – | Pulls frames from the runtime and post-processes them. |
| `training`| `Dockerfile.training` | – | Optional. Runs training scripts for the neural spacetime field. |

## Architecture

```
                    ┌────────────────────────┐
                    │  Head node             │
                    │  ─ Ray head            │
                    │  ─ CosmicRuntime API   │
                    │  ─ Job coordinator     │
                    └─────────┬──────────────┘
                              │ 6379 / 8765
              ┌───────────────┼───────────────┐
              │               │               │
       ┌──────┴───────┐ ┌─────┴─────┐  ┌──────┴──────┐
       │ AI workers   │ │ Render    │  │ Training    │
       │ ─ ONNX warp  │ │  workers  │  │  (optional) │
       │ ─ density    │ │ ─ Gaussian│  │  ─ PyTorch  │
       │ ─ neural st. │ │   splat   │  │  ─ ONNX out │
       └──────────────┘ └───────────┘  └─────────────┘
                              │
                       ┌──────┴──────┐
                       │ Viewer pods │
                       │ ─ AIViewer  │
                       └─────────────┘

  Shared storage (mounted on every node)
   /workspace/data      /workspace/models      /workspace/outputs
```

## Lambda Cloud deployment model

Lambda Cloud gives you bare GPU instances; there's **no managed Ray
autoscaler** at the time of writing. Provision instances by hand
(or with Terraform), share storage between them with a Lambda
filesystem, and bring up Ray manually using the start scripts in
[`scripts/`](scripts).

### Recommended Lambda instance types

| Role | Recommended | Why |
|---|---|---|
| Runtime / head | `gpu_1x_a10` or `cpu_1x` | Modest CPU, no need for big GPU. |
| AI worker | `gpu_1x_a10` / `gpu_1x_a100` | ONNX inference + neural warps. |
| Render worker | `gpu_1x_a10` | WebGPU / Gaussian splatting fits in one consumer GPU. |
| Training | `gpu_1x_a100` (40 GB) | The spacetime model is small but training is faster on A100. |
| Viewer | `cpu_1x` | Just receives frames + writes PPMs. |

## Single-node deployment

Quickest path to a running cluster — useful for smoke tests and
demos.

```bash
# 1. Provision one Lambda instance (gpu_1x_a10 is plenty).
# 2. Mount the shared filesystem at /workspace.
# 3. SSH in:
ssh ubuntu@<INSTANCE_IP>
git clone https://github.com/<your-org>/cosmicengine.git /workspace/cosmicengine
cd /workspace/cosmicengine

# 4. Install the engine + distributed extras.
pip install -e .[distributed]

# 5. Bring up Ray + runtime in the same process.
ray start --head --port=6379 --dashboard-host=0.0.0.0

# 6. Source the env file and start the runtime.
set -a; . deploy/lambda/config/runtime.env.example; set +a
bash deploy/lambda/scripts/start_runtime.sh
```

The runtime now listens on `0.0.0.0:8765`. From a viewer machine:

```bash
COSMIC_HEAD_IP=<INSTANCE_IP> COSMIC_RUNTIME_PORT=8765 \
    bash deploy/lambda/scripts/start_viewer.sh
```

## Multi-node Ray deployment

Pick one head node and any number of workers. They must share the
same `/workspace` filesystem (Lambda's mounted FS, an NFS export,
or rsync to an object store).

### 1. Start the head

```bash
ssh ubuntu@<HEAD_IP>
cd /workspace/cosmicengine
pip install -e .[distributed]

ray start --head --port=6379 --dashboard-host=0.0.0.0 --dashboard-port=8265

set -a; . deploy/lambda/config/runtime.env.example; set +a
bash deploy/lambda/scripts/start_runtime.sh
```

### 2. Attach workers

On every worker node:

```bash
ssh ubuntu@<WORKER_IP>
cd /workspace/cosmicengine
pip install -e .[distributed]

# Replace <HEAD_IP> with the head's IP (private if same VPC).
export COSMIC_HEAD_IP=<HEAD_IP>
export COSMIC_RAY_ADDRESS=<HEAD_IP>:6379
ray start --address=<HEAD_IP>:6379

bash deploy/lambda/scripts/start_worker.sh
```

### 3. Start a training node (optional)

```bash
ssh ubuntu@<TRAINING_IP>
cd /workspace/cosmicengine
pip install -e .[distributed]

ray start --address=<HEAD_IP>:6379
bash deploy/lambda/scripts/start_training.sh
# Then shell in separately and run e.g.
#   python examples/train_spacetime_model.py
```

## Persistent storage strategy

Every node expects three directories under `/workspace`:

| Path | Contents |
|---|---|
| `/workspace/data` | Catalog CSVs, JPL ephemeris snapshots, per-tile streaming data. |
| `/workspace/models`| ONNX / PyTorch checkpoints. |
| `/workspace/outputs`| Rendered PPMs, viewer captures, training artefacts. |

Two ways to share this between nodes:

1. **Lambda mounted filesystem (recommended).** Attach the same
   filesystem to every instance at `/workspace`.
2. **Object-store sync.** Use `rsync` or `aws s3 sync` periodically
   between `/workspace` and an S3-compatible bucket. Slower but
   works across regions.

## SSH / networking assumptions

* The head node must be reachable from every worker on **port 6379**
  (Ray GCS) and from every viewer on **port 8765** (CosmicRuntime).
* Workers don't need any inbound ports open; they reach back to the
  head over the same connection.
* The Ray dashboard listens on **port 8265** — useful for debugging
  but should be firewalled to your operator IP, not the public
  internet.

### Open ports cheat-sheet

| Direction | Port | Service |
|---|---|---|
| inbound on head | 6379 | Ray GCS |
| inbound on head | 8265 | Ray dashboard (operator only) |
| inbound on head | 10001 | Ray client |
| inbound on head | 8765 | CosmicRuntime API |

## Local docker-compose

For smoke-testing the topology without spinning up Lambda
instances, use [`docker-compose.lambda.yml`](docker-compose.lambda.yml):

```bash
cd deploy/lambda
docker compose -f docker-compose.lambda.yml up --build
```

The compose file launches `ray-head`, `worker-ai`, `worker-render`,
and a `viewer`. GPUs are not required (uncomment the
`deploy.resources.reservations.devices` block on a service to expose
one when the NVIDIA Container Toolkit is installed).

## Health checks

Every image bakes in a `HEALTHCHECK` that runs:

```bash
python -m cosmic_engine.distributed.entrypoints health
```

The command emits a JSON report on stdout summarising:

* Hostname, Python version, platform.
* `DistributedConfig` summary (role, ports, paths).
* CUDA availability (`available`, `device_count`, `device_names`).
* Ray availability (`available`, `initialized`, `cluster_resources`).
* Whether `data_dir` / `model_dir` / `output_dir` exist and are
  writable.

Run it ad-hoc on any node to debug a misconfiguration.

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `worker requires COSMIC_RAY_ADDRESS or COSMIC_HEAD_IP` | Worker started without an address. | Source `worker.env.example` and set `HEAD_NODE_IP`. |
| `ray is not installed` | Engine installed without the extras. | `pip install -e .[distributed]`. |
| Ray dashboard refuses connection | Port 8265 firewalled. | Open it (operator-only) or `kubectl port-forward` style tunnel through SSH. |
| `cuda.available=false` on a GPU instance | Container missing NVIDIA toolkit. | On Lambda this is preinstalled on GPU images; in compose, install NVIDIA Container Toolkit + uncomment the `devices` block. |
| Viewer cannot connect to runtime | `COSMIC_HEAD_IP` still set to `0.0.0.0`. | Use a routable IP / hostname (`runtime`, `ray-head`, or the public IP). |

## File map

```
deploy/lambda/
├── README.md                       # this file
├── docker-compose.lambda.yml       # local emulation
├── docker/
│   ├── Dockerfile.runtime          # CosmicRuntime API
│   ├── Dockerfile.worker           # Ray worker (GPU-ready)
│   ├── Dockerfile.viewer           # AI Viewer client
│   └── Dockerfile.training         # Training (CUDA devel)
├── scripts/
│   ├── start_runtime.sh
│   ├── start_worker.sh
│   ├── start_viewer.sh
│   ├── start_training.sh
│   └── healthcheck.sh
└── config/
    ├── runtime.env.example
    ├── worker.env.example
    ├── lambda_cluster.env.example
    └── ray_cluster.yaml.example
```
