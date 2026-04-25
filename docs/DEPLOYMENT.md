# Deployment

CosmicEngine has three deployment paths, in increasing order of
complexity:

1. **Local dev** — `pip install -e .` and run scripts directly.
2. **Docker compose** — `docker compose -f
   deploy/lambda/docker-compose.lambda.yml up`.
3. **Lambda Cloud** — multi-node Ray cluster.

## Local dev

The simplest path. No containers, no cluster.

```bash
git clone <repo> cosmicengine
cd cosmicengine
python -m venv .venv && source .venv/bin/activate
pip install -e .
python -m pytest tests/
```

Runtime in one terminal, viewer in another:

```bash
# terminal 1
python examples/runtime_server_demo.py

# terminal 2
python examples/runtime_client_demo.py
```

## Local Docker emulation

Spins up the full per-role topology on one host.

```bash
cd deploy/lambda
docker compose -f docker-compose.lambda.yml up --build
```

Services:
- `ray-head` — runtime + Ray head (ports `8765`, `6379`, `8265`).
- `worker-ai` — Ray worker.
- `worker-render` — Ray worker (Gaussian splatting).
- `viewer` — AI Viewer client.
- `runtime` (opt-in via `--profile dedicated-runtime`) — separate runtime.

GPU: uncomment the `deploy.resources.reservations.devices` block on
worker / training services and install the NVIDIA Container Toolkit.

## Lambda Cloud

The full operator guide lives in
[`deploy/lambda/README.md`](../deploy/lambda/README.md). TL;DR:

### Single node

```bash
ssh ubuntu@<INSTANCE_IP>
cd /workspace/cosmicengine
pip install -e .[distributed]
ray start --head --port=6379 --dashboard-host=0.0.0.0
set -a; . deploy/lambda/config/runtime.env.example; set +a
bash deploy/lambda/scripts/start_runtime.sh
```

### Multi-node

Head:

```bash
ray start --head --port=6379 --dashboard-host=0.0.0.0 --dashboard-port=8265
bash deploy/lambda/scripts/start_runtime.sh
```

Worker (replace `<HEAD_IP>`):

```bash
export COSMIC_HEAD_IP=<HEAD_IP> COSMIC_RAY_ADDRESS=<HEAD_IP>:6379
ray start --address=<HEAD_IP>:6379
bash deploy/lambda/scripts/start_worker.sh
```

### Health check

On any node:

```bash
python -m cosmic_engine.distributed.entrypoints health | jq .
```

Returns CUDA availability, Ray status, directory writability, and
the resolved `DistributedConfig`. Used as the Docker `HEALTHCHECK`
on every shipped image.

## Per-role images

| Image | Base | Includes |
|---|---|---|
| `runtime` | `python:3.11-slim` | CosmicRuntime API server. No CUDA. |
| `worker` | `nvidia/cuda:12.4.1-runtime-ubuntu22.04` | Ray worker, ONNX, Gaussian splat. |
| `viewer` | `python:3.11-slim` | AI Viewer client. No CUDA. |
| `training` | `nvidia/cuda:12.4.1-devel-ubuntu22.04` | PyTorch + onnxscript. |

Images are kept separate so a runtime pod doesn't pay the cost of
training dependencies, and a viewer pod doesn't pay the cost of
CUDA.

## Environment reference

All `COSMIC_*` env vars consumed by the entrypoints:

| Var | Default | Meaning |
|---|---|---|
| `COSMIC_ROLE` | `runtime` | One of `head / worker / runtime / viewer / training`. |
| `COSMIC_RAY_ADDRESS` | – | `host:port` of an existing Ray head. |
| `COSMIC_HEAD_IP` | – | Convenience; used to build `COSMIC_RAY_ADDRESS` if absent. |
| `COSMIC_HEAD_PORT` | `6379` | Ray GCS port. |
| `COSMIC_RUNTIME_HOST` | `0.0.0.0` | CosmicRuntime bind address. |
| `COSMIC_RUNTIME_PORT` | `8765` | CosmicRuntime port. |
| `COSMIC_DATA_DIR` | `/workspace/data` | Catalog data root. |
| `COSMIC_MODEL_DIR` | `/workspace/models` | ONNX / PyTorch checkpoints. |
| `COSMIC_OUTPUT_DIR` | `/workspace/outputs` | Rendered frames. |
| `COSMIC_ENABLE_GPU` | `false` | Informational; Ray init reads it. |
| `COSMIC_ENABLE_VIEWER` | `false` | Informational. |
| `COSMIC_ENABLE_TRAINING` | `false` | Informational. |
| `COSMIC_NUM_GPUS` | – | Override Ray init `num_gpus`. |
| `COSMIC_NUM_CPUS` | – | Override Ray init `num_cpus`. |
| `COSMIC_OBJECT_STORE_GB` | – | Override Ray init object store size. |
| `COSMIC_LAMBDA_INSTANCE_TYPE` | – | Free-form instance label. |
| `COSMIC_NODE_NAME` | – | Free-form node label. |

## Persistent storage

Every node expects three roots under `/workspace`:

```
/workspace/data       catalog CSVs, ephemeris snapshots, tile data
/workspace/models     ONNX / PyTorch checkpoints
/workspace/outputs    rendered frames + viewer captures
```

On Lambda Cloud, attach the same filesystem to every instance.
Alternatively, `rsync` between nodes and a shared bucket.

## Open ports cheat-sheet

| Port | Service | Direction |
|---|---|---|
| `8765` | CosmicRuntime API | inbound on head/runtime |
| `6379` | Ray GCS | inbound on head |
| `10001` | Ray client | inbound on head |
| `8265` | Ray dashboard | inbound on head, **operator only** |
