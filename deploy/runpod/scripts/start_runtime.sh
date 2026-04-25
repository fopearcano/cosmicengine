#!/usr/bin/env bash
# Start CosmicEngine runtime / API service inside a RunPod pod.
#
# RunPod exposes ports through its TCP proxy; the *internal* listen
# port stays at COSMIC_RUNTIME_PORT (default 8765). External clients
# reach the pod via the proxied URL printed in the RunPod dashboard.
set -euo pipefail

: "${COSMIC_ROLE:=runtime}"
: "${COSMIC_RUNTIME_HOST:=0.0.0.0}"
: "${COSMIC_RUNTIME_PORT:=8765}"
: "${COSMIC_DATA_DIR:=/workspace/data}"
: "${COSMIC_MODEL_DIR:=/workspace/models}"
: "${COSMIC_OUTPUT_DIR:=/workspace/outputs}"

mkdir -p "${COSMIC_DATA_DIR}" "${COSMIC_MODEL_DIR}" "${COSMIC_OUTPUT_DIR}"

echo "[start_runtime] role=${COSMIC_ROLE}"
echo "[start_runtime] runtime=${COSMIC_RUNTIME_HOST}:${COSMIC_RUNTIME_PORT}"
echo "[start_runtime] data=${COSMIC_DATA_DIR} models=${COSMIC_MODEL_DIR} outputs=${COSMIC_OUTPUT_DIR}"
echo "[start_runtime] ray_address=${COSMIC_RAY_ADDRESS:-<none>}"
echo "[start_runtime] runpod_pod_id=${RUNPOD_POD_ID:-<not in runpod>}"
echo "[start_runtime] runpod_public_ip=${RUNPOD_PUBLIC_IP:-<unset>}"

export COSMIC_ROLE=runtime
exec python -m cosmic_engine.distributed.entrypoints runtime
