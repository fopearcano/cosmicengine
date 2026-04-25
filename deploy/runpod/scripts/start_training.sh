#!/usr/bin/env bash
# Idle as an opt-in training pod. Operators SSH into the pod (RunPod
# provides per-pod SSH) and run training scripts manually; this
# entrypoint just keeps the pod alive and prints the available
# commands.
set -euo pipefail

: "${COSMIC_ROLE:=training}"
: "${COSMIC_DATA_DIR:=/workspace/data}"
: "${COSMIC_MODEL_DIR:=/workspace/models}"
: "${COSMIC_OUTPUT_DIR:=/workspace/outputs}"

mkdir -p "${COSMIC_DATA_DIR}" "${COSMIC_MODEL_DIR}" "${COSMIC_OUTPUT_DIR}"

echo "[start_training] data=${COSMIC_DATA_DIR} models=${COSMIC_MODEL_DIR}"
echo "[start_training] ray_address=${COSMIC_RAY_ADDRESS:-<none>}"
echo "[start_training] runpod_pod_id=${RUNPOD_POD_ID:-<not in runpod>}"

export COSMIC_ROLE=training
exec python -m cosmic_engine.distributed.entrypoints training
