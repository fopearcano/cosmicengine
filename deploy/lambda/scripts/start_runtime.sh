#!/usr/bin/env bash
# Start the CosmicEngine runtime / API service.
#
# Honours every COSMIC_* env var read by DistributedConfig.from_env.
# Connects to a Ray head if COSMIC_RAY_ADDRESS is set, then runs the
# blocking RuntimeServer on COSMIC_RUNTIME_HOST:COSMIC_RUNTIME_PORT.
set -euo pipefail

: "${COSMIC_ROLE:=runtime}"
: "${COSMIC_RUNTIME_HOST:=0.0.0.0}"
: "${COSMIC_RUNTIME_PORT:=8765}"
: "${COSMIC_DATA_DIR:=/workspace/data}"
: "${COSMIC_MODEL_DIR:=/workspace/models}"
: "${COSMIC_OUTPUT_DIR:=/workspace/outputs}"

mkdir -p "${COSMIC_DATA_DIR}" "${COSMIC_MODEL_DIR}" "${COSMIC_OUTPUT_DIR}"

echo "[start_runtime] role=${COSMIC_ROLE} host=${COSMIC_RUNTIME_HOST}:${COSMIC_RUNTIME_PORT}"
echo "[start_runtime] data=${COSMIC_DATA_DIR} models=${COSMIC_MODEL_DIR} outputs=${COSMIC_OUTPUT_DIR}"
echo "[start_runtime] ray_address=${COSMIC_RAY_ADDRESS:-<none>}"

export COSMIC_ROLE=runtime
exec python -m cosmic_engine.distributed.entrypoints runtime
