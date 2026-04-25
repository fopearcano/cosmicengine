#!/usr/bin/env bash
# Run the AI Viewer client against a CosmicRuntime server.
set -euo pipefail

: "${COSMIC_ROLE:=viewer}"
: "${COSMIC_HEAD_IP:=runtime}"
: "${COSMIC_RUNTIME_PORT:=8765}"
: "${COSMIC_OUTPUT_DIR:=/workspace/outputs}"

mkdir -p "${COSMIC_OUTPUT_DIR}"

echo "[start_viewer] connecting to ${COSMIC_HEAD_IP}:${COSMIC_RUNTIME_PORT}"
echo "[start_viewer] output_dir=${COSMIC_OUTPUT_DIR}"

export COSMIC_ROLE=viewer
exec python -m cosmic_engine.distributed.entrypoints viewer
