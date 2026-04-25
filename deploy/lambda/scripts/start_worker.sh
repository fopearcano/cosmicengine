#!/usr/bin/env bash
# Attach this node to an existing Ray head and idle as a Ray worker.
set -euo pipefail

: "${COSMIC_ROLE:=worker}"
: "${COSMIC_HEAD_PORT:=6379}"

if [[ -z "${COSMIC_RAY_ADDRESS:-}" && -z "${COSMIC_HEAD_IP:-}" ]]; then
    echo "[start_worker] error: set COSMIC_RAY_ADDRESS or COSMIC_HEAD_IP" >&2
    exit 1
fi

if [[ -z "${COSMIC_RAY_ADDRESS:-}" ]]; then
    export COSMIC_RAY_ADDRESS="${COSMIC_HEAD_IP}:${COSMIC_HEAD_PORT}"
fi

echo "[start_worker] connecting to ${COSMIC_RAY_ADDRESS}"
echo "[start_worker] node_name=${COSMIC_NODE_NAME:-<unset>}"
echo "[start_worker] enable_gpu=${COSMIC_ENABLE_GPU:-false}"

export COSMIC_ROLE=worker
exec python -m cosmic_engine.distributed.entrypoints worker
