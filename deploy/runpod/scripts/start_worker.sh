#!/usr/bin/env bash
# Attach this RunPod pod to an existing Ray head and idle as a Ray
# worker. The head address typically comes in via the COSMIC_*
# environment vars set on the pod template.
set -euo pipefail

: "${COSMIC_ROLE:=worker}"
: "${COSMIC_HEAD_PORT:=6379}"

if [[ -z "${COSMIC_RAY_ADDRESS:-}" && -z "${COSMIC_HEAD_IP:-}" ]]; then
    echo "[start_worker] error: set COSMIC_RAY_ADDRESS or COSMIC_HEAD_IP" >&2
    echo "[start_worker]        on the RunPod pod template (or via runpodctl)" >&2
    exit 1
fi

if [[ -z "${COSMIC_RAY_ADDRESS:-}" ]]; then
    export COSMIC_RAY_ADDRESS="${COSMIC_HEAD_IP}:${COSMIC_HEAD_PORT}"
fi

echo "[start_worker] connecting to ${COSMIC_RAY_ADDRESS}"
echo "[start_worker] node_name=${COSMIC_NODE_NAME:-<unset>}"
echo "[start_worker] enable_gpu=${COSMIC_ENABLE_GPU:-false}"
echo "[start_worker] runpod_pod_id=${RUNPOD_POD_ID:-<not in runpod>}"

export COSMIC_ROLE=worker
exec python -m cosmic_engine.distributed.entrypoints worker
