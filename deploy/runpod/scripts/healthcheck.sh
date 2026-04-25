#!/usr/bin/env bash
# Print the CosmicEngine health report as JSON.
# Used by Docker HEALTHCHECK and by operators running ad-hoc on a
# RunPod pod (`runpodctl exec <pod_id> -- bash deploy/runpod/scripts/healthcheck.sh`).
set -euo pipefail

exec python -m cosmic_engine.distributed.entrypoints health
