"""Distributed runtime layer (Phase 39, Lambda Cloud-ready).

Optional. Without ``ray[default]`` installed, importing this package
still works — every Ray-dependent function is gated behind
:func:`is_ray_available` so unit tests and local single-node runs
never need a cluster.

Quick import roadmap:

- :class:`DistributedConfig` — env-driven role + resource config.
- :func:`init_ray` / :func:`shutdown_ray` / :func:`is_ray_available`.
- :mod:`tasks` — Ray remote tasks (with non-Ray fallbacks).
- :mod:`health` — node-level health report (CUDA / Ray / data dirs).
- :mod:`entrypoints` — head / worker / runtime / viewer / training.
"""

from cosmic_engine.distributed.config import DistributedConfig
from cosmic_engine.distributed.health import (
    check_cuda_available,
    check_data_dirs,
    check_ray_status,
    health_report,
)
from cosmic_engine.distributed.ray_runtime import (
    init_ray,
    is_ray_available,
    shutdown_ray,
)

__all__ = [
    "DistributedConfig",
    "check_cuda_available",
    "check_data_dirs",
    "check_ray_status",
    "health_report",
    "init_ray",
    "is_ray_available",
    "shutdown_ray",
]
