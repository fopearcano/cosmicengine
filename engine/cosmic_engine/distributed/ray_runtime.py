"""Ray cluster bring-up wrapper.

Every helper here is safe to import without ``ray`` installed.
:func:`is_ray_available` is the cheap check; the real
:func:`init_ray` attempts the import and raises a clear
:class:`RuntimeError` when Ray is missing.
"""

from __future__ import annotations

import logging
from typing import Any

from cosmic_engine.distributed.config import DistributedConfig


_LOG = logging.getLogger(__name__)


def is_ray_available() -> bool:
    """Return ``True`` iff ``ray`` can be imported in this process."""
    try:
        import ray  # noqa: F401
    except ImportError:
        return False
    return True


def init_ray(config: DistributedConfig) -> Any:
    """Initialize / connect to a Ray cluster according to ``config``.

    - ``role == "head"`` and ``ray_address`` empty: start a new local
      Ray runtime (cluster head).
    - Any other role with ``ray_address`` set: connect as a client.
    - ``role`` left blank with no address: start an in-process Ray
      (handy for single-node smoke tests).

    Raises :class:`RuntimeError` if Ray isn't installed.
    """
    if not is_ray_available():
        raise RuntimeError(
            "ray is not installed; install the 'distributed' extra "
            "(`pip install -e .[distributed]`) to enable distributed mode"
        )
    import ray  # local import — only succeeds when extra is present

    config.validate()
    init_kwargs: dict[str, Any] = {}
    if config.num_cpus is not None:
        init_kwargs["num_cpus"] = config.num_cpus
    if config.num_gpus is not None:
        init_kwargs["num_gpus"] = config.num_gpus
    if config.object_store_memory_gb is not None:
        init_kwargs["object_store_memory"] = int(
            config.object_store_memory_gb * 1024 * 1024 * 1024
        )

    if config.ray_address:
        _LOG.info("ray.init: connecting to %s", config.ray_address)
        return ray.init(address=config.ray_address, **init_kwargs)

    _LOG.info("ray.init: starting local runtime (role=%s)", config.role)
    return ray.init(**init_kwargs)


def shutdown_ray() -> None:
    """Best-effort ``ray.shutdown()``; no-op if Ray isn't running."""
    if not is_ray_available():
        return
    try:
        import ray

        if ray.is_initialized():
            ray.shutdown()
    except Exception as e:  # pragma: no cover - defensive
        _LOG.warning("ray.shutdown failed: %s", e)
