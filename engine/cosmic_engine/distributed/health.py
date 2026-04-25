"""Lightweight node-level health probes."""

from __future__ import annotations

import os
import platform
import socket
import sys
import time
from pathlib import Path
from typing import Any

from cosmic_engine.distributed.config import DistributedConfig
from cosmic_engine.distributed.ray_runtime import is_ray_available


def check_cuda_available() -> dict[str, Any]:
    """Return CUDA availability + device count, defensively.

    Doesn't import torch at module load. Returns ``{"available": False,
    "reason": "..."}`` on any failure rather than raising — health
    checks should never crash the caller.
    """
    try:
        import torch

        try:
            available = bool(torch.cuda.is_available())
        except Exception as e:  # pragma: no cover - defensive
            return {"available": False, "reason": f"torch.cuda failed: {e}"}
        if not available:
            return {"available": False, "reason": "no CUDA device detected"}
        try:
            device_count = int(torch.cuda.device_count())
        except Exception:  # pragma: no cover - defensive
            device_count = 0
        device_names: list[str] = []
        try:
            for i in range(device_count):
                device_names.append(str(torch.cuda.get_device_name(i)))
        except Exception:  # pragma: no cover - defensive
            pass
        return {
            "available": True,
            "device_count": device_count,
            "device_names": device_names,
        }
    except ImportError:
        return {"available": False, "reason": "torch not installed"}


def check_ray_status() -> dict[str, Any]:
    """Return Ray installation + initialization state.

    Reports ``available`` (the import works) and ``initialized``
    (a Ray runtime is attached to this process).
    """
    if not is_ray_available():
        return {"available": False, "initialized": False}
    try:
        import ray

        initialized = bool(ray.is_initialized())
        info: dict[str, Any] = {
            "available": True,
            "initialized": initialized,
            "version": getattr(ray, "__version__", "unknown"),
        }
        if initialized:
            try:
                rsrc = ray.cluster_resources()
                info["cluster_resources"] = {
                    k: float(v) for k, v in rsrc.items()
                }
            except Exception:  # pragma: no cover - defensive
                pass
        return info
    except Exception as e:  # pragma: no cover - defensive
        return {
            "available": True,
            "initialized": False,
            "error": f"{type(e).__name__}: {e}",
        }


def check_data_dirs(config: DistributedConfig) -> dict[str, Any]:
    """Existence + writability check for ``data_dir / model_dir / output_dir``."""
    out: dict[str, Any] = {}
    for label, path in (
        ("data_dir", config.data_dir),
        ("model_dir", config.model_dir),
        ("output_dir", config.output_dir),
    ):
        if not path:
            out[label] = {"path": path, "exists": False, "writable": False}
            continue
        p = Path(path)
        out[label] = {
            "path": str(p),
            "exists": p.is_dir(),
            "writable": (
                os.access(str(p), os.W_OK) if p.is_dir() else False
            ),
        }
    return out


def health_report(config: DistributedConfig | None = None) -> dict[str, Any]:
    """Aggregate every probe into a single JSON-serializable dict.

    Pass ``None`` to use :meth:`DistributedConfig.from_env`.
    """
    if config is None:
        config = DistributedConfig.from_env()
    return {
        "timestamp": time.time(),
        "hostname": socket.gethostname(),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "config": config.summary(),
        "cuda": check_cuda_available(),
        "ray": check_ray_status(),
        "directories": check_data_dirs(config),
    }
