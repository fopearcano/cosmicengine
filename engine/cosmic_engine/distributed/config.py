"""Environment-driven distributed configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass


_VALID_ROLES = ("head", "worker", "runtime", "viewer", "training")


@dataclass
class DistributedConfig:
    """Per-node configuration for a Lambda-Cloud-style deployment.

    Every field maps directly to a ``COSMIC_*`` environment variable
    so containers can be configured without code changes. Missing
    variables fall back to the documented defaults; unknown roles
    raise :class:`ValueError` from :meth:`validate`.
    """

    ray_address: str | None = None
    role: str = "runtime"
    num_gpus: float | None = None
    num_cpus: int | None = None
    object_store_memory_gb: float | None = None
    head_ip: str | None = None
    head_port: int = 6379
    runtime_host: str = "0.0.0.0"
    runtime_port: int = 8765
    data_dir: str = "/workspace/data"
    model_dir: str = "/workspace/models"
    output_dir: str = "/workspace/outputs"
    enable_gpu: bool = False
    enable_viewer: bool = False
    enable_training: bool = False
    instance_type: str | None = None
    node_name: str | None = None

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> "DistributedConfig":
        """Build a config from ``os.environ`` (or a passed-in dict for tests)."""
        e = env if env is not None else os.environ

        def _f(key: str) -> float | None:
            v = e.get(key)
            if v is None or v == "":
                return None
            try:
                return float(v)
            except ValueError:
                return None

        def _i(key: str, default: int | None = None) -> int | None:
            v = e.get(key)
            if v is None or v == "":
                return default
            try:
                return int(float(v))
            except ValueError:
                return default

        def _b(key: str, default: bool = False) -> bool:
            v = e.get(key)
            if v is None or v == "":
                return default
            return v.strip().lower() in ("1", "true", "yes", "on")

        return cls(
            ray_address=e.get("COSMIC_RAY_ADDRESS") or None,
            role=e.get("COSMIC_ROLE", "runtime") or "runtime",
            num_gpus=_f("COSMIC_NUM_GPUS"),
            num_cpus=_i("COSMIC_NUM_CPUS"),
            object_store_memory_gb=_f("COSMIC_OBJECT_STORE_GB"),
            head_ip=e.get("COSMIC_HEAD_IP") or None,
            head_port=_i("COSMIC_HEAD_PORT", 6379) or 6379,
            runtime_host=e.get("COSMIC_RUNTIME_HOST", "0.0.0.0") or "0.0.0.0",
            runtime_port=_i("COSMIC_RUNTIME_PORT", 8765) or 8765,
            data_dir=e.get("COSMIC_DATA_DIR", "/workspace/data"),
            model_dir=e.get("COSMIC_MODEL_DIR", "/workspace/models"),
            output_dir=e.get("COSMIC_OUTPUT_DIR", "/workspace/outputs"),
            enable_gpu=_b("COSMIC_ENABLE_GPU"),
            enable_viewer=_b("COSMIC_ENABLE_VIEWER"),
            enable_training=_b("COSMIC_ENABLE_TRAINING"),
            instance_type=e.get("COSMIC_LAMBDA_INSTANCE_TYPE") or None,
            node_name=e.get("COSMIC_NODE_NAME") or None,
        )

    def validate(self) -> None:
        """Raise :class:`ValueError` on bad role / port / GPU values."""
        if self.role not in _VALID_ROLES:
            raise ValueError(
                f"unknown role {self.role!r}; "
                f"expected one of {list(_VALID_ROLES)}"
            )
        if not (0 < self.head_port < 65_536):
            raise ValueError(
                f"head_port out of range: {self.head_port}"
            )
        if not (0 < self.runtime_port < 65_536):
            raise ValueError(
                f"runtime_port out of range: {self.runtime_port}"
            )
        if self.num_gpus is not None and self.num_gpus < 0:
            raise ValueError("num_gpus must be non-negative")
        if self.num_cpus is not None and self.num_cpus < 0:
            raise ValueError("num_cpus must be non-negative")
        if (
            self.object_store_memory_gb is not None
            and self.object_store_memory_gb < 0.0
        ):
            raise ValueError("object_store_memory_gb must be non-negative")

    def summary(self) -> dict:
        """Return a JSON-friendly dict for logging / health reports."""
        return {
            "role": self.role,
            "node_name": self.node_name,
            "instance_type": self.instance_type,
            "ray_address": self.ray_address,
            "head_ip": self.head_ip,
            "head_port": self.head_port,
            "runtime_host": self.runtime_host,
            "runtime_port": self.runtime_port,
            "num_gpus": self.num_gpus,
            "num_cpus": self.num_cpus,
            "object_store_memory_gb": self.object_store_memory_gb,
            "enable_gpu": self.enable_gpu,
            "enable_viewer": self.enable_viewer,
            "enable_training": self.enable_training,
            "data_dir": self.data_dir,
            "model_dir": self.model_dir,
            "output_dir": self.output_dir,
        }
