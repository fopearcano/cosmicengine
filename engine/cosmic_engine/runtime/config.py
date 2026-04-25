"""Runtime configuration for :class:`CosmicRuntime`."""

from __future__ import annotations

from dataclasses import dataclass


_VALID_PHYSICS_BACKENDS = ("none", "exact_nbody", "barnes_hut")


@dataclass
class RuntimeConfig:
    """Top-level dial for what the runtime actually does each frame.

    The defaults give a passive registry with no physics and no AI;
    rendering still works (perception transforms apply on top of the
    deterministic photon field).
    """

    enable_physics: bool = True
    physics_backend: str = "none"

    enable_perception: bool = True
    enable_ai_warp: bool = False
    enable_density_ai: bool = False

    max_active_objects: int = 100_000
    active_radius_m: float | None = None

    render_width: int = 512
    render_height: int = 512
    output_directory: str = "outputs"

    def validate(self) -> None:
        """Raise :class:`ValueError` if any field violates the invariants."""
        if self.physics_backend not in _VALID_PHYSICS_BACKENDS:
            raise ValueError(
                f"unknown physics_backend {self.physics_backend!r}; "
                f"expected one of {list(_VALID_PHYSICS_BACKENDS)}"
            )
        if self.render_width <= 0 or self.render_height <= 0:
            raise ValueError(
                "render_width and render_height must be positive"
            )
        if self.max_active_objects <= 0:
            raise ValueError("max_active_objects must be positive")
        if self.active_radius_m is not None and self.active_radius_m <= 0.0:
            raise ValueError("active_radius_m, if set, must be positive")
