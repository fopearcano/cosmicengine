"""Per-zone description used by the multi-scale system."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# Scale tags accepted by the multi-scale system; mirrored in
# :mod:`cosmic_engine.multiscale.representation`.
_VALID_REPRESENTATION_TYPES = (
    "galaxy_field",
    "star_field",
    "nbody",
    "density_field",
    "neural_field",
)


@dataclass
class ScaleZone:
    """A half-open scale interval ``[min_scale_m, max_scale_m)`` with a
    rendering hint.

    ``representation_type`` chooses how the runtime should turn the
    objects in this zone into something the renderer can draw.
    """

    name: str
    min_scale_m: float
    max_scale_m: float
    representation_type: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("ScaleZone.name must be a non-empty string")
        if self.min_scale_m < 0.0:
            raise ValueError(
                f"ScaleZone.min_scale_m must be non-negative; got {self.min_scale_m}"
            )
        if self.max_scale_m <= self.min_scale_m:
            raise ValueError(
                f"ScaleZone.max_scale_m must be greater than min_scale_m; "
                f"got [{self.min_scale_m}, {self.max_scale_m})"
            )
        if self.representation_type not in _VALID_REPRESENTATION_TYPES:
            raise ValueError(
                f"ScaleZone.representation_type must be one of "
                f"{list(_VALID_REPRESENTATION_TYPES)}; "
                f"got {self.representation_type!r}"
            )

    def contains_scale(self, distance_m: float) -> bool:
        """``True`` iff ``distance_m`` lies in ``[min, max)``."""
        return self.min_scale_m <= distance_m < self.max_scale_m
