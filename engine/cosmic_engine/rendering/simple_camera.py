"""Minimal camera pose and field-of-view description."""

from __future__ import annotations

from dataclasses import dataclass

from cosmic_engine.core.vector import Vector3


_ZERO = Vector3.zero()


@dataclass
class SimpleCamera:
    """Pinhole-style camera pose.

    ``forward`` and ``up`` are directions in the same Cartesian frame as
    the universe objects (equatorial). They need not be unit length or
    perfectly orthogonal — renderers should orthonormalize as needed.
    """

    position_m: Vector3
    forward: Vector3
    up: Vector3
    fov_degrees: float = 90.0
    image_width: int = 512
    image_height: int = 512

    def validate(self) -> None:
        """Raise :class:`ValueError` if any field is out of bounds."""
        if not (0.0 < self.fov_degrees < 180.0):
            raise ValueError(
                f"fov_degrees must be in (0, 180); got {self.fov_degrees}"
            )
        if self.image_width <= 0 or self.image_height <= 0:
            raise ValueError(
                "image_width and image_height must be positive"
            )
        if self.forward == _ZERO:
            raise ValueError("forward must not be the zero vector")
        if self.up == _ZERO:
            raise ValueError("up must not be the zero vector")
