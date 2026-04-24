"""State carried by the perceiving observer."""

from __future__ import annotations

import math
from dataclasses import dataclass

from cosmic_engine.core.units import SPEED_OF_LIGHT_M_S
from cosmic_engine.core.vector import Vector3


_ZERO = Vector3.zero()


@dataclass
class ObserverState:
    """Pose, motion, and perception-warp knob of the observer.

    ``velocity_m_s`` is the observer's motion in the same Cartesian
    frame as universe objects. ``warp_factor`` >= 1.0 is a deliberate
    knob that scales the perception effects beyond their physical
    magnitude — it has no direct physical meaning, just a dial for
    visualizing the effects at low real velocities.
    """

    position_m: Vector3
    velocity_m_s: Vector3
    forward: Vector3
    up: Vector3
    warp_factor: float = 1.0

    def speed_magnitude(self) -> float:
        """Return |velocity| in m/s."""
        v = self.velocity_m_s
        return math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z)

    def beta(self) -> float:
        """Return v/c, dimensionless."""
        return self.speed_magnitude() / SPEED_OF_LIGHT_M_S

    def validate(self) -> None:
        """Raise :class:`ValueError` if any field violates the invariants."""
        if self.warp_factor < 1.0:
            raise ValueError(
                f"warp_factor must be >= 1.0; got {self.warp_factor}"
            )
        if self.beta() >= 1.0:
            raise ValueError("observer velocity must be subluminal (beta < 1)")
        if self.forward == _ZERO:
            raise ValueError("forward must not be the zero vector")
        if self.up == _ZERO:
            raise ValueError("up must not be the zero vector")
