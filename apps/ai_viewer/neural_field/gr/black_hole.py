"""Schwarzschild-style black-hole renderer (approximate).

The :class:`BlackHole` carries a position and mass and exposes
helpers used by the GR rendering paths:

- :meth:`schwarzschild_radius` — ``R_s = 2GM/c²``.
- :meth:`is_inside_event_horizon` — point inclusion test.
- :meth:`deflection_strength` — saturating intensity proxy that grows
  toward infinity as the impact distance approaches ``R_s``.

Combined with :func:`apply_lensing_to_points`, this is enough for a
real-time-capable visualization of a black hole's gravitational
lensing on a Gaussian field.
"""

from __future__ import annotations

import math

import numpy as np

from ai_viewer.neural_field.gaussian import GaussianPoint
from ai_viewer.neural_field.gr.lensing import apply_lensing_to_points
from cosmic_engine.physics.nbody import GRAVITATIONAL_CONSTANT
from cosmic_engine.core.units import SPEED_OF_LIGHT_M_S


_MAX_STRENGTH: float = 1.0e6


class BlackHole:
    """A point-mass black hole used for GR-flavored rendering effects."""

    def __init__(
        self,
        position: np.ndarray | tuple[float, float, float],
        mass_kg: float,
    ) -> None:
        if mass_kg <= 0.0:
            raise ValueError("mass_kg must be positive")
        self.position = np.asarray(position, dtype=np.float64).reshape(3)
        self.mass_kg = float(mass_kg)

    def schwarzschild_radius(self) -> float:
        """``R_s = 2GM/c²``."""
        return (
            2.0
            * GRAVITATIONAL_CONSTANT
            * self.mass_kg
            / (SPEED_OF_LIGHT_M_S * SPEED_OF_LIGHT_M_S)
        )

    def is_inside_event_horizon(
        self,
        point: np.ndarray | tuple[float, float, float],
    ) -> bool:
        """Strict-inside test (boundary inclusive)."""
        p = np.asarray(point, dtype=np.float64).reshape(3)
        d = float(np.linalg.norm(p - self.position))
        return d <= self.schwarzschild_radius()

    def deflection_strength(self, distance: float) -> float:
        """Dimensionless strength that scales with ``R_s / (distance - R_s)``.

        Saturates at ``_MAX_STRENGTH`` at or below the event horizon
        so callers can use it as a multiplier without producing infs.
        """
        if not math.isfinite(distance) or distance < 0.0:
            return 0.0
        rs = self.schwarzschild_radius()
        if distance <= rs:
            return _MAX_STRENGTH
        gap = distance - rs
        if gap <= 0.0:
            return _MAX_STRENGTH
        strength = rs / gap
        if not math.isfinite(strength):
            return _MAX_STRENGTH
        return min(strength, _MAX_STRENGTH)


def apply_black_hole_to_points(
    points: list[GaussianPoint],
    black_hole: BlackHole,
    observer_position: np.ndarray,
) -> list[GaussianPoint]:
    """Combine lensing and event-horizon absorption.

    Points strictly inside the event horizon are removed; the rest
    are bent by the standard weak-field deflection. The input list
    is never mutated.
    """
    if not points:
        return []
    survivors: list[GaussianPoint] = [
        p for p in points if not black_hole.is_inside_event_horizon(p.position)
    ]
    if not survivors:
        return []
    return apply_lensing_to_points(
        survivors,
        black_hole.position,
        black_hole.mass_kg,
        np.asarray(observer_position, dtype=np.float64).reshape(3),
    )
