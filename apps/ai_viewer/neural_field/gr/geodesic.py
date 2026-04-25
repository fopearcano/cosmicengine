"""Approximate photon geodesics around a Schwarzschild mass.

Stepwise trajectory integration: at each substep we compute the
gravitational deflection rate at the photon's current location and
rotate its direction toward the mass by a small angle, then advance
the position. The math is intentionally simpler than full GR — we
use ``2 G M / (c² r²)`` as the perpendicular deflection coefficient,
which reproduces the textbook ``α = 4 G M / (c² b)`` for a grazing
ray when integrated over a long-enough trajectory and is stable
enough for real-time stepping.

The mass is assumed to sit at the origin; callers in
:mod:`ai_viewer.neural_field.gr.ray_marcher` translate by the black
hole's offset before calling these functions.
"""

from __future__ import annotations

import math

import numpy as np

from cosmic_engine.core.units import SPEED_OF_LIGHT_M_S
from cosmic_engine.physics.nbody import GRAVITATIONAL_CONSTANT


# Cap per-step deflection so a tiny ``r`` near the singularity can't
# rotate the direction by an arbitrary angle.
_MAX_STEP_DEFLECTION_RAD: float = math.pi / 4.0


def schwarzschild_acceleration(
    position: np.ndarray,
    velocity: np.ndarray,
    mass_kg: float,
) -> np.ndarray:
    """Approximate "acceleration" on a photon near a mass at the origin.

    Returns a 3-vector pointing toward the mass with magnitude
    ``2 G M / r²`` — the factor 2 captures the GR deflection
    enhancement vs a Newtonian calculation. The ``velocity`` argument
    is unused at this approximation level but is part of the API so
    a future upgrade (Christoffel symbols / coupling to v) can drop in
    without changing the signature.
    """
    if mass_kg <= 0.0:
        return np.zeros(3, dtype=np.float64)
    p = np.asarray(position, dtype=np.float64).reshape(3)
    r = float(np.linalg.norm(p))
    if r == 0.0:
        return np.zeros(3, dtype=np.float64)
    r_hat = -p / r  # toward the mass at origin
    magnitude = (
        2.0 * GRAVITATIONAL_CONSTANT * mass_kg / (r * r)
    )
    return r_hat * magnitude


def integrate_geodesic_step(
    position: np.ndarray,
    direction: np.ndarray,
    step_size: float,
    mass_kg: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Advance one ``step_size`` along the geodesic.

    Returns ``(new_position, new_direction)``. Direction is
    renormalized to unit length on output. ``step_size`` should be
    much smaller than the local curvature radius (the test suite
    exercises the stable regime).
    """
    if step_size <= 0.0:
        raise ValueError("step_size must be positive")
    p = np.asarray(position, dtype=np.float64).reshape(3)
    d = np.asarray(direction, dtype=np.float64).reshape(3)
    n = float(np.linalg.norm(d))
    if n == 0.0:
        return p, d
    d = d / n

    if mass_kg > 0.0:
        r = float(np.linalg.norm(p))
        if r > 0.0:
            r_hat = -p / r  # toward the mass
            along = float(np.dot(r_hat, d))
            perp_vec = r_hat - along * d
            perp_norm = float(np.linalg.norm(perp_vec))
            if perp_norm > 0.0:
                n_perp = perp_vec / perp_norm
                dtheta = (
                    2.0
                    * GRAVITATIONAL_CONSTANT
                    * mass_kg
                    * step_size
                    / (
                        SPEED_OF_LIGHT_M_S
                        * SPEED_OF_LIGHT_M_S
                        * r
                        * r
                    )
                )
                if not math.isfinite(dtheta) or dtheta < 0.0:
                    dtheta = 0.0
                if dtheta > _MAX_STEP_DEFLECTION_RAD:
                    dtheta = _MAX_STEP_DEFLECTION_RAD
                if dtheta > 0.0:
                    bent = (
                        math.cos(dtheta) * d
                        + math.sin(dtheta) * n_perp
                    )
                    bent_norm = float(np.linalg.norm(bent))
                    if bent_norm > 0.0 and math.isfinite(bent_norm):
                        d = bent / bent_norm

    new_position = p + d * float(step_size)
    return new_position, d
