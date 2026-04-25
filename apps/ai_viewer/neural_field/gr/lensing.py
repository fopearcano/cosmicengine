"""Weak-field gravitational lensing for a point mass.

Uses the textbook deflection angle ``α = 4GM/(c²b)`` for a ray with
impact parameter ``b`` past a mass ``M``. The deflection is applied
as a rotation of the ray direction toward the lens by the (small)
angle ``α``. Strictly correct only when ``b ≫ R_s``; the
:func:`apply_lensing` function clamps the rotation to keep things
stable when that's not true.
"""

from __future__ import annotations

import math

import numpy as np

from ai_viewer.neural_field.gaussian import GaussianPoint
from cosmic_engine.physics.nbody import GRAVITATIONAL_CONSTANT
from cosmic_engine.core.units import SPEED_OF_LIGHT_M_S


# Cap the deflection so the perpendicular bend can't flip past a
# right angle; beyond that the weak-field formula is meaningless.
_MAX_DEFLECTION_RAD: float = math.radians(85.0)


def compute_deflection_angle(
    impact_parameter_m: float,
    mass_kg: float,
) -> float:
    """Return ``α = 4GM/(c²b)`` in radians.

    Returns 0 for non-positive ``impact_parameter_m`` or ``mass_kg``.
    The result is capped at ``_MAX_DEFLECTION_RAD`` so callers can
    safely apply it as a 2D rotation.
    """
    if impact_parameter_m <= 0.0 or mass_kg <= 0.0:
        return 0.0
    alpha = (
        4.0 * GRAVITATIONAL_CONSTANT * mass_kg
        / (SPEED_OF_LIGHT_M_S * SPEED_OF_LIGHT_M_S * impact_parameter_m)
    )
    if not math.isfinite(alpha):
        return _MAX_DEFLECTION_RAD
    return min(alpha, _MAX_DEFLECTION_RAD)


def apply_lensing(
    direction: np.ndarray,
    lens_position: np.ndarray,
    observer_position: np.ndarray,
    mass_kg: float,
) -> np.ndarray:
    """Bend ``direction`` toward ``lens_position`` and return a unit vector.

    ``direction`` is treated as the ray's outgoing unit direction from
    ``observer_position``. The ray is bent by the deflection angle
    around the axis perpendicular to (ray, lens-to-ray-axis); rays
    passing on either side of the lens converge toward it.
    """
    direction = np.asarray(direction, dtype=np.float64).reshape(3)
    lens_position = np.asarray(lens_position, dtype=np.float64).reshape(3)
    observer_position = np.asarray(observer_position, dtype=np.float64).reshape(3)

    norm = np.linalg.norm(direction)
    if norm == 0.0:
        return direction
    d = direction / norm

    lens_offset = lens_position - observer_position
    along = float(np.dot(lens_offset, d))
    perp_vec = lens_offset - along * d
    impact = float(np.linalg.norm(perp_vec))
    if impact == 0.0 or along <= 0.0:
        # Lens is behind the observer or directly along the ray; no
        # well-defined perpendicular bend either way.
        return d

    alpha = compute_deflection_angle(impact, mass_kg)
    if alpha == 0.0:
        return d

    n_perp = perp_vec / impact
    bent = math.cos(alpha) * d + math.sin(alpha) * n_perp
    bent_norm = np.linalg.norm(bent)
    if bent_norm == 0.0 or not math.isfinite(bent_norm):
        return d
    return bent / bent_norm


def apply_lensing_to_points(
    points: list[GaussianPoint],
    lens_position: np.ndarray,
    mass_kg: float,
    observer_position: np.ndarray,
) -> list[GaussianPoint]:
    """Lens every point's apparent position around ``lens_position``.

    Each point's distance from the observer is preserved; only the
    direction is bent. Points coincident with the observer pass
    through unchanged. The original list is never mutated.
    """
    if not points:
        return []
    lens = np.asarray(lens_position, dtype=np.float64).reshape(3)
    observer = np.asarray(observer_position, dtype=np.float64).reshape(3)

    out: list[GaussianPoint] = []
    for p in points:
        offset = p.position - observer
        distance = float(np.linalg.norm(offset))
        if distance == 0.0:
            out.append(_clone_with_metadata(p, lensed=False))
            continue
        direction = offset / distance
        new_direction = apply_lensing(direction, lens, observer, mass_kg)
        new_position = observer + new_direction * distance
        out.append(_clone_with_metadata(p, lensed=True, position=new_position))
    return out


def _clone_with_metadata(
    point: GaussianPoint,
    *,
    lensed: bool,
    position: np.ndarray | None = None,
) -> GaussianPoint:
    metadata = dict(point.metadata)
    metadata["lensed"] = lensed
    return GaussianPoint(
        position=position if position is not None else point.position.copy(),
        color=point.color.copy(),
        intensity=point.intensity,
        sigma=point.sigma,
        object_id=point.object_id,
        truth_level=point.truth_level,
        metadata=metadata,
    )
