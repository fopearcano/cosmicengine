"""Neural-field-driven geodesic integration with deterministic fallback.

When a :class:`SpacetimeFieldModel` is available, the per-step
acceleration is sampled from the model. If the model raises (or
isn't present), we fall back to the analytical Schwarzschild
formula in :func:`schwarzschild_acceleration`. The integration
shape is identical to :func:`integrate_geodesic_step` so the rest
of the marcher is unchanged.
"""

from __future__ import annotations

import math

import numpy as np

from ai_viewer.neural_field.gr.geodesic import (
    schwarzschild_acceleration,
)
from cosmic_engine.ai.spacetime_field import SpacetimeFieldModel
from cosmic_engine.core.units import SPEED_OF_LIGHT_M_S


_MAX_STEP_DEFLECTION_RAD: float = math.pi / 4.0


def integrate_geodesic_step_neural(
    position: np.ndarray,
    direction: np.ndarray,
    step_size: float,
    model: SpacetimeFieldModel | None,
    fallback_mass_kg: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Advance one ``step_size`` using the neural model, with analytical fallback.

    Returns ``(new_position, new_direction)``. If ``model`` is ``None``
    or its query fails, the analytical Schwarzschild acceleration with
    ``fallback_mass_kg`` is used instead. Direction is renormalized;
    per-step deflection capped at π/4 to keep the integrator stable.
    """
    if step_size <= 0.0:
        raise ValueError("step_size must be positive")
    p = np.asarray(position, dtype=np.float64).reshape(3)
    d = np.asarray(direction, dtype=np.float64).reshape(3)
    n = float(np.linalg.norm(d))
    if n == 0.0:
        return p, d
    d = d / n

    accel: np.ndarray | None = None
    if model is not None:
        try:
            accel = np.asarray(
                model.query_acceleration(p, d), dtype=np.float64
            ).reshape(3)
            if not np.isfinite(accel).all():
                accel = None
        except Exception:
            accel = None
    if accel is None:
        accel = schwarzschild_acceleration(p, d, fallback_mass_kg)

    # Convert acceleration into a per-step rotation toward the
    # acceleration direction. The magnitude that survives goes through
    # the same dtheta = a * step / c^2 path the analytical step uses.
    accel_mag = float(np.linalg.norm(accel))
    if accel_mag > 0.0:
        a_hat = accel / accel_mag
        along = float(np.dot(a_hat, d))
        perp_vec = a_hat - along * d
        perp_norm = float(np.linalg.norm(perp_vec))
        if perp_norm > 0.0:
            n_perp = perp_vec / perp_norm
            dtheta = (
                accel_mag * step_size
                / (SPEED_OF_LIGHT_M_S * SPEED_OF_LIGHT_M_S)
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
