"""Proper-time advancement helpers (SR + weak-field GR)."""

from __future__ import annotations

import math

import numpy as np

from cosmic_engine.core.units import SPEED_OF_LIGHT_M_S


# Cap beta to keep gamma finite even when callers feed in stale or
# extreme values. Anything within (- _BETA_CEIL, + _BETA_CEIL) gives a
# well-conditioned gamma.
_BETA_CEIL = 1.0 - 1.0e-9
# Smallest separation we'll trust for a Newtonian potential. Below this
# we clamp the contribution to keep the result finite.
_MIN_R = 1.0
# Floor for the (1 + Φ/c²) factor — never let proper time run backwards
# even in deep wells.
_GR_FACTOR_FLOOR = 1.0e-3
_GRAVITATIONAL_CONSTANT = 6.67430e-11


def gamma_from_beta(beta: float) -> float:
    """Return ``γ = 1 / sqrt(1 − β²)``, clamped to ``β < 1``.

    Negative ``beta`` is treated as ``|beta|`` since gamma is even in
    its argument.
    """
    b = abs(float(beta))
    if b >= _BETA_CEIL:
        b = _BETA_CEIL
    return 1.0 / math.sqrt(1.0 - b * b)


def advance_proper_time(
    tau: float,
    delta_t: float,
    beta: float,
    gravitational_potential: float | None = None,
) -> float:
    """Return the new proper time after a coordinate-time step ``delta_t``.

    Special relativity: ``dτ = dt / γ(β)``.
    Optional weak-field gravity: ``dτ *= (1 + Φ/c²)``. ``Φ`` should be
    *negative* in a gravity well; the factor is floored at a small
    positive value so dτ never goes negative for absurd inputs.
    """
    if delta_t < 0.0:
        raise ValueError(f"delta_t must be non-negative; got {delta_t}")
    gamma = gamma_from_beta(beta)
    dtau = delta_t / gamma
    if gravitational_potential is not None:
        c2 = SPEED_OF_LIGHT_M_S * SPEED_OF_LIGHT_M_S
        factor = 1.0 + gravitational_potential / c2
        if factor < _GR_FACTOR_FLOOR:
            factor = _GR_FACTOR_FLOOR
        dtau *= factor
    return float(tau) + dtau


def gravitational_potential_weak(
    position: np.ndarray,
    masses: list[tuple[np.ndarray, float]],
) -> float:
    """Return ``Φ ≈ -Σ G·M / r`` evaluated at ``position``.

    ``masses`` is a list of ``(position, mass_kg)`` pairs. Distances
    smaller than ``_MIN_R`` are clamped to that floor so a coincident
    mass doesn't produce ``-inf``. Returns ``0.0`` if ``masses`` is
    empty or every contribution is invalid.
    """
    if not masses:
        return 0.0
    p = np.asarray(position, dtype=np.float64)
    if p.shape != (3,):
        raise ValueError("position must have shape (3,)")
    total = 0.0
    for mass_position, mass_kg in masses:
        if mass_kg is None or mass_kg <= 0.0:
            continue
        mp = np.asarray(mass_position, dtype=np.float64)
        if mp.shape != (3,):
            raise ValueError("each mass_position must have shape (3,)")
        delta = mp - p
        r = float(np.linalg.norm(delta))
        if r < _MIN_R:
            r = _MIN_R
        total -= _GRAVITATIONAL_CONSTANT * float(mass_kg) / r
    return total
