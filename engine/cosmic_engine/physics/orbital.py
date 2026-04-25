"""Simplified two-body Keplerian orbital mechanics.

What this is:
- Newton's iteration for Kepler's equation ``M = E − e·sin(E)``.
- Heliocentric ecliptic Cartesian position from a fixed set of orbital
  elements at an epoch.

What this is **not**:
- A perturbation theory or N-body integrator.
- Relativistic.
- Time-resolved beyond mean-motion advance from the epoch.

Use this for solar-system-scale local dynamics, NOT for galaxies or
high-precision ephemerides.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from cosmic_engine.core.vector import Vector3


_SECONDS_PER_DAY = 86_400.0
_TWO_PI = 2.0 * math.pi


@dataclass
class OrbitalElements:
    """Classical orbital elements at an epoch (heliocentric, ecliptic).

    All angles are in degrees, distances in meters, period in seconds.
    """

    semi_major_axis_m: float
    eccentricity: float
    inclination_deg: float
    longitude_ascending_node_deg: float
    argument_of_periapsis_deg: float
    mean_anomaly_at_epoch_deg: float
    epoch_julian_date: float
    orbital_period_seconds: float


def mean_motion(period_seconds: float) -> float:
    """Return ``n = 2π / T`` in radians per second."""
    if period_seconds <= 0.0:
        raise ValueError("period_seconds must be positive")
    return _TWO_PI / period_seconds


def solve_kepler_equation(
    mean_anomaly_rad: float,
    eccentricity: float,
    *,
    tol: float = 1.0e-12,
    max_iter: int = 100,
) -> float:
    """Solve ``M = E − e·sin(E)`` for ``E`` via Newton's method.

    Only valid for elliptical orbits (``0 ≤ e < 1``); raises
    :class:`ValueError` otherwise.
    """
    if eccentricity < 0.0 or eccentricity >= 1.0:
        raise ValueError(
            f"eccentricity must be in [0, 1); got {eccentricity}"
        )
    e = eccentricity
    m = mean_anomaly_rad
    # warm-start: E≈M for low e, π for high e (better basin of attraction)
    E = m if e < 0.8 else math.pi
    for _ in range(max_iter):
        f = E - e * math.sin(E) - m
        fp = 1.0 - e * math.cos(E)
        delta = f / fp
        E -= delta
        if abs(delta) < tol:
            return E
    return E


def orbital_position_from_elements(
    elements: OrbitalElements,
    julian_date: float,
) -> Vector3:
    """Heliocentric ecliptic Cartesian position (meters) at ``julian_date``.

    Two-body, no perturbations, no light-time corrections.
    """
    if elements.eccentricity < 0.0 or elements.eccentricity >= 1.0:
        raise ValueError(
            "OrbitalElements.eccentricity must be in [0, 1) for this solver"
        )

    dt_seconds = (julian_date - elements.epoch_julian_date) * _SECONDS_PER_DAY
    n = mean_motion(elements.orbital_period_seconds)

    m0 = math.radians(elements.mean_anomaly_at_epoch_deg)
    mean_anomaly = m0 + n * dt_seconds
    # wrap into [−π, π] so the Newton iteration stays in its sweet spot
    mean_anomaly = ((mean_anomaly + math.pi) % _TWO_PI) - math.pi

    e = elements.eccentricity
    eccentric_anomaly = solve_kepler_equation(mean_anomaly, e)

    sin_half = math.sin(0.5 * eccentric_anomaly)
    cos_half = math.cos(0.5 * eccentric_anomaly)
    true_anomaly = 2.0 * math.atan2(
        math.sqrt(1.0 + e) * sin_half,
        math.sqrt(1.0 - e) * cos_half,
    )

    a = elements.semi_major_axis_m
    r = a * (1.0 - e * math.cos(eccentric_anomaly))

    omega = math.radians(elements.argument_of_periapsis_deg)
    big_omega = math.radians(elements.longitude_ascending_node_deg)
    inclination = math.radians(elements.inclination_deg)

    angle = omega + true_anomaly
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    cos_o = math.cos(big_omega)
    sin_o = math.sin(big_omega)
    cos_i = math.cos(inclination)
    sin_i = math.sin(inclination)

    x = r * (cos_o * cos_a - sin_o * sin_a * cos_i)
    y = r * (sin_o * cos_a + cos_o * sin_a * cos_i)
    z = r * (sin_a * sin_i)
    return Vector3(x, y, z)
