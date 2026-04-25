"""Approximate redshift → distance helpers.

Uses the linear Hubble law:

    v = z * c
    d = v / H0

with ``H0 = 70 km/s/Mpc``. This is **only** a low-redshift
approximation (good to a few percent for ``z ≲ 0.3``). It ignores
spatial curvature, the cosmological constant, matter density, and
peculiar velocities. Treat it as a placeholder until a proper
ΛCDM distance integrator lands.
"""

from __future__ import annotations

from cosmic_engine.core.units import LIGHTYEAR_IN_METERS, SPEED_OF_LIGHT_M_S


# Hubble constant in km/s per Mpc (Planck-ish reference value).
HUBBLE_CONSTANT_KM_S_MPC: float = 70.0

# 1 Mpc in meters; used only to convert H0 into SI.
_MPC_IN_METERS: float = 3.0856775814913673e22

# Hubble constant expressed as 1/s, so v = H0 * d works in SI directly.
HUBBLE_CONSTANT_PER_S: float = (
    HUBBLE_CONSTANT_KM_S_MPC * 1_000.0 / _MPC_IN_METERS
)


def redshift_to_velocity(z: float) -> float:
    """Return ``v ≈ z * c`` in m/s (low-redshift approximation)."""
    return float(z) * SPEED_OF_LIGHT_M_S


def redshift_to_distance_m(z: float) -> float:
    """Return Hubble-law distance in meters.

    Valid only for small ``z`` (≲ 0.3). Future replacement should swap
    this for a comoving / luminosity / proper-distance integrator that
    accounts for curvature, dark energy, and matter density.
    """
    return redshift_to_velocity(z) / HUBBLE_CONSTANT_PER_S


def redshift_to_distance_lightyears(z: float) -> float:
    """Return Hubble-law distance in light-years."""
    return redshift_to_distance_m(z) / LIGHTYEAR_IN_METERS
