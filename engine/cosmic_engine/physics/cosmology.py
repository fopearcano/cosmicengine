"""Redshift → distance helpers with selectable cosmology mode.

Two cosmology modes are available:

- ``"lcdm"`` (default): flat ΛCDM via numerical Simpson integration of
  ``c / H(z)``; see :mod:`cosmic_engine.physics.cosmology_lcdm`.
- ``"hubble"``: linear Hubble law ``v = z·c, d = v / H₀``. Valid only
  for ``z ≲ 0.3``; kept for backwards compatibility and quick checks.

Switch globally with :func:`set_cosmology_mode`.
"""

from __future__ import annotations

from cosmic_engine.core.units import LIGHTYEAR_IN_METERS, SPEED_OF_LIGHT_M_S
from cosmic_engine.physics.cosmology_lcdm import (
    H0_KM_S_MPC,
    H0_to_SI,
    angular_diameter_distance_m,
    comoving_distance_m,
    distance_modulus,
    luminosity_distance_m,
)


# --- mode selection ---------------------------------------------------

_MODE: str = "lcdm"


def set_cosmology_mode(mode: str) -> None:
    """Switch the global cosmology mode (``"lcdm"`` or ``"hubble"``)."""
    global _MODE
    if mode not in ("lcdm", "hubble"):
        raise ValueError(
            f"unknown cosmology mode {mode!r}; expected 'lcdm' or 'hubble'"
        )
    _MODE = mode


def get_cosmology_mode() -> str:
    """Return the current global cosmology mode."""
    return _MODE


# --- legacy Hubble-law constants (kept for backwards compatibility) ---

HUBBLE_CONSTANT_KM_S_MPC: float = H0_KM_S_MPC
HUBBLE_CONSTANT_PER_S: float = H0_to_SI(HUBBLE_CONSTANT_KM_S_MPC)


# --- public helpers ----------------------------------------------------


def redshift_to_velocity(z: float) -> float:
    """Return ``v ≈ z · c`` in m/s. Independent of cosmology mode."""
    return float(z) * SPEED_OF_LIGHT_M_S


def _hubble_distance_m(z: float) -> float:
    """Linear Hubble-law distance in meters (small-z only)."""
    return redshift_to_velocity(z) / HUBBLE_CONSTANT_PER_S


def redshift_to_distance_m(z: float) -> float:
    """Return the redshift → distance value for the active mode.

    ``"lcdm"`` (default) returns the flat-ΛCDM comoving distance.
    ``"hubble"`` returns the linear Hubble-law distance.
    """
    if _MODE == "lcdm":
        return comoving_distance_m(float(z))
    return _hubble_distance_m(float(z))


def redshift_to_distance_lightyears(z: float) -> float:
    """Return the active-mode redshift → distance in light-years."""
    return redshift_to_distance_m(z) / LIGHTYEAR_IN_METERS


__all__ = [
    "HUBBLE_CONSTANT_KM_S_MPC",
    "HUBBLE_CONSTANT_PER_S",
    "angular_diameter_distance_m",
    "comoving_distance_m",
    "distance_modulus",
    "get_cosmology_mode",
    "luminosity_distance_m",
    "redshift_to_distance_lightyears",
    "redshift_to_distance_m",
    "redshift_to_velocity",
    "set_cosmology_mode",
]
