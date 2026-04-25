"""Flat ΛCDM cosmology with numerical comoving-distance integration.

Assumptions and limitations:
- Flat geometry (Ω_k = 0).
- Cold dark matter + cosmological constant only — no radiation,
  no neutrinos, no time-varying dark energy equation of state.
- Single-component matter density Ω_m = 0.3, Ω_Λ = 0.7 by default.
- Distances computed with Simpson's rule on the line-of-sight integral
  of ``c / H(z)``.

This is a self-contained implementation for moderate catalogs. For
sub-percent precision over ``z > ~3`` or non-flat geometries, replace
with a proper integrator (Astropy / camb) once those become acceptable
dependencies.
"""

from __future__ import annotations

import math
from typing import Callable

from cosmic_engine.core.units import PARSEC_IN_METERS, SPEED_OF_LIGHT_M_S


# --- model parameters (flat ΛCDM defaults) ---

H0_KM_S_MPC: float = 70.0
OMEGA_M: float = 0.3
OMEGA_L: float = 0.7

# 1 Mpc in meters; used for the H0 unit conversion only.
MPC_IN_METERS: float = 3.0856775814913673e22


def H0_to_SI(H0_km_s_mpc: float) -> float:
    """Return the Hubble constant expressed in 1/s."""
    return H0_km_s_mpc * 1_000.0 / MPC_IN_METERS


def E_z(z: float) -> float:
    """Dimensionless expansion rate ``E(z) = sqrt(Ω_m·(1+z)³ + Ω_Λ)``."""
    one_plus_z = 1.0 + z
    return math.sqrt(OMEGA_M * one_plus_z * one_plus_z * one_plus_z + OMEGA_L)


def integrate_simpson(
    f: Callable[[float], float],
    a: float,
    b: float,
    n: int,
) -> float:
    """Composite Simpson's rule on ``n`` (even) sub-intervals over ``[a, b]``."""
    if n <= 0:
        raise ValueError(f"n must be positive; got {n}")
    if n % 2 != 0:
        raise ValueError(f"n must be even for Simpson's rule; got {n}")
    if a == b:
        return 0.0
    h = (b - a) / n
    total = f(a) + f(b)
    for i in range(1, n):
        x = a + i * h
        total += (4.0 if i % 2 else 2.0) * f(x)
    return total * h / 3.0


def comoving_distance_m(z: float, steps: int = 256) -> float:
    """Line-of-sight comoving distance ``D_C`` in meters.

    Defined as ``D_C(z) = (c/H0) · ∫₀ᶻ dz' / E(z')``. Returns 0 for
    ``z == 0``. Raises ``ValueError`` for negative ``z``.
    """
    if z < 0.0:
        raise ValueError(f"z must be non-negative; got {z}")
    if z == 0.0:
        return 0.0
    hubble_distance_m = SPEED_OF_LIGHT_M_S / H0_to_SI(H0_KM_S_MPC)
    integral = integrate_simpson(lambda zp: 1.0 / E_z(zp), 0.0, z, steps)
    return hubble_distance_m * integral


def luminosity_distance_m(z: float, steps: int = 256) -> float:
    """Luminosity distance in meters; ``D_L = (1+z) · D_C`` for a flat universe."""
    return (1.0 + z) * comoving_distance_m(z, steps)


def angular_diameter_distance_m(z: float, steps: int = 256) -> float:
    """Angular diameter distance in meters; ``D_A = D_C / (1+z)`` for flat geometry."""
    if z < 0.0:
        raise ValueError(f"z must be non-negative; got {z}")
    return comoving_distance_m(z, steps) / (1.0 + z)


def distance_modulus(z: float, steps: int = 256) -> float:
    """Distance modulus ``μ = 5·log10(D_L / 10 pc)`` (dimensionless magnitudes).

    Undefined at ``z = 0`` (D_L = 0); raises ``ValueError`` there.
    """
    d_l_m = luminosity_distance_m(z, steps)
    if d_l_m <= 0.0:
        raise ValueError("distance_modulus is undefined at z = 0")
    d_l_pc = d_l_m / PARSEC_IN_METERS
    return 5.0 * (math.log10(d_l_pc) - 1.0)
