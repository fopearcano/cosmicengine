"""Tests for Phase 11 flat-ΛCDM cosmology."""

from __future__ import annotations

import math

import pytest

from cosmic_engine.core.units import PARSEC_IN_METERS, SPEED_OF_LIGHT_M_S
from cosmic_engine.physics.cosmology import (
    get_cosmology_mode,
    redshift_to_distance_m,
    set_cosmology_mode,
)
from cosmic_engine.physics.cosmology_lcdm import (
    H0_KM_S_MPC,
    H0_to_SI,
    MPC_IN_METERS,
    OMEGA_L,
    OMEGA_M,
    angular_diameter_distance_m,
    comoving_distance_m,
    distance_modulus,
    E_z,
    integrate_simpson,
    luminosity_distance_m,
)


# --- E(z) ---


def test_E_z_at_zero_is_one():
    assert E_z(0.0) == pytest.approx(1.0)


def test_E_z_increases_with_z():
    assert E_z(2.0) > E_z(1.0) > E_z(0.5) > E_z(0.0)


def test_E_z_consistent_with_omegas():
    # E(1)^2 should equal Ω_m*8 + Ω_Λ
    assert E_z(1.0) ** 2 == pytest.approx(OMEGA_M * 8.0 + OMEGA_L)


# --- Simpson integrator ---


def test_simpson_integrates_polynomial_exactly():
    # ∫₀² x² dx = 8/3
    assert integrate_simpson(lambda x: x * x, 0.0, 2.0, 10) == pytest.approx(
        8.0 / 3.0
    )


def test_simpson_rejects_odd_n():
    with pytest.raises(ValueError):
        integrate_simpson(lambda x: x, 0.0, 1.0, 7)


def test_simpson_rejects_non_positive_n():
    with pytest.raises(ValueError):
        integrate_simpson(lambda x: x, 0.0, 1.0, 0)


def test_simpson_zero_interval():
    assert integrate_simpson(lambda x: 1.0, 1.0, 1.0, 4) == 0.0


# --- comoving / luminosity / angular diameter distances ---


def test_comoving_distance_at_zero_is_zero():
    assert comoving_distance_m(0.0) == 0.0


def test_comoving_distance_rejects_negative_z():
    with pytest.raises(ValueError):
        comoving_distance_m(-0.1)


def test_comoving_distance_increases_with_z():
    assert (
        comoving_distance_m(0.05)
        < comoving_distance_m(0.1)
        < comoving_distance_m(0.5)
        < comoving_distance_m(1.0)
    )


def test_low_z_lcdm_close_to_hubble():
    # For z << 1 the integrator should agree with the linear Hubble law.
    z = 0.01
    hubble = z * SPEED_OF_LIGHT_M_S / H0_to_SI(H0_KM_S_MPC)
    lcdm = comoving_distance_m(z)
    assert lcdm == pytest.approx(hubble, rel=1e-2)


def test_d_l_greater_than_d_c_and_d_a_less_than_d_c():
    z = 0.5
    d_c = comoving_distance_m(z)
    d_l = luminosity_distance_m(z)
    d_a = angular_diameter_distance_m(z)
    assert d_l > d_c
    assert d_a < d_c
    # flat-universe identities
    assert d_l == pytest.approx((1.0 + z) * d_c, rel=1e-12)
    assert d_a == pytest.approx(d_c / (1.0 + z), rel=1e-12)


def test_simpson_step_count_stability():
    z = 0.5
    coarse = comoving_distance_m(z, steps=64)
    fine = comoving_distance_m(z, steps=512)
    assert coarse == pytest.approx(fine, rel=1e-4)


def test_distance_modulus_z_zero_raises():
    with pytest.raises(ValueError):
        distance_modulus(0.0)


def test_distance_modulus_consistent_with_d_l():
    z = 0.3
    mu = distance_modulus(z)
    expected = 5.0 * (
        math.log10(luminosity_distance_m(z) / PARSEC_IN_METERS) - 1.0
    )
    assert mu == pytest.approx(expected, rel=1e-12)


# --- mode switching ---


def test_default_mode_is_lcdm():
    assert get_cosmology_mode() == "lcdm"


def test_set_cosmology_mode_changes_distance_function():
    z = 0.1
    set_cosmology_mode("lcdm")
    lcdm = redshift_to_distance_m(z)
    set_cosmology_mode("hubble")
    hubble = redshift_to_distance_m(z)
    set_cosmology_mode("lcdm")  # restore for other tests
    # at z=0.1 LCDM and Hubble agree to within ~5%, but they should differ
    assert lcdm != hubble
    assert lcdm == pytest.approx(hubble, rel=0.1)


def test_set_cosmology_mode_rejects_unknown():
    with pytest.raises(ValueError):
        set_cosmology_mode("ekpyrotic")


# --- H0 conversion ---


def test_H0_to_SI_value():
    # 70 km/s/Mpc ≈ 2.27e-18 / s
    assert H0_to_SI(70.0) == pytest.approx(2.268e-18, rel=1e-3)
    # 1 Mpc cancels out: H0_SI · MPC_IN_METERS == H0_km_s
    assert H0_to_SI(70.0) * MPC_IN_METERS == pytest.approx(70_000.0, rel=1e-12)
