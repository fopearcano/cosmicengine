"""Tests for Phase 10 cosmology helpers."""

from __future__ import annotations

import math

import pytest

from cosmic_engine.core.units import LIGHTYEAR_IN_METERS, SPEED_OF_LIGHT_M_S
from cosmic_engine.physics.cosmology import (
    HUBBLE_CONSTANT_KM_S_MPC,
    HUBBLE_CONSTANT_PER_S,
    redshift_to_distance_lightyears,
    redshift_to_distance_m,
    redshift_to_velocity,
)


def test_hubble_constant_value():
    assert HUBBLE_CONSTANT_KM_S_MPC == 70.0
    # 70 km/s/Mpc ≈ 2.27e-18 1/s
    assert HUBBLE_CONSTANT_PER_S == pytest.approx(2.268e-18, rel=1e-3)


def test_redshift_to_velocity_zero():
    assert redshift_to_velocity(0.0) == 0.0


def test_redshift_to_velocity_low_z_matches_zc():
    assert redshift_to_velocity(0.1) == pytest.approx(
        0.1 * SPEED_OF_LIGHT_M_S
    )


def test_redshift_to_distance_zero_is_zero():
    assert redshift_to_distance_m(0.0) == 0.0
    assert redshift_to_distance_lightyears(0.0) == 0.0


def test_redshift_to_distance_positive():
    assert redshift_to_distance_m(0.1) > 0.0
    assert redshift_to_distance_lightyears(0.1) > 0.0


def test_redshift_to_distance_monotonic():
    assert redshift_to_distance_m(0.2) > redshift_to_distance_m(0.1)
    assert redshift_to_distance_m(0.05) > redshift_to_distance_m(0.01)


def test_redshift_to_distance_lightyears_consistent_with_meters():
    z = 0.1
    assert redshift_to_distance_lightyears(z) == pytest.approx(
        redshift_to_distance_m(z) / LIGHTYEAR_IN_METERS, rel=1e-12
    )


def test_low_redshift_distance_within_expected_order_of_magnitude():
    # z=0.1 -> ~430 Mpc with H0=70 km/s/Mpc: ~1.32e25 m, ~1.4e9 ly
    d = redshift_to_distance_m(0.1)
    assert 1.0e25 < d < 2.0e25
    ly = redshift_to_distance_lightyears(0.1)
    assert 1.0e9 < ly < 2.0e9


def test_redshift_negative_velocity_for_blueshift():
    # peculiar negative-z values still flow through linearly
    assert redshift_to_velocity(-0.001) == pytest.approx(
        -0.001 * SPEED_OF_LIGHT_M_S
    )
