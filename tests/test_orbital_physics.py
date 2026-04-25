"""Tests for Phase 13 orbital mechanics and the simplified solar system."""

from __future__ import annotations

import math

import pytest

from cosmic_engine.core.object_types import CosmicObjectType
from cosmic_engine.core.truth import TruthLevel
from cosmic_engine.core.units import AU_IN_METERS
from cosmic_engine.physics.orbital import (
    OrbitalElements,
    mean_motion,
    orbital_position_from_elements,
    solve_kepler_equation,
)
from cosmic_engine.physics.solar_system import create_solar_system_objects


_J2000 = 2_451_545.0
_YEAR_S = 365.25 * 86_400.0


def _circular_elements(a_m: float, period_s: float) -> OrbitalElements:
    return OrbitalElements(
        semi_major_axis_m=a_m,
        eccentricity=0.0,
        inclination_deg=0.0,
        longitude_ascending_node_deg=0.0,
        argument_of_periapsis_deg=0.0,
        mean_anomaly_at_epoch_deg=45.0,
        epoch_julian_date=_J2000,
        orbital_period_seconds=period_s,
    )


# --- mean_motion ---


def test_mean_motion_positive_for_positive_period():
    assert mean_motion(_YEAR_S) > 0.0
    assert mean_motion(_YEAR_S) == pytest.approx(2.0 * math.pi / _YEAR_S)


def test_mean_motion_rejects_non_positive_period():
    with pytest.raises(ValueError):
        mean_motion(0.0)
    with pytest.raises(ValueError):
        mean_motion(-1.0)


# --- Kepler solver ---


def test_kepler_zero_eccentricity_returns_M():
    for m in (-1.0, 0.0, 0.5, 1.5):
        assert solve_kepler_equation(m, 0.0) == pytest.approx(m)


def test_kepler_satisfies_equation():
    for e in (0.1, 0.3, 0.6, 0.9):
        for m in (-2.0, -0.3, 0.0, 0.7, 2.5):
            E = solve_kepler_equation(m, e)
            assert m == pytest.approx(E - e * math.sin(E), abs=1e-10)


@pytest.mark.parametrize("bad_e", [-0.1, 1.0, 1.5])
def test_kepler_rejects_invalid_eccentricity(bad_e):
    with pytest.raises(ValueError):
        solve_kepler_equation(0.5, bad_e)


# --- orbital_position_from_elements ---


def test_orbital_position_returns_finite_vector():
    elements = _circular_elements(1.0e11, _YEAR_S)
    pos = orbital_position_from_elements(elements, _J2000 + 7.0)
    for c in (pos.x, pos.y, pos.z):
        assert math.isfinite(c)


def test_circular_orbit_distance_equals_semi_major_axis():
    a = 1.5e11
    elements = _circular_elements(a, _YEAR_S)
    for delta_days in (0.0, 30.0, 90.0, 180.0):
        pos = orbital_position_from_elements(elements, _J2000 + delta_days)
        d = math.sqrt(pos.x ** 2 + pos.y ** 2 + pos.z ** 2)
        assert d == pytest.approx(a, rel=1e-12)


def test_orbital_position_changes_over_one_orbit():
    elements = _circular_elements(1.0e11, _YEAR_S)
    p0 = orbital_position_from_elements(elements, _J2000)
    p_quarter = orbital_position_from_elements(
        elements, _J2000 + (_YEAR_S / 4.0) / 86_400.0
    )
    assert p0 != p_quarter


def test_orbital_position_periodic_after_one_period():
    elements = _circular_elements(1.0e11, _YEAR_S)
    p0 = orbital_position_from_elements(elements, _J2000)
    p_full = orbital_position_from_elements(
        elements, _J2000 + _YEAR_S / 86_400.0
    )
    assert p0.x == pytest.approx(p_full.x, rel=1e-9)
    assert p0.y == pytest.approx(p_full.y, rel=1e-9, abs=1e-3)
    assert p0.z == pytest.approx(p_full.z, abs=1e-3)


def test_orbital_position_rejects_invalid_eccentricity():
    bad = OrbitalElements(
        semi_major_axis_m=1.0e11,
        eccentricity=1.0,
        inclination_deg=0.0,
        longitude_ascending_node_deg=0.0,
        argument_of_periapsis_deg=0.0,
        mean_anomaly_at_epoch_deg=0.0,
        epoch_julian_date=_J2000,
        orbital_period_seconds=_YEAR_S,
    )
    with pytest.raises(ValueError):
        orbital_position_from_elements(bad, _J2000)


# --- create_solar_system_objects ---


def test_solar_system_has_sun_plus_eight_planets():
    objects = create_solar_system_objects(_J2000)
    by_id = {o.id: o for o in objects}
    assert set(by_id) == {
        "sun",
        "mercury",
        "venus",
        "earth",
        "mars",
        "jupiter",
        "saturn",
        "uranus",
        "neptune",
    }
    assert by_id["sun"].object_type is CosmicObjectType.STAR
    for planet_id in by_id:
        if planet_id == "sun":
            continue
        assert by_id[planet_id].object_type is CosmicObjectType.PLANET
        assert by_id[planet_id].truth_level is TruthLevel.PHYSICS_SIMULATED
        assert by_id[planet_id].source == "approx_solar_system"
        assert by_id[planet_id].metadata["model"] == "simplified_keplerian"


def test_solar_system_earth_distance_near_one_au():
    earth = next(
        o for o in create_solar_system_objects(_J2000) if o.id == "earth"
    )
    d = math.sqrt(
        earth.position_m.x ** 2
        + earth.position_m.y ** 2
        + earth.position_m.z ** 2
    )
    # Earth's orbit has e≈0.0167 → r ∈ [a(1−e), a(1+e)] ≈ [0.983, 1.017] AU
    assert d == pytest.approx(AU_IN_METERS, rel=0.05)


def test_solar_system_positions_change_over_time():
    earth_now = next(
        o for o in create_solar_system_objects(_J2000) if o.id == "earth"
    )
    earth_later = next(
        o
        for o in create_solar_system_objects(_J2000 + 30.0)
        if o.id == "earth"
    )
    assert earth_now.position_m != earth_later.position_m


def test_solar_system_planet_radii_and_masses_set():
    for obj in create_solar_system_objects(_J2000):
        assert obj.mass_kg is not None and obj.mass_kg > 0.0
        assert obj.radius_m is not None and obj.radius_m > 0.0
