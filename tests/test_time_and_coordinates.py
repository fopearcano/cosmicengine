"""Tests for time, units, and coordinate utilities."""

from __future__ import annotations

import math
from datetime import datetime, timezone

import pytest

from cosmic_engine.core.coordinates import (
    cartesian_to_ra_dec_distance,
    distance,
    ra_dec_distance_to_cartesian,
)
from cosmic_engine.core.time import (
    SimulationClock,
    datetime_to_julian,
    julian_to_datetime,
)
from cosmic_engine.core.units import (
    AU_IN_METERS,
    PARSEC_IN_METERS,
    au_to_meters,
    lightyear_to_meters,
    parsec_to_meters,
)
from cosmic_engine.core.vector import Vector3


# --- time ---


def test_clock_tick_advances_julian_date():
    clock = SimulationClock(current_julian_date=2451545.0)
    clock.tick(86400.0)  # one sidereal-irrelevant day of wall time
    assert clock.get_julian_date() == pytest.approx(2451546.0, abs=1e-9)


def test_clock_pause_prevents_advancement():
    clock = SimulationClock(current_julian_date=2451545.0)
    clock.pause()
    clock.tick(86400.0)
    assert clock.get_julian_date() == 2451545.0
    clock.resume()
    clock.tick(86400.0)
    assert clock.get_julian_date() == pytest.approx(2451546.0, abs=1e-9)


def test_clock_time_scale_multiplies_advancement():
    clock = SimulationClock(current_julian_date=0.0)
    clock.set_time_scale(60.0)
    clock.tick(60.0)  # 60 real seconds * 60x = 3600 sim seconds
    assert clock.get_julian_date() == pytest.approx(3600.0 / 86400.0, abs=1e-12)


def test_clock_set_julian_date():
    clock = SimulationClock(current_julian_date=0.0)
    clock.set_julian_date(2451545.0)
    assert clock.get_julian_date() == 2451545.0


def test_julian_round_trip_j2000():
    # J2000.0 epoch: 2000-01-01T12:00:00 UTC == JD 2451545.0
    j2000 = datetime(2000, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    jd = datetime_to_julian(j2000)
    assert jd == pytest.approx(2451545.0, abs=1e-9)
    restored = julian_to_datetime(jd)
    assert restored == j2000


def test_julian_round_trip_arbitrary_datetime():
    dt = datetime(2026, 4, 24, 15, 30, 45, 123456, tzinfo=timezone.utc)
    restored = julian_to_datetime(datetime_to_julian(dt))
    # microsecond precision is within float tolerance at modern JDs
    assert abs((restored - dt).total_seconds()) < 1e-3


def test_datetime_to_julian_treats_naive_as_utc():
    naive = datetime(2000, 1, 1, 12, 0, 0)
    aware = datetime(2000, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    assert datetime_to_julian(naive) == datetime_to_julian(aware)


# --- units ---


def test_au_to_meters_sanity():
    assert au_to_meters(1.0) == AU_IN_METERS
    # Earth-Sun distance is ~1 AU ≈ 1.496e11 m.
    assert au_to_meters(1.0) == pytest.approx(1.496e11, rel=1e-3)
    assert au_to_meters(2.0) == pytest.approx(2.0 * AU_IN_METERS)


def test_parsec_to_meters_sanity():
    assert parsec_to_meters(1.0) == PARSEC_IN_METERS
    # 1 pc ≈ 3.26 ly ≈ 3.086e16 m.
    assert parsec_to_meters(1.0) == pytest.approx(3.086e16, rel=1e-3)
    assert parsec_to_meters(1.0) == pytest.approx(
        lightyear_to_meters(3.2615638), rel=1e-4
    )


# --- coordinates ---


def test_distance_basic():
    assert distance(Vector3.zero(), Vector3(3.0, 4.0, 0.0)) == pytest.approx(5.0)
    assert distance(Vector3(1.0, 2.0, 3.0), Vector3(1.0, 2.0, 3.0)) == 0.0


def test_ra_dec_cartesian_round_trip_axes():
    d = 1.0e16
    eps = d * 1e-12  # tolerance must scale with magnitude for float math
    # RA=0, Dec=0 -> +x
    v = ra_dec_distance_to_cartesian(0.0, 0.0, d)
    assert v.x == pytest.approx(d)
    assert v.y == pytest.approx(0.0, abs=eps)
    assert v.z == pytest.approx(0.0, abs=eps)
    # RA=90, Dec=0 -> +y
    v = ra_dec_distance_to_cartesian(90.0, 0.0, d)
    assert v.x == pytest.approx(0.0, abs=eps)
    assert v.y == pytest.approx(d)
    assert v.z == pytest.approx(0.0, abs=eps)
    # Dec=+90 -> +z
    v = ra_dec_distance_to_cartesian(123.0, 90.0, d)
    assert v.x == pytest.approx(0.0, abs=eps)
    assert v.y == pytest.approx(0.0, abs=eps)
    assert v.z == pytest.approx(d)


@pytest.mark.parametrize(
    "ra_deg, dec_deg, dist_m",
    [
        (0.0, 0.0, 1.0),
        (45.0, 30.0, 1.0e12),
        (180.0, -45.0, 3.086e16),
        (359.5, 89.9, 1.0e10),
        (270.0, 0.0, 2.5e14),
    ],
)
def test_ra_dec_cartesian_full_round_trip(ra_deg, dec_deg, dist_m):
    v = ra_dec_distance_to_cartesian(ra_deg, dec_deg, dist_m)
    ra2, dec2, d2 = cartesian_to_ra_dec_distance(v)
    assert d2 == pytest.approx(dist_m, rel=1e-12)
    assert dec2 == pytest.approx(dec_deg, abs=1e-9)
    assert ra2 == pytest.approx(ra_deg, abs=1e-9)


def test_cartesian_to_ra_dec_normalizes_negative_ra():
    # A point at RA=-90 should come back as 270.
    v = ra_dec_distance_to_cartesian(-90.0, 0.0, 1.0)
    ra, _, _ = cartesian_to_ra_dec_distance(v)
    assert ra == pytest.approx(270.0, abs=1e-9)


def test_cartesian_to_ra_dec_zero_vector():
    ra, dec, d = cartesian_to_ra_dec_distance(Vector3.zero())
    assert (ra, dec, d) == (0.0, 0.0, 0.0)


def test_cartesian_distance_matches_norm():
    v = ra_dec_distance_to_cartesian(42.0, -17.0, 7.5e15)
    assert distance(Vector3.zero(), v) == pytest.approx(7.5e15, rel=1e-12)
    # sanity: vector magnitude equals expected distance via math.hypot
    assert math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z) == pytest.approx(7.5e15)
