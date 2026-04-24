"""Tests for Phase 4 perception transforms."""

from __future__ import annotations

import math

import pytest

from cosmic_engine.core.units import SPEED_OF_LIGHT_M_S
from cosmic_engine.core.vector import Vector3
from cosmic_engine.perception import (
    ObserverState,
    apply_brightness_warp,
    apply_color_warp,
    apply_direction_warp,
    transform_photon_field,
    transform_photon_sample,
)
from cosmic_engine.rendering.photon_field import PhotonSample


FORWARD = Vector3(0.0, 1.0, 0.0)
UP = Vector3(0.0, 0.0, 1.0)


def _observer(
    *,
    speed_fraction_c: float = 0.5,
    warp_factor: float = 1.0,
) -> ObserverState:
    return ObserverState(
        position_m=Vector3.zero(),
        velocity_m_s=Vector3(0.0, speed_fraction_c * SPEED_OF_LIGHT_M_S, 0.0),
        forward=FORWARD,
        up=UP,
        warp_factor=warp_factor,
    )


def _sample(direction: Vector3, *, brightness: float = 1.0) -> PhotonSample:
    return PhotonSample(
        object_id="s",
        name="S",
        object_type="star",
        direction=direction,
        distance_m=1.0e16,
        apparent_brightness=brightness,
        color_rgb=(200, 200, 200),
        truth_level="catalog_imported",
    )


# --- observer ---


def test_beta_matches_velocity_over_c():
    obs = _observer(speed_fraction_c=0.25)
    assert obs.beta() == pytest.approx(0.25)
    assert obs.speed_magnitude() == pytest.approx(0.25 * SPEED_OF_LIGHT_M_S)


def test_observer_zero_velocity_gives_zero_beta():
    obs = ObserverState(
        position_m=Vector3.zero(),
        velocity_m_s=Vector3.zero(),
        forward=FORWARD,
        up=UP,
    )
    assert obs.beta() == 0.0
    obs.validate()


@pytest.mark.parametrize(
    "overrides",
    [
        {"warp_factor": 0.5},
        {"warp_factor": -1.0},
        {"velocity_m_s": Vector3(SPEED_OF_LIGHT_M_S, 0.0, 0.0)},
        {"velocity_m_s": Vector3(2.0 * SPEED_OF_LIGHT_M_S, 0.0, 0.0)},
        {"forward": Vector3.zero()},
        {"up": Vector3.zero()},
    ],
)
def test_observer_validate_rejects_bad_fields(overrides):
    obs = _observer()
    for k, v in overrides.items():
        setattr(obs, k, v)
    with pytest.raises(ValueError):
        obs.validate()


# --- direction warp ---


def test_direction_warp_returns_unit_vector():
    obs = _observer(warp_factor=10.0)
    for direction in (
        Vector3(0.0, 1.0, 0.0),
        Vector3(1.0, 0.0, 0.0),
        Vector3(0.0, -1.0, 0.0),
        _normalize(Vector3(0.6, 0.8, 0.0)),
        _normalize(Vector3(0.3, 0.7, -0.5)),
    ):
        warped = apply_direction_warp(direction, obs)
        mag = math.sqrt(warped.x ** 2 + warped.y ** 2 + warped.z ** 2)
        assert mag == pytest.approx(1.0, abs=1e-12)


def test_direction_warp_zero_when_no_motion_and_unit_warp():
    obs = _observer(speed_fraction_c=0.0, warp_factor=1.0)
    d = _normalize(Vector3(0.6, 0.8, 0.0))
    assert apply_direction_warp(d, obs) == d


def test_direction_warp_pulls_side_direction_toward_forward():
    obs = _observer(speed_fraction_c=0.5, warp_factor=10.0)
    side = _normalize(Vector3(0.6, 0.8, 0.0))
    before_align = side.y  # forward is +y
    warped = apply_direction_warp(side, obs)
    assert warped.y > before_align  # more aligned with forward


# --- brightness warp ---


def test_brightness_warp_forward_direction_brighter():
    obs = _observer(speed_fraction_c=0.5, warp_factor=1.0)
    b = apply_brightness_warp(1.0, FORWARD, obs)
    assert b > 1.0


def test_brightness_warp_backward_direction_dimmer():
    obs = _observer(speed_fraction_c=0.5, warp_factor=1.0)
    back = Vector3(0.0, -1.0, 0.0)
    b = apply_brightness_warp(1.0, back, obs)
    assert 0.0 <= b < 1.0


def test_brightness_always_non_negative_and_finite():
    obs_hi = _observer(speed_fraction_c=0.9, warp_factor=50.0)
    obs_lo = _observer(speed_fraction_c=0.01, warp_factor=1.0)
    directions = [
        FORWARD,
        Vector3(0.0, -1.0, 0.0),
        Vector3(1.0, 0.0, 0.0),
        _normalize(Vector3(0.3, 0.7, -0.5)),
    ]
    for obs in (obs_hi, obs_lo):
        for d in directions:
            b = apply_brightness_warp(1.0, d, obs)
            assert b >= 0.0
            assert math.isfinite(b)


# --- color warp ---


def test_color_warp_bounds_preserved():
    obs = _observer(speed_fraction_c=0.9, warp_factor=50.0)
    for direction in (
        FORWARD,
        Vector3(0.0, -1.0, 0.0),
        _normalize(Vector3(0.5, 0.5, 0.5)),
        _normalize(Vector3(-0.3, 0.4, 0.8)),
    ):
        r, g, b = apply_color_warp((128, 128, 128), direction, obs)
        for c in (r, g, b):
            assert 0 <= c <= 255


def test_color_warp_forward_shifts_toward_blue():
    obs = _observer(speed_fraction_c=0.5, warp_factor=5.0)
    r, _, b = apply_color_warp((200, 200, 200), FORWARD, obs)
    assert b >= 200 and r <= 200


def test_color_warp_backward_shifts_toward_red():
    obs = _observer(speed_fraction_c=0.5, warp_factor=5.0)
    back = Vector3(0.0, -1.0, 0.0)
    r, _, b = apply_color_warp((200, 200, 200), back, obs)
    assert r >= 200 and b <= 200


def test_color_warp_noop_when_no_shift():
    obs = _observer(speed_fraction_c=0.0, warp_factor=1.0)
    assert apply_color_warp((200, 200, 200), FORWARD, obs) == (200, 200, 200)


# --- full transform ---


def test_transform_photon_sample_preserves_identity_fields():
    obs = _observer(warp_factor=5.0)
    s = _sample(_normalize(Vector3(0.5, 0.7, 0.3)))
    out = transform_photon_sample(s, obs)
    assert out.object_id == s.object_id
    assert out.name == s.name
    assert out.object_type == s.object_type
    assert out.distance_m == s.distance_m
    assert out.truth_level == s.truth_level


def test_transform_photon_field_preserves_count():
    obs = _observer(speed_fraction_c=0.7, warp_factor=10.0)
    samples = [
        _sample(FORWARD),
        _sample(Vector3(0.0, -1.0, 0.0)),
        _sample(_normalize(Vector3(1.0, 1.0, 0.0))),
    ]
    out = transform_photon_field(samples, obs)
    assert len(out) == 3
    for before, after in zip(samples, out):
        assert after.object_id == before.object_id
        assert after.apparent_brightness >= 0.0
        for c in after.color_rgb:
            assert 0 <= c <= 255


def test_transform_is_deterministic():
    obs = _observer(warp_factor=7.0)
    s = _sample(_normalize(Vector3(0.4, 0.6, -0.2)), brightness=2.5)
    a = transform_photon_sample(s, obs)
    b = transform_photon_sample(s, obs)
    assert a == b


def _normalize(v: Vector3) -> Vector3:
    n = math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z)
    return Vector3(v.x / n, v.y / n, v.z / n)
