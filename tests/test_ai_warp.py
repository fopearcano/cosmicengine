"""Tests for Phase 5 AI-assisted perception."""

from __future__ import annotations

import math

import pytest

from cosmic_engine.ai import AIWarpModel, SimpleNeuralWarp
from cosmic_engine.core.units import SPEED_OF_LIGHT_M_S
from cosmic_engine.core.vector import Vector3
from cosmic_engine.perception import ObserverState, transform_photon_field
from cosmic_engine.rendering.photon_field import PhotonSample


FORWARD = Vector3(0.0, 1.0, 0.0)
UP = Vector3(0.0, 0.0, 1.0)


def _observer(*, warp_factor: float = 1.0) -> ObserverState:
    return ObserverState(
        position_m=Vector3.zero(),
        velocity_m_s=Vector3(0.0, 0.5 * SPEED_OF_LIGHT_M_S, 0.0),
        forward=FORWARD,
        up=UP,
        warp_factor=warp_factor,
    )


def _unit(v: Vector3) -> Vector3:
    n = math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z)
    return Vector3(v.x / n, v.y / n, v.z / n)


def _sample(direction: Vector3, *, brightness: float = 1.0) -> PhotonSample:
    return PhotonSample(
        object_id="s",
        name="S",
        object_type="star",
        direction=direction,
        distance_m=1.0e16,
        apparent_brightness=brightness,
        color_rgb=(200, 180, 160),
        truth_level="catalog_imported",
    )


# --- base class ---


def test_ai_warp_base_raises_not_implemented():
    model = AIWarpModel()
    obs = _observer()
    with pytest.raises(NotImplementedError):
        model.predict_direction(FORWARD, obs)
    with pytest.raises(NotImplementedError):
        model.predict_brightness(1.0, FORWARD, obs)
    with pytest.raises(NotImplementedError):
        model.predict_color((100, 100, 100), FORWARD, obs)
    with pytest.raises(NotImplementedError):
        model.confidence()


# --- SimpleNeuralWarp primitives ---


def test_simple_neural_warp_direction_is_unit():
    model = SimpleNeuralWarp()
    obs = _observer(warp_factor=25.0)
    for direction in (
        FORWARD,
        Vector3(0.0, -1.0, 0.0),
        _unit(Vector3(0.3, 0.7, -0.5)),
        _unit(Vector3(-0.4, 0.2, 0.9)),
    ):
        out = model.predict_direction(direction, obs)
        mag = math.sqrt(out.x * out.x + out.y * out.y + out.z * out.z)
        assert mag == pytest.approx(1.0, abs=1e-12)


def test_simple_neural_warp_brightness_positive_and_finite():
    model = SimpleNeuralWarp()
    for warp in (1.0, 10.0, 50.0):
        obs = _observer(warp_factor=warp)
        b = model.predict_brightness(1.5, _unit(Vector3(0.2, 0.9, 0.1)), obs)
        assert b > 0.0
        assert math.isfinite(b)


def test_simple_neural_warp_color_bounds():
    model = SimpleNeuralWarp()
    obs = _observer(warp_factor=50.0)
    for color in ((255, 0, 0), (0, 255, 0), (0, 0, 255), (123, 45, 200)):
        r, g, b = model.predict_color(color, FORWARD, obs)
        for c in (r, g, b):
            assert 0 <= c <= 255


def test_simple_neural_warp_confidence_is_constant():
    assert SimpleNeuralWarp().confidence() == 0.5


# --- integration ---


def test_transform_photon_field_none_matches_deterministic():
    obs = _observer(warp_factor=5.0)
    samples = [
        _sample(_unit(Vector3(0.3, 0.8, -0.2))),
        _sample(_unit(Vector3(0.0, -1.0, 0.0)), brightness=2.0),
    ]
    a = transform_photon_field(samples, obs)
    b = transform_photon_field(samples, obs, ai_model=None)
    assert a == b


def test_transform_photon_field_with_ai_changes_output():
    obs = _observer(warp_factor=50.0)
    samples = [_sample(_unit(Vector3(0.3, 0.8, 0.3)))]
    deterministic = transform_photon_field(samples, obs)
    with_ai = transform_photon_field(samples, obs, SimpleNeuralWarp())
    # different path; must produce different sample for a non-trivial input
    assert deterministic[0] != with_ai[0]
    # but core identity fields must still match
    assert with_ai[0].object_id == deterministic[0].object_id
    assert with_ai[0].distance_m == deterministic[0].distance_m


def test_ai_does_not_change_sample_count():
    obs = _observer(warp_factor=10.0)
    samples = [
        _sample(_unit(Vector3(0.3, 0.8, 0.3))),
        _sample(_unit(Vector3(0.1, 0.9, -0.4))),
        _sample(_unit(Vector3(0.0, -1.0, 0.0))),
    ]
    out = transform_photon_field(samples, obs, SimpleNeuralWarp())
    assert len(out) == len(samples)
    for s in out:
        assert s.apparent_brightness >= 0.0
        for c in s.color_rgb:
            assert 0 <= c <= 255


def test_ai_path_is_deterministic():
    obs = _observer(warp_factor=7.0)
    sample = _sample(_unit(Vector3(0.4, 0.6, -0.2)), brightness=2.5)
    model = SimpleNeuralWarp()
    a = transform_photon_field([sample], obs, model)
    b = transform_photon_field([sample], obs, model)
    assert a == b
