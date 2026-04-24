"""Tests for Phase 6 ONNX-backed AI warp model and fallback safety."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from cosmic_engine.ai import (
    AIWarpModel,
    ONNXModelWrapper,
    ONNXWarpModel,
)
from cosmic_engine.core.units import SPEED_OF_LIGHT_M_S
from cosmic_engine.core.vector import Vector3
from cosmic_engine.perception import ObserverState, transform_photon_field
from cosmic_engine.rendering.photon_field import PhotonSample


_REPO_ROOT = Path(__file__).resolve().parent.parent
_MODEL = _REPO_ROOT / "data" / "mock_warp_model.onnx"


FORWARD = Vector3(0.0, 1.0, 0.0)
UP = Vector3(0.0, 0.0, 1.0)


def _unit(v: Vector3) -> Vector3:
    n = math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z)
    return Vector3(v.x / n, v.y / n, v.z / n)


def _observer(*, warp_factor: float = 5.0) -> ObserverState:
    return ObserverState(
        position_m=Vector3.zero(),
        velocity_m_s=Vector3(0.0, 0.5 * SPEED_OF_LIGHT_M_S, 0.0),
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
        color_rgb=(200, 180, 160),
        truth_level="catalog_imported",
    )


# --- ONNXModelWrapper ---


def test_onnx_wrapper_loads_bundled_model():
    wrapper = ONNXModelWrapper(str(_MODEL))
    assert wrapper.input_name
    assert wrapper.output_name


def test_onnx_wrapper_predict_returns_expected_size():
    wrapper = ONNXModelWrapper(str(_MODEL))
    # 9 inputs -> 7 outputs (mock model shape)
    out = wrapper.predict([0.0, 1.0, 0.0, 0.5, 5.0, 1.0, 200.0, 180.0, 160.0])
    assert isinstance(out, list)
    assert len(out) == 7
    for v in out:
        assert math.isfinite(v)


def test_onnx_wrapper_missing_file(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        ONNXModelWrapper(str(tmp_path / "nope.onnx"))


def test_onnx_wrapper_invalid_model(tmp_path: Path):
    bogus = tmp_path / "bad.onnx"
    bogus.write_bytes(b"not an onnx file at all")
    with pytest.raises(ValueError):
        ONNXModelWrapper(str(bogus))


def test_onnx_wrapper_rejects_wrong_length():
    wrapper = ONNXModelWrapper(str(_MODEL))
    with pytest.raises(ValueError):
        wrapper.predict([0.0, 1.0, 0.0])  # only 3 floats, model wants 9


# --- ONNXWarpModel ---


def test_onnx_warp_direction_is_unit_vector():
    model = ONNXWarpModel(str(_MODEL))
    obs = _observer()
    for direction in (
        FORWARD,
        _unit(Vector3(0.3, 0.7, -0.5)),
        _unit(Vector3(-0.4, 0.8, 0.2)),
    ):
        out = model.predict_direction(direction, obs)
        mag = math.sqrt(out.x * out.x + out.y * out.y + out.z * out.z)
        assert mag == pytest.approx(1.0, abs=1e-6)


def test_onnx_warp_brightness_non_negative_and_finite():
    model = ONNXWarpModel(str(_MODEL))
    obs = _observer(warp_factor=50.0)
    for b in (0.1, 1.0, 10.0, 1000.0):
        out = model.predict_brightness(b, _unit(Vector3(0.2, 0.9, 0.1)), obs)
        assert out >= 0.0
        assert math.isfinite(out)


def test_onnx_warp_color_stays_in_bounds():
    model = ONNXWarpModel(str(_MODEL))
    obs = _observer()
    for color in ((0, 0, 0), (255, 255, 255), (200, 100, 50), (30, 200, 180)):
        r, g, b = model.predict_color(color, _unit(Vector3(0.2, 0.9, 0.1)), obs)
        for c in (r, g, b):
            assert 0 <= c <= 255


def test_onnx_warp_confidence():
    assert ONNXWarpModel(str(_MODEL)).confidence() == 0.8


def test_onnx_warp_missing_file_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        ONNXWarpModel(str(tmp_path / "missing.onnx"))


# --- fallback safety ---


class _BrokenModel(AIWarpModel):
    """Model whose predictions always raise; used to exercise fallback."""

    def predict_direction(self, direction, observer):
        raise RuntimeError("boom")

    def predict_brightness(self, brightness, direction, observer):
        raise RuntimeError("boom")

    def predict_color(self, color_rgb, direction, observer):
        raise RuntimeError("boom")

    def confidence(self):
        return 0.0


def test_transform_falls_back_to_deterministic_when_ai_raises():
    obs = _observer()
    samples = [_sample(_unit(Vector3(0.3, 0.8, -0.2)))]
    deterministic = transform_photon_field(samples, obs)
    via_broken = transform_photon_field(samples, obs, _BrokenModel())
    assert via_broken == deterministic


def test_transform_with_onnx_preserves_sample_count():
    obs = _observer()
    samples = [
        _sample(_unit(Vector3(0.3, 0.8, 0.2))),
        _sample(_unit(Vector3(-0.1, 0.9, 0.4)), brightness=3.0),
        _sample(_unit(Vector3(0.0, -1.0, 0.0))),
    ]
    out = transform_photon_field(samples, obs, ONNXWarpModel(str(_MODEL)))
    assert len(out) == len(samples)
    for s in out:
        assert s.apparent_brightness >= 0.0
        for c in s.color_rgb:
            assert 0 <= c <= 255
        mag = math.sqrt(
            s.direction.x ** 2 + s.direction.y ** 2 + s.direction.z ** 2
        )
        assert mag == pytest.approx(1.0, abs=1e-6)


def test_onnx_path_is_deterministic():
    obs = _observer()
    sample = _sample(_unit(Vector3(0.4, 0.6, -0.2)), brightness=2.5)
    model = ONNXWarpModel(str(_MODEL))
    a = transform_photon_field([sample], obs, model)
    b = transform_photon_field([sample], obs, model)
    assert a == b
