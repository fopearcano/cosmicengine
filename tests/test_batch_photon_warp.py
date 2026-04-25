"""Tests for Phase 22 batch ONNX photon warp."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from cosmic_engine.ai import BatchONNXPhotonWarpModel
from cosmic_engine.core.coordinates import ra_dec_distance_to_cartesian
from cosmic_engine.core.object_types import CosmicObjectType
from cosmic_engine.core.truth import TruthLevel
from cosmic_engine.core.units import LIGHTYEAR_IN_METERS, SPEED_OF_LIGHT_M_S
from cosmic_engine.core.universe_object import UniverseObject
from cosmic_engine.core.vector import Vector3
from cosmic_engine.perception.observer import ObserverState
from cosmic_engine.perception.vectorized_ai_transform import (
    apply_batch_ai_warp,
    build_photon_warp_input,
    validate_warp_output,
)
from cosmic_engine.perception.vectorized_transform import (
    transform_photon_field_batch,
)
from cosmic_engine.rendering import SimpleCamera, build_star_photon_field_batch
from cosmic_engine.rendering.vectorized_photon_field import PhotonFieldBatch


_REPO_ROOT = Path(__file__).resolve().parent.parent
_MODEL = _REPO_ROOT / "data" / "photon_warp_model.onnx"


def _camera() -> SimpleCamera:
    return SimpleCamera(
        position_m=Vector3.zero(),
        forward=Vector3(0.0, 1.0, 0.0),
        up=Vector3(0.0, 0.0, 1.0),
        fov_degrees=140.0,
        image_width=64,
        image_height=64,
    )


def _observer(*, warp_factor: float = 5.0) -> ObserverState:
    return ObserverState(
        position_m=Vector3.zero(),
        velocity_m_s=Vector3(0.0, 0.5 * SPEED_OF_LIGHT_M_S, 0.0),
        forward=Vector3(0.0, 1.0, 0.0),
        up=Vector3(0.0, 0.0, 1.0),
        warp_factor=warp_factor,
    )


def _star(i: int) -> UniverseObject:
    return UniverseObject(
        id=f"s{i}",
        name=f"S{i}",
        object_type=CosmicObjectType.STAR,
        position_m=ra_dec_distance_to_cartesian(
            (i * 37.9) % 360.0,
            ((i * 73.1) % 180.0) - 90.0,
            (10.0 + i) * LIGHTYEAR_IN_METERS,
        ),
        velocity_m_s=Vector3.zero(),
        truth_level=TruthLevel.CATALOG_IMPORTED,
        source="test",
        spectral_class="G2V",
        metadata={"apparent_magnitude": float(i % 5)},
    )


def _make_batch(n: int = 8) -> PhotonFieldBatch:
    return build_star_photon_field_batch([_star(i) for i in range(n)], _camera())


def _empty_batch() -> PhotonFieldBatch:
    return build_star_photon_field_batch([], _camera())


# --- build_photon_warp_input ---


def test_build_photon_warp_input_shape_is_n_by_9():
    batch = _make_batch(5)
    feed = build_photon_warp_input(batch, _observer())
    assert feed.shape == (5, 9)
    assert feed.dtype == np.float32


def test_build_photon_warp_input_columns_match_batch():
    batch = _make_batch(3)
    obs = _observer(warp_factor=7.0)
    feed = build_photon_warp_input(batch, obs)
    np.testing.assert_allclose(feed[:, 0:3], batch.directions, atol=1e-6)
    np.testing.assert_allclose(feed[:, 3], batch.brightness, atol=1e-6)
    np.testing.assert_allclose(feed[:, 4:7], batch.colors_rgb, atol=1e-6)
    np.testing.assert_allclose(feed[:, 7], obs.beta(), atol=1e-6)
    np.testing.assert_allclose(feed[:, 8], 7.0, atol=1e-6)


def test_build_photon_warp_input_empty_batch_yields_zero_rows():
    feed = build_photon_warp_input(_empty_batch(), _observer())
    assert feed.shape == (0, 9)


# --- validate_warp_output ---


def test_validate_warp_output_passes_correct_shape():
    arr = np.zeros((3, 7), dtype=np.float32)
    out = validate_warp_output(arr, expected_count=3)
    assert out is arr


def test_validate_warp_output_rejects_wrong_columns():
    with pytest.raises(ValueError):
        validate_warp_output(np.zeros((3, 5), dtype=np.float32), 3)


def test_validate_warp_output_rejects_wrong_row_count():
    with pytest.raises(ValueError):
        validate_warp_output(np.zeros((3, 7), dtype=np.float32), expected_count=4)


def test_validate_warp_output_rejects_wrong_ndim():
    with pytest.raises(ValueError):
        validate_warp_output(np.zeros((7,), dtype=np.float32), 1)


# --- BatchONNXPhotonWarpModel ---


def test_batch_model_construction_raises_on_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        BatchONNXPhotonWarpModel(str(tmp_path / "nope.onnx"))


def test_batch_model_construction_raises_on_invalid_file(tmp_path):
    bogus = tmp_path / "bad.onnx"
    bogus.write_bytes(b"not an onnx file at all")
    with pytest.raises(ValueError):
        BatchONNXPhotonWarpModel(str(bogus))


@pytest.mark.skipif(not _MODEL.is_file(), reason="bundled photon warp model missing")
def test_batch_model_predict_returns_n_by_7():
    model = BatchONNXPhotonWarpModel(str(_MODEL))
    arr = np.random.default_rng(42).random((6, 9)).astype(np.float32)
    out = model.predict_batch(arr)
    assert out.shape == (6, 7)
    # directions normalized
    norms = np.linalg.norm(out[:, 0:3], axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-6)
    # brightness non-negative
    assert (out[:, 3] >= 0.0).all()
    # color clamped
    assert (out[:, 4:7] >= 0.0).all()
    assert (out[:, 4:7] <= 255.0).all()


@pytest.mark.skipif(not _MODEL.is_file(), reason="bundled photon warp model missing")
def test_batch_model_empty_batch_returns_zero_rows():
    model = BatchONNXPhotonWarpModel(str(_MODEL))
    out = model.predict_batch(np.zeros((0, 9), dtype=np.float32))
    assert out.shape == (0, 7)


@pytest.mark.skipif(not _MODEL.is_file(), reason="bundled photon warp model missing")
def test_batch_model_rejects_bad_input_shape():
    model = BatchONNXPhotonWarpModel(str(_MODEL))
    with pytest.raises(ValueError):
        model.predict_batch(np.zeros((4, 8), dtype=np.float32))
    with pytest.raises(ValueError):
        model.predict_batch(np.zeros((4,), dtype=np.float32))


@pytest.mark.skipif(not _MODEL.is_file(), reason="bundled photon warp model missing")
def test_batch_model_confidence_is_high_on_clean_run():
    model = BatchONNXPhotonWarpModel(str(_MODEL))
    arr = np.random.default_rng(1).random((4, 9)).astype(np.float32)
    model.predict_batch(arr)
    assert model.confidence() == pytest.approx(0.85)


# --- apply_batch_ai_warp integration ---


def test_apply_batch_ai_warp_none_model_uses_deterministic():
    batch = _make_batch(4)
    obs = _observer(warp_factor=3.0)
    deterministic = transform_photon_field_batch(batch, obs)
    out = apply_batch_ai_warp(batch, obs, model=None)
    np.testing.assert_array_equal(out.directions, deterministic.directions)
    np.testing.assert_array_equal(out.brightness, deterministic.brightness)
    np.testing.assert_array_equal(out.colors_rgb, deterministic.colors_rgb)


def test_apply_batch_ai_warp_empty_batch_is_safe():
    batch = _empty_batch()
    out = apply_batch_ai_warp(batch, _observer(), model=None)
    assert len(out) == 0
    out_ai = apply_batch_ai_warp(
        batch, _observer(), model=_FailingModel()
    )
    assert len(out_ai) == 0


class _FailingModel:
    """Mock model whose predict_batch always raises; tests fallback path."""

    last_error = None

    def predict_batch(self, _arr: np.ndarray) -> np.ndarray:
        raise RuntimeError("simulated inference failure")

    def confidence(self) -> float:
        return 0.3


def test_apply_batch_ai_warp_falls_back_when_model_raises():
    batch = _make_batch(5)
    obs = _observer(warp_factor=3.0)
    deterministic = transform_photon_field_batch(batch, obs)
    out = apply_batch_ai_warp(batch, obs, model=_FailingModel())
    np.testing.assert_array_equal(out.directions, deterministic.directions)
    np.testing.assert_array_equal(out.brightness, deterministic.brightness)
    np.testing.assert_array_equal(out.colors_rgb, deterministic.colors_rgb)


class _WrongShapeModel:
    """Returns the wrong number of columns; integration must catch that."""

    last_error = None

    def predict_batch(self, arr: np.ndarray) -> np.ndarray:
        return np.zeros((arr.shape[0], 5), dtype=np.float32)

    def confidence(self) -> float:
        return 0.3


def test_apply_batch_ai_warp_falls_back_on_wrong_output_shape():
    batch = _make_batch(3)
    obs = _observer()
    deterministic = transform_photon_field_batch(batch, obs)
    out = apply_batch_ai_warp(batch, obs, model=_WrongShapeModel())
    np.testing.assert_array_equal(out.directions, deterministic.directions)


def test_apply_batch_ai_warp_preserves_object_count_and_identity():
    batch = _make_batch(6)
    obs = _observer(warp_factor=2.0)
    out = apply_batch_ai_warp(batch, obs, model=_FailingModel())
    assert len(out) == len(batch)
    assert out.object_ids == batch.object_ids
    assert out.names == batch.names
    assert out.object_types == batch.object_types
    assert out.truth_levels == batch.truth_levels
    np.testing.assert_array_equal(out.distances_m, batch.distances_m)


@pytest.mark.skipif(not _MODEL.is_file(), reason="bundled photon warp model missing")
def test_apply_batch_ai_warp_with_real_model_outputs_are_valid():
    batch = _make_batch(8)
    obs = _observer(warp_factor=10.0)
    model = BatchONNXPhotonWarpModel(str(_MODEL))
    out = apply_batch_ai_warp(batch, obs, model=model)
    assert len(out) == len(batch)
    norms = np.linalg.norm(out.directions, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-6)
    assert (out.brightness >= 0.0).all()
    assert ((out.colors_rgb >= 0.0) & (out.colors_rgb <= 255.0)).all()


# --- viewer integration ---


def test_neural_warp_viewer_batch_mode_runs():
    from ai_viewer import NeuralWarpViewer

    from cosmic_engine.runtime import CosmicRuntime, RuntimeConfig

    runtime = CosmicRuntime(
        config=RuntimeConfig(
            enable_physics=False, enable_perception=True
        )
    )
    runtime.add_objects([_star(i) for i in range(20)])

    viewer = NeuralWarpViewer(
        runtime,
        _camera(),
        _observer(),
        warp_model=None,
        batch_warp_model=_FailingModel(),
    )
    assert viewer.mode == "batch_neural"
    samples = viewer.run_once(output_path=None)
    # batch fallback runs the deterministic vectorized path; viewer should
    # still produce one sample per star (galaxies absent here).
    assert len(samples) == 20
