"""Tests for Phase 30 neural spacetime field."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from ai_viewer import (
    AIViewerConfig,
    BlackHole,
    GeodesicRayMarcher,
    GaussianPoint,
    GaussianSplatRenderer,
    integrate_geodesic_step_neural,
)

from cosmic_engine.ai import (
    ONNXSpacetimeField,
    SpacetimeFieldModel,
)
from cosmic_engine.core.vector import Vector3
from cosmic_engine.rendering.simple_camera import SimpleCamera


_REPO_ROOT = Path(__file__).resolve().parent.parent
_MODEL = _REPO_ROOT / "data" / "spacetime_field_identity.onnx"


def _camera() -> SimpleCamera:
    return SimpleCamera(
        position_m=Vector3.zero(),
        forward=Vector3(0.0, 1.0, 0.0),
        up=Vector3(0.0, 0.0, 1.0),
        fov_degrees=90.0,
        image_width=64,
        image_height=64,
    )


# --- SpacetimeFieldModel base class ---


def test_base_class_methods_raise_not_implemented():
    model = SpacetimeFieldModel()
    with pytest.raises(NotImplementedError):
        model.query_acceleration(np.zeros(3))
    with pytest.raises(NotImplementedError):
        model.confidence()


# --- ONNXSpacetimeField loading ---


def test_onnx_spacetime_field_missing_file_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        ONNXSpacetimeField(str(tmp_path / "nope.onnx"))


def test_onnx_spacetime_field_invalid_model(tmp_path: Path):
    bogus = tmp_path / "bad.onnx"
    bogus.write_bytes(b"not an onnx file")
    with pytest.raises(ValueError):
        ONNXSpacetimeField(str(bogus))


@pytest.mark.skipif(
    not _MODEL.is_file(), reason="bundled spacetime model missing"
)
def test_onnx_spacetime_field_query_returns_3_vector():
    model = ONNXSpacetimeField(str(_MODEL))
    out = model.query_acceleration(np.array([1.0e10, 2.0e10, 0.0]))
    assert out.shape == (3,)
    assert np.isfinite(out).all()


@pytest.mark.skipif(
    not _MODEL.is_file(), reason="bundled spacetime model missing"
)
def test_onnx_spacetime_field_pulls_toward_origin():
    """The bundled identity model returns a = -k * position; verify direction."""
    model = ONNXSpacetimeField(str(_MODEL))
    out = model.query_acceleration(np.array([1.0e10, 0.0, 0.0]))
    assert out[0] < 0.0  # toward origin from +x
    assert abs(out[1]) < 1e-6
    assert abs(out[2]) < 1e-6


@pytest.mark.skipif(
    not _MODEL.is_file(), reason="bundled spacetime model missing"
)
def test_onnx_spacetime_field_handles_optional_direction():
    model = ONNXSpacetimeField(str(_MODEL))
    pos = np.array([1.0e10, 1.0e10, 0.0])
    a_no_dir = model.query_acceleration(pos)
    a_with_dir = model.query_acceleration(pos, direction=np.array([0.0, 1.0, 0.0]))
    # The bundled model ignores direction, so results should match.
    np.testing.assert_allclose(a_no_dir, a_with_dir)


@pytest.mark.skipif(
    not _MODEL.is_file(), reason="bundled spacetime model missing"
)
def test_onnx_spacetime_field_no_nans():
    model = ONNXSpacetimeField(str(_MODEL))
    for pos in (
        np.array([0.0, 0.0, 0.0]),
        np.array([1.0e22, 0.0, 0.0]),
        np.array([1.0e8, 1.0e8, 1.0e8]),
    ):
        out = model.query_acceleration(pos)
        assert np.isfinite(out).all()


@pytest.mark.skipif(
    not _MODEL.is_file(), reason="bundled spacetime model missing"
)
def test_onnx_spacetime_field_confidence_high_after_clean_run():
    model = ONNXSpacetimeField(str(_MODEL))
    model.query_acceleration(np.array([1.0e10, 0.0, 0.0]))
    assert model.confidence() == pytest.approx(0.9)


# --- integrate_geodesic_step_neural ---


def test_integrate_neural_with_none_model_uses_analytical():
    """No model -> analytical Schwarzschild path."""
    pos = np.array([1.0e22, 0.0, 0.0])
    direction = np.array([0.0, 1.0, 0.0])
    new_pos, new_dir = integrate_geodesic_step_neural(
        pos, direction, step_size=1.0e22, model=None,
        fallback_mass_kg=1.0e50,
    )
    assert math.isclose(float(np.linalg.norm(new_dir)), 1.0, abs_tol=1e-9)
    # The position must advance by step_size along the (slightly bent)
    # direction, so the step must be at most step_size from the start.
    delta = float(np.linalg.norm(new_pos - pos))
    assert delta == pytest.approx(1.0e22, rel=1e-6)


@pytest.mark.skipif(
    not _MODEL.is_file(), reason="bundled spacetime model missing"
)
def test_integrate_neural_with_model_returns_unit_direction():
    model = ONNXSpacetimeField(str(_MODEL))
    pos = np.array([1.0e22, 0.0, 0.0])
    direction = np.array([0.0, 1.0, 0.0])
    new_pos, new_dir = integrate_geodesic_step_neural(
        pos, direction, step_size=1.0e22, model=model, fallback_mass_kg=1.0e50
    )
    assert math.isclose(float(np.linalg.norm(new_dir)), 1.0, abs_tol=1e-9)
    assert np.isfinite(new_pos).all()
    assert np.isfinite(new_dir).all()


def test_integrate_neural_falls_back_when_model_raises():
    """When the model raises, fall back silently to the analytical path."""

    class _BadModel(SpacetimeFieldModel):
        def query_acceleration(self, position, direction=None):
            raise RuntimeError("boom")

        def confidence(self) -> float:
            return 0.0

    pos = np.array([1.0e22, 0.0, 0.0])
    direction = np.array([0.0, 1.0, 0.0])
    new_pos_bad, new_dir_bad = integrate_geodesic_step_neural(
        pos, direction, step_size=1.0e22, model=_BadModel(),
        fallback_mass_kg=1.0e50,
    )
    new_pos_ana, new_dir_ana = integrate_geodesic_step_neural(
        pos, direction, step_size=1.0e22, model=None,
        fallback_mass_kg=1.0e50,
    )
    # Bad-model result must equal the analytical result.
    np.testing.assert_array_equal(new_pos_bad, new_pos_ana)
    np.testing.assert_array_equal(new_dir_bad, new_dir_ana)


def test_integrate_neural_falls_back_on_nan_output():
    """A model returning NaN should be ignored too."""

    class _NaNModel(SpacetimeFieldModel):
        def query_acceleration(self, position, direction=None):
            return np.array([float("nan"), 0.0, 0.0])

        def confidence(self) -> float:
            return 0.0

    pos = np.array([1.0e22, 0.0, 0.0])
    direction = np.array([0.0, 1.0, 0.0])
    new_pos, new_dir = integrate_geodesic_step_neural(
        pos, direction, step_size=1.0e22, model=_NaNModel(),
        fallback_mass_kg=1.0e50,
    )
    assert np.isfinite(new_pos).all()
    assert np.isfinite(new_dir).all()


def test_integrate_neural_zero_direction_passes_through():
    new_pos, new_dir = integrate_geodesic_step_neural(
        np.array([1.0, 2.0, 3.0]),
        np.zeros(3),
        step_size=1.0e10,
        model=None,
        fallback_mass_kg=1.0e36,
    )
    np.testing.assert_array_equal(new_pos, np.array([1.0, 2.0, 3.0]))
    np.testing.assert_array_equal(new_dir, np.zeros(3))


def test_integrate_neural_rejects_non_positive_step_size():
    with pytest.raises(ValueError):
        integrate_geodesic_step_neural(
            np.zeros(3),
            np.array([0.0, 1.0, 0.0]),
            step_size=0.0,
            model=None,
            fallback_mass_kg=1.0e36,
        )


# --- GeodesicRayMarcher with spacetime_model ---


def test_marcher_accepts_spacetime_model_argument():
    bh = BlackHole((0.0, 0.0, 0.0), 1.0e36)
    if _MODEL.is_file():
        model = ONNXSpacetimeField(str(_MODEL))
    else:
        model = None
    marcher = GeodesicRayMarcher(
        bh, step_size=1.0e10, max_steps=4, spacetime_model=model
    )
    assert marcher.spacetime_model is model


def test_marcher_with_failing_model_still_terminates():
    bh = BlackHole((0.0, 0.0, 0.0), 1.0e36)

    class _BadModel(SpacetimeFieldModel):
        def query_acceleration(self, position, direction=None):
            raise RuntimeError("bad")

        def confidence(self) -> float:
            return 0.0

    marcher = GeodesicRayMarcher(
        bh, step_size=1.0e10, max_steps=4, spacetime_model=_BadModel()
    )
    direction, absorbed = marcher.trace_ray(
        origin=np.array([0.0, -1.0e22, 0.0]),
        direction=np.array([0.0, 1.0, 0.0]),
    )
    assert absorbed is False
    assert math.isclose(float(np.linalg.norm(direction)), 1.0, abs_tol=1e-6)


# --- AIViewerConfig new fields ---


def test_config_neural_spacetime_defaults():
    cfg = AIViewerConfig()
    assert cfg.use_neural_spacetime is False
    assert cfg.spacetime_model_path is None
    cfg.validate()


def test_config_neural_spacetime_enabled_validates_with_path():
    AIViewerConfig(
        use_neural_spacetime=True, spacetime_model_path="/nonexistent.onnx"
    ).validate()


def test_config_neural_spacetime_enabled_without_path_validates():
    """Spec: missing model under use_neural_spacetime is allowed (fallback)."""
    AIViewerConfig(
        use_neural_spacetime=True, spacetime_model_path=None
    ).validate()


# --- WebGPU renderer state ---


def test_webgpu_renderer_neural_field_flag_state():
    from ai_viewer.neural_field.gpu import (
        WebGPUDevice,
        WebGPUSplatRenderer,
    )

    device = WebGPUDevice(backend="cpu")  # no real GPU; just exercise state
    renderer = WebGPUSplatRenderer(device, 32, 32)
    assert renderer._use_neural_field is False
    renderer.enable_gr_effects(
        BlackHole((0.0, 0.0, 0.0), 4.0e36),
        gr_mode="geodesic",
        use_neural_field=True,
        neural_confidence=0.9,
    )
    assert renderer._use_neural_field is True
    assert renderer._neural_confidence == pytest.approx(0.9)
    renderer.enable_gr_effects(black_hole=None, gr_mode="none")
    assert renderer._use_neural_field is False


# --- end-to-end smoke test ---


def test_renderer_does_not_crash_with_neural_marcher():
    bh = BlackHole((0.0, 0.0, 0.0), 1.0e50)
    if _MODEL.is_file():
        model = ONNXSpacetimeField(str(_MODEL))
    else:
        model = None
    marcher = GeodesicRayMarcher(
        bh, step_size=1.0e22, max_steps=8, spacetime_model=model
    )
    base = [
        GaussianPoint(
            position=np.array([2.0e15, 1.0e16, 0.0]),
            color=np.array([255.0, 200.0, 100.0]),
            intensity=1.0,
            sigma=2.0e14,
        )
    ]
    from ai_viewer.neural_field import trace_points_through_geodesic

    out = trace_points_through_geodesic(
        base, marcher, observer_position=np.array([0.0, -1.0e16, 0.0])
    )
    image = GaussianSplatRenderer(64, 64, _camera()).render(out)
    assert image.shape == (64, 64, 3)
    assert np.isfinite(image).all()
