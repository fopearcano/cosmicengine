"""Tests for Phase 21 neural photon-space warp."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from ai_viewer import AIViewerConfig, NeuralWarpViewer

from cosmic_engine.ai import ONNXPhotonWarpModel
from cosmic_engine.core.object_types import CosmicObjectType
from cosmic_engine.core.registry import UniverseRegistry
from cosmic_engine.core.truth import TruthLevel
from cosmic_engine.core.units import SPEED_OF_LIGHT_M_S
from cosmic_engine.core.universe_object import UniverseObject
from cosmic_engine.core.vector import Vector3
from cosmic_engine.perception.observer import ObserverState
from cosmic_engine.perception.transform import transform_photon_field
from cosmic_engine.rendering.photon_field import PhotonSample
from cosmic_engine.rendering.simple_camera import SimpleCamera
from cosmic_engine.runtime import CosmicRuntime, RuntimeConfig


_REPO_ROOT = Path(__file__).resolve().parent.parent
_MODEL = _REPO_ROOT / "data" / "photon_warp_model.onnx"


FORWARD = Vector3(0.0, 1.0, 0.0)
UP = Vector3(0.0, 0.0, 1.0)


def _observer(*, warp_factor: float = 5.0) -> ObserverState:
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


def _star(object_id: str, position: Vector3) -> UniverseObject:
    return UniverseObject(
        id=object_id,
        name=object_id,
        object_type=CosmicObjectType.STAR,
        position_m=position,
        velocity_m_s=Vector3.zero(),
        truth_level=TruthLevel.CATALOG_IMPORTED,
        source="gaia",
        spectral_class="G2V",
        metadata={"apparent_magnitude": 1.0},
    )


# --- ONNXPhotonWarpModel: missing-file fallback ---


def test_photon_warp_missing_model_records_error():
    model = ONNXPhotonWarpModel("/no/such/photon_model.onnx")
    assert model.last_error is not None
    assert model.confidence() == pytest.approx(0.3)


def test_photon_warp_missing_model_predicts_via_deterministic_fallback():
    """Predictions must still produce sane output even when the model is absent."""
    model = ONNXPhotonWarpModel("/no/such/photon_model.onnx")
    obs = _observer(warp_factor=5.0)

    direction = _unit(Vector3(0.4, 0.8, -0.2))
    out_dir = model.predict_direction(direction, obs)
    mag = math.sqrt(out_dir.x ** 2 + out_dir.y ** 2 + out_dir.z ** 2)
    assert mag == pytest.approx(1.0, abs=1e-12)

    out_b = model.predict_brightness(2.0, direction, obs)
    assert out_b > 0.0
    assert math.isfinite(out_b)

    out_c = model.predict_color((150, 100, 50), direction, obs)
    for c in out_c:
        assert 0 <= c <= 255


# --- ONNXPhotonWarpModel: real model end-to-end ---


@pytest.mark.skipif(not _MODEL.is_file(), reason="bundled photon warp model missing")
def test_photon_warp_real_model_loads_and_runs():
    model = ONNXPhotonWarpModel(str(_MODEL))
    assert model.last_error is None
    assert model.confidence() == pytest.approx(0.85)

    obs = _observer(warp_factor=5.0)
    direction = _unit(Vector3(0.3, 0.7, -0.5))

    out_dir = model.predict_direction(direction, obs)
    mag = math.sqrt(out_dir.x ** 2 + out_dir.y ** 2 + out_dir.z ** 2)
    assert mag == pytest.approx(1.0, abs=1e-6)

    out_b = model.predict_brightness(1.5, direction, obs)
    assert out_b > 0.0
    assert math.isfinite(out_b)

    r, g, b = model.predict_color((200, 180, 160), direction, obs)
    for c in (r, g, b):
        assert 0 <= c <= 255


@pytest.mark.skipif(not _MODEL.is_file(), reason="bundled photon warp model missing")
def test_photon_warp_real_model_diverges_from_deterministic():
    model = ONNXPhotonWarpModel(str(_MODEL))
    obs = _observer(warp_factor=5.0)
    sample = _sample(_unit(Vector3(0.3, 0.7, -0.2)), brightness=1.5)
    deterministic = transform_photon_field([sample], obs)
    neural = transform_photon_field([sample], obs, ai_model=model)
    assert deterministic[0] != neural[0]
    assert deterministic[0].object_id == neural[0].object_id


# --- transform_photon_field with neural warp ---


def test_neural_warp_preserves_photon_count():
    """Switching between None and a neural model must keep len equal."""
    obs = _observer(warp_factor=10.0)
    samples = [
        _sample(_unit(Vector3(0.3, 0.8, 0.1))),
        _sample(_unit(Vector3(-0.4, 0.7, 0.5)), brightness=2.0),
        _sample(_unit(Vector3(0.0, -1.0, 0.0))),
    ]
    deterministic = transform_photon_field(samples, obs, ai_model=None)
    if _MODEL.is_file():
        model = ONNXPhotonWarpModel(str(_MODEL))
    else:
        model = ONNXPhotonWarpModel("/missing.onnx")
    neural = transform_photon_field(samples, obs, ai_model=model)
    assert len(neural) == len(deterministic) == len(samples)


def test_neural_warp_outputs_stay_within_bounds():
    obs = _observer(warp_factor=50.0)
    samples = [
        _sample(_unit(Vector3(0.3, 0.8, 0.1)), brightness=3.0),
        _sample(_unit(Vector3(0.0, -1.0, 0.0)), brightness=0.5),
        _sample(_unit(Vector3(0.6, 0.2, -0.2)), brightness=1.0),
    ]
    if _MODEL.is_file():
        model = ONNXPhotonWarpModel(str(_MODEL))
    else:
        model = ONNXPhotonWarpModel("/missing.onnx")
    out = transform_photon_field(samples, obs, ai_model=model)
    for s in out:
        mag = math.sqrt(
            s.direction.x ** 2 + s.direction.y ** 2 + s.direction.z ** 2
        )
        assert mag == pytest.approx(1.0, abs=1e-6)
        assert s.apparent_brightness >= 0.0
        for c in s.color_rgb:
            assert 0 <= c <= 255


# --- AIViewerConfig: photon-warp toggles ---


def test_config_use_photon_warp_default_false():
    cfg = AIViewerConfig()
    assert cfg.use_photon_warp is False
    assert cfg.photon_warp_model_path is None
    cfg.validate()


def test_config_use_photon_warp_with_model_path_validates():
    AIViewerConfig(
        use_photon_warp=True, photon_warp_model_path=str(_MODEL)
    ).validate()


def test_config_use_photon_warp_without_model_path_validates_with_fallback():
    """Spec: missing model under use_photon_warp is allowed (fallback warning)."""
    AIViewerConfig(use_photon_warp=True, photon_warp_model_path=None).validate()


# --- NeuralWarpViewer ---


def _make_runtime() -> CosmicRuntime:
    runtime = CosmicRuntime(
        config=RuntimeConfig(
            enable_physics=False,
            physics_backend="none",
            enable_perception=True,
        )
    )
    runtime.add_objects(
        [
            _star("a", Vector3(0.0, 1.0e15, 0.0)),
            _star("b", Vector3(1.0e14, 1.0e15, 0.5e14)),
        ]
    )
    return runtime


def _camera() -> SimpleCamera:
    return SimpleCamera(
        position_m=Vector3.zero(),
        forward=FORWARD,
        up=UP,
        fov_degrees=120.0,
        image_width=64,
        image_height=64,
    )


def test_neural_warp_viewer_deterministic_mode():
    runtime = _make_runtime()
    viewer = NeuralWarpViewer(runtime, _camera(), _observer(), warp_model=None)
    assert viewer.mode == "deterministic"
    assert viewer.confidence() == 1.0
    samples = viewer.run_once(output_path=None)
    assert len(samples) == 2
    assert viewer.last_photon_count == 2


def test_neural_warp_viewer_neural_mode_with_missing_model():
    runtime = _make_runtime()
    model = ONNXPhotonWarpModel("/no/such/photon_model.onnx")
    viewer = NeuralWarpViewer(
        runtime, _camera(), _observer(), warp_model=model
    )
    assert viewer.mode == "neural"
    assert viewer.confidence() == pytest.approx(0.3)
    samples = viewer.run_once(output_path=None)
    assert len(samples) == 2


def test_neural_warp_viewer_writes_ppm(tmp_path: Path):
    runtime = _make_runtime()
    viewer = NeuralWarpViewer(runtime, _camera(), _observer(), warp_model=None)
    out = tmp_path / "out.ppm"
    viewer.run_once(output_path=str(out))
    assert out.is_file()
    assert viewer.last_frame_path == str(out)
    assert viewer.frames_rendered == 1


def test_neural_warp_viewer_run_loop_renders_n_frames(tmp_path: Path):
    runtime = _make_runtime()
    viewer = NeuralWarpViewer(runtime, _camera(), _observer(), warp_model=None)
    pattern = str(tmp_path / "f_{n}.ppm")
    rendered = viewer.run_loop(max_frames=3, output_pattern=pattern)
    assert rendered == 3
    assert viewer.frames_rendered == 3
    for i in range(3):
        assert (tmp_path / f"f_{i}.ppm").is_file()


def test_neural_warp_viewer_empty_registry_does_not_crash():
    runtime = CosmicRuntime(
        config=RuntimeConfig(
            enable_physics=False, enable_perception=True
        )
    )
    viewer = NeuralWarpViewer(runtime, _camera(), _observer(), warp_model=None)
    samples = viewer.run_once(output_path=None)
    assert samples == []
    assert viewer.last_photon_count == 0


# --- AIViewer: use_photon_warp option ---


def _stub_client():
    class _C:
        def receive_message(self): return None
        def connect(self): pass
        def disconnect(self): pass

    return _C()


def test_ai_viewer_use_photon_warp_bypasses_image_postprocess(capsys):
    from ai_viewer import AIViewer
    from ai_viewer.postprocess import FramePostProcessor

    cfg = AIViewerConfig(
        enable_window=False,
        enable_postprocess=True,
        use_photon_warp=True,
        photon_warp_model_path=str(_MODEL) if _MODEL.is_file() else None,
        output_directory="outputs/viewer-test",
    )
    viewer = AIViewer(cfg, _stub_client())
    # postprocessor must be the bare passthrough base, not the deterministic
    # composite (which Phase 19 would otherwise build).
    assert type(viewer.postprocessor) is FramePostProcessor


def test_ai_viewer_use_photon_warp_without_model_path_warns(capsys):
    from ai_viewer import AIViewer
    from ai_viewer.postprocess import FramePostProcessor

    cfg = AIViewerConfig(
        enable_window=False,
        enable_postprocess=True,
        use_photon_warp=True,
        photon_warp_model_path=None,
        output_directory="outputs/viewer-test",
    )
    viewer = AIViewer(cfg, _stub_client())
    captured = capsys.readouterr().out
    assert "use_photon_warp=True but no" in captured
    assert type(viewer.postprocessor) is FramePostProcessor
