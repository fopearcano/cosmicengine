"""Tests for Phase 26 GPU splat pipeline foundation."""

from __future__ import annotations

import numpy as np
import pytest

from ai_viewer import (
    AIViewerConfig,
    CPUSplatFallback,
    GPUDevice,
    GPUGaussianBuffer,
    GaussianPoint,
    GaussianSplatPipeline,
    GaussianSplatRenderer,
)

from cosmic_engine.core.vector import Vector3
from cosmic_engine.rendering.simple_camera import SimpleCamera


def _camera(width: int = 64, height: int = 64) -> SimpleCamera:
    return SimpleCamera(
        position_m=Vector3.zero(),
        forward=Vector3(0.0, 1.0, 0.0),
        up=Vector3(0.0, 0.0, 1.0),
        fov_degrees=120.0,
        image_width=width,
        image_height=height,
    )


def _point(
    *,
    position=(0.0, 1.0e16, 0.0),
    color=(255.0, 200.0, 100.0),
    intensity: float = 1.0,
    sigma: float = 2.0e14,
) -> GaussianPoint:
    return GaussianPoint(
        position=np.asarray(position, dtype=np.float64),
        color=np.asarray(color, dtype=np.float64),
        intensity=intensity,
        sigma=sigma,
    )


# --- GPUDevice ---


def test_gpu_device_default_backend_is_cpu():
    device = GPUDevice()
    assert device.get_backend() == "cpu"
    assert device.is_available() is True


def test_gpu_device_mock_backend_is_available():
    device = GPUDevice(backend="mock_gpu")
    assert device.get_backend() == "mock_gpu"
    assert device.is_available() is True


def test_gpu_device_none_backend_unavailable():
    device = GPUDevice(backend="none")
    assert device.get_backend() == "none"
    assert device.is_available() is False


def test_gpu_device_rejects_unknown_backend():
    with pytest.raises(ValueError):
        GPUDevice(backend="vulkan")


# --- GPUGaussianBuffer ---


def test_buffer_from_empty_list_yields_zero_rows():
    buf = GPUGaussianBuffer.from_gaussian_points([])
    assert len(buf) == 0
    assert buf.positions.shape == (0, 3)
    assert buf.colors.shape == (0, 3)
    assert buf.intensities.shape == (0,)
    assert buf.sigmas.shape == (0,)


def test_buffer_shapes_and_values_correct():
    points = [
        _point(intensity=0.5, sigma=1.0e14),
        _point(position=(1.0, 2.0, 3.0), color=(10.0, 20.0, 30.0), intensity=1.5, sigma=2.0e14),
    ]
    buf = GPUGaussianBuffer.from_gaussian_points(points)
    assert len(buf) == 2
    assert buf.positions.shape == (2, 3)
    assert buf.colors.shape == (2, 3)
    assert buf.intensities.shape == (2,)
    assert buf.sigmas.shape == (2,)
    np.testing.assert_array_equal(buf.positions[1], [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(buf.colors[1], [10.0, 20.0, 30.0])
    assert buf.intensities[1] == pytest.approx(1.5)
    assert buf.sigmas[1] == pytest.approx(2.0e14)


def test_buffer_validate_passes_on_consistent_shapes():
    GPUGaussianBuffer.from_gaussian_points([_point()]).validate()


def test_buffer_validate_rejects_inconsistent_shapes():
    buf = GPUGaussianBuffer.from_gaussian_points([_point()])
    buf.positions = np.zeros((2, 3))
    with pytest.raises(ValueError):
        buf.validate()


def test_buffer_to_numpy_returns_independent_copies():
    points = [_point()]
    buf = GPUGaussianBuffer.from_gaussian_points(points)
    snapshot = buf.to_numpy()
    snapshot["positions"][0] = 999.0
    assert buf.positions[0, 0] != 999.0


# --- CPUSplatFallback ---


def test_cpu_fallback_renders_an_image():
    points = [_point()]
    buf = GPUGaussianBuffer.from_gaussian_points(points)
    image = CPUSplatFallback().render(buf, _camera())
    assert image.shape == (64, 64, 3)
    assert image.dtype == np.uint8
    assert image.max() > 0


def test_cpu_fallback_empty_buffer_returns_black():
    buf = GPUGaussianBuffer.from_gaussian_points([])
    image = CPUSplatFallback().render(buf, _camera())
    assert image.shape == (64, 64, 3)
    assert (image == 0).all()


# --- GaussianSplatPipeline ---


def test_pipeline_returns_correct_shape_for_cpu_backend():
    pipeline = GaussianSplatPipeline(GPUDevice(backend="cpu"))
    image = pipeline.render([_point()], _camera())
    assert image.shape == (64, 64, 3)
    assert image.dtype == np.uint8


def test_pipeline_returns_correct_shape_for_mock_gpu_backend():
    pipeline = GaussianSplatPipeline(GPUDevice(backend="mock_gpu"))
    image = pipeline.render([_point()], _camera())
    assert image.shape == (64, 64, 3)
    assert image.dtype == np.uint8


def test_pipeline_empty_input_returns_black_for_both_backends():
    cpu_image = GaussianSplatPipeline(GPUDevice(backend="cpu")).render([], _camera())
    mock_image = GaussianSplatPipeline(GPUDevice(backend="mock_gpu")).render([], _camera())
    assert cpu_image.shape == mock_image.shape
    assert (cpu_image == 0).all()
    assert (mock_image == 0).all()


def test_pipeline_none_backend_falls_back_silently():
    pipeline = GaussianSplatPipeline(GPUDevice(backend="none"))
    image = pipeline.render([_point()], _camera())
    assert image.shape == (64, 64, 3)


def test_pipeline_does_not_produce_nans():
    points = [
        _point(),
        _point(position=(1.0e15, 1.0e16, -2.0e14), intensity=10.0),
    ]
    for backend in ("cpu", "mock_gpu", "none"):
        image = GaussianSplatPipeline(GPUDevice(backend=backend)).render(
            points, _camera()
        )
        assert np.isfinite(image).all()


def test_pipeline_switching_backends_keeps_output_shape():
    points = [_point()]
    cam = _camera(48, 32)
    cpu_img = GaussianSplatPipeline(GPUDevice(backend="cpu")).render(points, cam)
    mock_img = GaussianSplatPipeline(GPUDevice(backend="mock_gpu")).render(points, cam)
    assert cpu_img.shape == mock_img.shape == (32, 48, 3)


def test_pipeline_mock_gpu_lights_pixels_for_visible_point():
    pipeline = GaussianSplatPipeline(GPUDevice(backend="mock_gpu"))
    image = pipeline.render([_point(intensity=10.0)], _camera())
    assert image.max() > 0


def test_pipeline_mock_gpu_skips_behind_camera():
    pipeline = GaussianSplatPipeline(GPUDevice(backend="mock_gpu"))
    behind = _point(position=(0.0, -1.0e16, 0.0), intensity=10.0)
    image = pipeline.render([behind], _camera())
    assert (image == 0).all()


# --- AIViewerConfig: use_gpu_pipeline ---


def test_config_use_gpu_pipeline_default_is_false():
    cfg = AIViewerConfig()
    assert cfg.use_gpu_pipeline is False
    cfg.validate()


def test_config_use_gpu_pipeline_true_validates():
    AIViewerConfig(use_gpu_pipeline=True).validate()


# --- AIViewer note printing ---


class _StubClient:
    def receive_message(self): return None
    def connect(self): pass
    def disconnect(self): pass


def test_ai_viewer_gaussian_with_gpu_flag_prints_pipeline_info(capsys):
    from ai_viewer import AIViewer

    cfg = AIViewerConfig(
        enable_window=False,
        render_mode="gaussian",
        use_gpu_pipeline=True,
        output_directory="outputs/viewer-test",
    )
    AIViewer(cfg, _StubClient())
    out = capsys.readouterr().out
    assert "GPU mock" in out


def test_ai_viewer_gaussian_without_gpu_flag_prints_cpu(capsys):
    from ai_viewer import AIViewer

    cfg = AIViewerConfig(
        enable_window=False,
        render_mode="gaussian",
        use_gpu_pipeline=False,
        output_directory="outputs/viewer-test",
    )
    AIViewer(cfg, _StubClient())
    out = capsys.readouterr().out
    assert "CPU" in out


# --- end-to-end: pipeline produces same shape as direct CPU renderer ---


def test_pipeline_cpu_matches_direct_renderer_shape():
    points = [
        _point(intensity=1.0),
        _point(position=(1.0e15, 1.5e16, 0.5e15), color=(50.0, 50.0, 200.0)),
    ]
    cam = _camera(48, 48)
    direct = GaussianSplatRenderer(48, 48, cam).render(points)
    via_pipeline = GaussianSplatPipeline(GPUDevice(backend="cpu")).render(points, cam)
    assert direct.shape == via_pipeline.shape
