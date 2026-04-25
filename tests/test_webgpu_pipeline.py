"""Tests for Phase 27 WebGPU splat pipeline.

Tests must pass on machines without a GPU. The fallback path is
deliberately the most-tested route here.
"""

from __future__ import annotations

import numpy as np
import pytest

from ai_viewer import (
    AIViewerConfig,
    GPUDevice,
    GaussianPoint,
    GaussianSplatPipeline,
)
from ai_viewer.neural_field.gpu import (
    WebGPUDevice,
    WebGPUGaussianBuffer,
    WebGPUSplatRenderer,
)
from ai_viewer.neural_field.gpu.webgpu_buffer import (
    BYTES_PER_POINT,
    FLOATS_PER_POINT,
    pack_points,
)

from cosmic_engine.core.units import SPEED_OF_LIGHT_M_S
from cosmic_engine.core.vector import Vector3
from cosmic_engine.perception.observer import ObserverState
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


def _point() -> GaussianPoint:
    return GaussianPoint(
        position=np.array([0.0, 1.0e16, 0.0]),
        color=np.array([255.0, 200.0, 100.0]),
        intensity=1.0,
        sigma=2.0e14,
    )


def _observer() -> ObserverState:
    return ObserverState(
        position_m=Vector3.zero(),
        velocity_m_s=Vector3(0.0, 0.5 * SPEED_OF_LIGHT_M_S, 0.0),
        forward=Vector3(0.0, 1.0, 0.0),
        up=Vector3(0.0, 0.0, 1.0),
        warp_factor=2.0,
    )


# --- WebGPUDevice ---


def test_webgpu_device_initializes_or_falls_back_gracefully():
    """On machines without a GPU, the device must downgrade to 'cpu'."""
    device = WebGPUDevice()
    assert device.get_backend() in ("webgpu", "cpu")
    assert device.is_available() is True
    if device.get_backend() == "cpu":
        # The downgrade path must record an explanation.
        assert device.last_error is not None


def test_webgpu_device_explicit_cpu_backend_does_not_initialize():
    device = WebGPUDevice(backend="cpu")
    assert device.get_backend() == "cpu"
    assert device.adapter is None
    assert device.device is None


def test_webgpu_device_is_a_gpu_device_subclass():
    assert issubclass(WebGPUDevice, GPUDevice)


# --- buffer packing ---


def test_pack_points_layout_is_eight_floats_per_point():
    points = [
        GaussianPoint(
            position=np.array([1.0, 2.0, 3.0]),
            color=np.array([10.0, 20.0, 30.0]),
            intensity=4.0,
            sigma=5.0,
        ),
        GaussianPoint(
            position=np.array([6.0, 7.0, 8.0]),
            color=np.array([40.0, 50.0, 60.0]),
            intensity=9.0,
            sigma=10.0,
        ),
    ]
    packed = pack_points(points)
    assert packed.shape == (2 * FLOATS_PER_POINT,)
    assert packed.dtype == np.float32
    # Row 0: pos.xyz, intensity, color.rgb, sigma
    np.testing.assert_array_equal(
        packed[:FLOATS_PER_POINT],
        np.array([1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 5.0], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        packed[FLOATS_PER_POINT:],
        np.array([6.0, 7.0, 8.0, 9.0, 40.0, 50.0, 60.0, 10.0], dtype=np.float32),
    )


def test_pack_points_empty_returns_empty_array():
    packed = pack_points([])
    assert packed.shape == (0,)
    assert packed.dtype == np.float32


def test_bytes_per_point_constant_matches_layout():
    assert BYTES_PER_POINT == FLOATS_PER_POINT * 4 == 32


def test_webgpu_buffer_upload_sets_count_and_cpu_copy():
    device = WebGPUDevice(backend="cpu")  # no real device → CPU shadow only
    buf = WebGPUGaussianBuffer()
    buf.upload([_point(), _point()], device)
    assert buf.count == 2
    cpu = buf.cpu_copy()
    assert cpu.shape == (2 * FLOATS_PER_POINT,)
    assert cpu.dtype == np.float32


def test_webgpu_buffer_release_idempotent():
    device = WebGPUDevice(backend="cpu")
    buf = WebGPUGaussianBuffer()
    buf.upload([_point()], device)
    buf.release()
    buf.release()  # second call must not raise
    assert buf.gpu_buffer is None
    assert buf.count == 0


# --- pipeline integration ---


def test_pipeline_with_webgpu_device_returns_valid_image():
    """End-to-end pipeline with a WebGPU device.

    On a GPU-less host the WebGPUDevice downgrades to ``cpu`` at
    construction, so the pipeline routes through CPUSplatFallback —
    the call must still return a sensible image.
    """
    device = WebGPUDevice()
    pipeline = GaussianSplatPipeline(device)
    image = pipeline.render([_point()], _camera(), observer=_observer())
    assert image.shape == (64, 64, 3)
    assert image.dtype == np.uint8


def test_pipeline_webgpu_empty_input_returns_black():
    device = WebGPUDevice()
    pipeline = GaussianSplatPipeline(device)
    image = pipeline.render([], _camera(), observer=_observer())
    assert image.shape == (64, 64, 3)
    assert (image == 0).all()


def test_pipeline_webgpu_no_observer_still_works():
    device = WebGPUDevice()
    pipeline = GaussianSplatPipeline(device)
    image = pipeline.render([_point()], _camera())
    assert image.shape == (64, 64, 3)


# --- WebGPUSplatRenderer fallback signaling ---


def test_webgpu_renderer_raises_when_device_unavailable():
    """Without a real WebGPU device the renderer must signal fallback."""
    cpu_only_device = WebGPUDevice(backend="cpu")
    renderer = WebGPUSplatRenderer(cpu_only_device, 32, 32)
    with pytest.raises(RuntimeError):
        renderer.render([_point()], _camera())


def test_webgpu_renderer_validates_dimensions():
    cpu_only_device = WebGPUDevice(backend="cpu")
    with pytest.raises(ValueError):
        WebGPUSplatRenderer(cpu_only_device, 0, 32)
    with pytest.raises(ValueError):
        WebGPUSplatRenderer(cpu_only_device, 32, -1)


# --- AIViewerConfig ---


def test_config_gpu_backend_default_is_auto():
    cfg = AIViewerConfig()
    assert cfg.gpu_backend == "auto"
    assert cfg.enable_shader_warp is True
    cfg.validate()


@pytest.mark.parametrize("backend", ["auto", "webgpu", "cpu"])
def test_config_gpu_backend_valid_values(backend):
    AIViewerConfig(gpu_backend=backend).validate()


def test_config_gpu_backend_invalid_value_rejected():
    with pytest.raises(ValueError):
        AIViewerConfig(gpu_backend="vulkan").validate()


# --- AIViewer note text ---


class _StubClient:
    def receive_message(self): return None
    def connect(self): pass
    def disconnect(self): pass


def test_ai_viewer_gpu_note_includes_backend_name(capsys):
    from ai_viewer import AIViewer

    cfg = AIViewerConfig(
        enable_window=False,
        render_mode="gaussian",
        use_gpu_pipeline=True,
        gpu_backend="auto",
        output_directory="outputs/viewer-test",
    )
    AIViewer(cfg, _StubClient())
    out = capsys.readouterr().out
    # The line should mention either webgpu or cpu (whichever resolved).
    assert ("backend=webgpu" in out) or ("backend=cpu" in out)
    assert "warp:" in out


def test_ai_viewer_with_shader_warp_off_announces_off(capsys):
    from ai_viewer import AIViewer

    cfg = AIViewerConfig(
        enable_window=False,
        render_mode="gaussian",
        use_gpu_pipeline=True,
        enable_shader_warp=False,
        output_directory="outputs/viewer-test",
    )
    AIViewer(cfg, _StubClient())
    out = capsys.readouterr().out
    assert "warp: off" in out
