"""Tests for Phase 20 neural postprocessing."""

from __future__ import annotations

import base64
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from ai_viewer import (
    AIViewer,
    AIViewerConfig,
    BrightnessProcessor,
    CompositeProcessor,
    ContrastBoostProcessor,
    FramePostProcessor,
    ONNXFrameProcessor,
    RuntimeClient,
    SafeProcessor,
    build_postprocessor_from_config,
)


_REPO_ROOT = Path(__file__).resolve().parent.parent
_IDENTITY_MODEL = _REPO_ROOT / "data" / "neural_postprocess_identity.onnx"


def _solid_image(width: int = 32, height: int = 32, color=(120, 80, 200)) -> Image.Image:
    return Image.new("RGB", (width, height), color=color)


def _ppm_bytes(image: Image.Image) -> bytes:
    arr = np.asarray(image.convert("RGB"), dtype=np.int32)
    h, w = arr.shape[:2]
    body = " ".join(
        " ".join(f"{r} {g} {b}" for r, g, b in row) for row in arr
    )
    return f"P3\n{w} {h}\n255\n{body}\n".encode("ascii")


# --- ONNXFrameProcessor: missing/invalid model fallback ---


def test_onnx_processor_missing_model_records_error_and_returns_input():
    proc = ONNXFrameProcessor("/nonexistent/never.onnx")
    assert proc.last_error is not None
    assert "model not found" in proc.last_error
    image = _solid_image()
    out = proc.process(image)
    np.testing.assert_array_equal(np.asarray(out), np.asarray(image))


def test_onnx_processor_invalid_model_records_error(tmp_path: Path):
    bogus = tmp_path / "bad.onnx"
    bogus.write_bytes(b"not an onnx file")
    proc = ONNXFrameProcessor(str(bogus))
    assert proc.last_error is not None
    assert "failed to load" in proc.last_error
    out = proc.process(_solid_image())
    assert out.size == (32, 32)


# --- ONNXFrameProcessor: real identity model end-to-end ---


def test_onnx_processor_identity_preserves_size():
    proc = ONNXFrameProcessor(
        str(_IDENTITY_MODEL), input_size=(32, 32), normalize=True
    )
    assert proc.session is not None
    assert proc.last_error is None
    image = _solid_image(32, 32, color=(120, 80, 200))
    out = proc.process(image)
    assert out.size == image.size
    # round-trip through Image conversion + uint8 cast can shift values by 1
    np.testing.assert_allclose(
        np.asarray(out, dtype=np.int16),
        np.asarray(image, dtype=np.int16),
        atol=1,
    )


def test_onnx_processor_resizes_input_and_restores_original_size():
    proc = ONNXFrameProcessor(
        str(_IDENTITY_MODEL), input_size=(32, 32), normalize=True
    )
    image = _solid_image(64, 48, color=(50, 150, 220))
    out = proc.process(image)
    assert out.size == (64, 48)


def test_onnx_processor_no_explicit_input_size_defaults_to_image_size(tmp_path: Path):
    """When input_size is None, the processor uses the image's size — but the
    bundled identity model has a fixed shape, so a wrong size fails inference
    and the processor must surface the error and return the original image.
    """
    proc = ONNXFrameProcessor(str(_IDENTITY_MODEL), input_size=None)
    image = _solid_image(20, 20)
    out = proc.process(image)
    assert out.size == (20, 20)
    assert proc.last_error is not None
    assert "inference failed" in proc.last_error


# --- SafeProcessor ---


class _RaisingProcessor(FramePostProcessor):
    def process(self, image: Image.Image) -> Image.Image:
        raise RuntimeError("boom")


class _FlagErrorProcessor(FramePostProcessor):
    """Processor that does its job but advertises a non-fatal error."""

    def __init__(self) -> None:
        self.last_error = "advisory: model warmed up cold"

    def process(self, image: Image.Image) -> Image.Image:
        return image


def test_safe_processor_catches_exception_and_returns_input():
    safe = SafeProcessor(_RaisingProcessor())
    image = _solid_image(8, 8)
    out = safe.process(image)
    assert out is image
    assert safe.last_error == "boom"


def test_safe_processor_propagates_inner_last_error():
    safe = SafeProcessor(_FlagErrorProcessor())
    safe.process(_solid_image(8, 8))
    assert safe.last_error == "advisory: model warmed up cold"


def test_safe_processor_clean_run_leaves_last_error_none():
    safe = SafeProcessor(BrightnessProcessor(factor=1.1))
    safe.process(_solid_image())
    assert safe.last_error is None


# --- AIViewerConfig validation for new fields ---


def test_config_neural_input_size_must_be_paired():
    cfg = AIViewerConfig(neural_input_width=64)
    with pytest.raises(ValueError):
        cfg.validate()
    cfg = AIViewerConfig(neural_input_height=64)
    with pytest.raises(ValueError):
        cfg.validate()


def test_config_neural_input_dimensions_must_be_positive():
    cfg = AIViewerConfig(neural_input_width=0, neural_input_height=64)
    with pytest.raises(ValueError):
        cfg.validate()
    cfg = AIViewerConfig(neural_input_width=64, neural_input_height=-1)
    with pytest.raises(ValueError):
        cfg.validate()


def test_config_neural_input_size_paired_validates():
    AIViewerConfig(
        neural_input_width=32, neural_input_height=32
    ).validate()


def test_config_neural_normalize_default_true():
    assert AIViewerConfig().neural_normalize is True


# --- build_postprocessor_from_config ---


def test_builder_returns_passthrough_when_postprocess_disabled():
    cfg = AIViewerConfig(enable_postprocess=False)
    proc = build_postprocessor_from_config(cfg)
    assert type(proc) is FramePostProcessor


def test_builder_returns_composite_when_no_neural_model():
    cfg = AIViewerConfig(enable_postprocess=True, neural_model_path=None)
    proc = build_postprocessor_from_config(cfg)
    assert isinstance(proc, CompositeProcessor)
    assert any(isinstance(p, ContrastBoostProcessor) for p in proc.processors)
    assert any(isinstance(p, BrightnessProcessor) for p in proc.processors)


def test_builder_returns_safe_wrapped_neural_when_model_path_set():
    cfg = AIViewerConfig(neural_model_path=str(_IDENTITY_MODEL))
    proc = build_postprocessor_from_config(cfg)
    assert isinstance(proc, SafeProcessor)
    assert isinstance(proc.wrapped, ONNXFrameProcessor)


def test_builder_neural_with_missing_model_still_safe():
    cfg = AIViewerConfig(neural_model_path="/does/not/exist.onnx")
    proc = build_postprocessor_from_config(cfg)
    assert isinstance(proc, SafeProcessor)
    image = _solid_image()
    out = proc.process(image)
    np.testing.assert_array_equal(np.asarray(out), np.asarray(image))


def test_builder_neural_passes_input_size_through():
    cfg = AIViewerConfig(
        neural_model_path=str(_IDENTITY_MODEL),
        neural_input_width=32,
        neural_input_height=32,
    )
    proc = build_postprocessor_from_config(cfg)
    inner = proc.wrapped
    assert inner.input_size == (32, 32)


# --- AIViewer integration ---


class _StubClient:
    def __init__(self, messages):
        self.messages = list(messages)
        self.connected = False
        self.disconnected = False

    def connect(self) -> None:
        self.connected = True

    def disconnect(self) -> None:
        self.disconnected = True

    def receive_message(self):
        if not self.messages:
            return None
        return self.messages.pop(0)


def _frame_message(image: Image.Image) -> dict:
    return {
        "type": "frame",
        "data": base64.b64encode(_ppm_bytes(image)).decode("ascii"),
    }


def test_viewer_auto_builds_safe_neural_postprocessor(tmp_path: Path, capsys):
    cfg = AIViewerConfig(
        output_directory=str(tmp_path),
        enable_window=False,
        neural_model_path=str(_IDENTITY_MODEL),
        neural_input_width=32,
        neural_input_height=32,
    )
    viewer = AIViewer(cfg, _StubClient([]))
    captured = capsys.readouterr().out
    assert isinstance(viewer.postprocessor, SafeProcessor)
    assert isinstance(viewer.postprocessor.wrapped, ONNXFrameProcessor)
    assert "SafeProcessor(ONNXFrameProcessor)" in captured


def test_viewer_with_missing_model_warns_once(tmp_path: Path, capsys):
    cfg = AIViewerConfig(
        output_directory=str(tmp_path),
        enable_window=False,
        neural_model_path="/no/such/model.onnx",
    )
    image = _solid_image(8, 8)
    viewer = AIViewer(
        cfg,
        _StubClient([_frame_message(image), _frame_message(image)]),
    )
    capsys.readouterr()  # drop init line
    viewer.run_once()
    viewer.run_once()
    out = capsys.readouterr().out
    assert out.count("postprocessor warning") == 1


def test_viewer_neural_passthrough_preserves_frame_dimensions(tmp_path: Path):
    cfg = AIViewerConfig(
        output_directory=str(tmp_path),
        enable_window=False,
        neural_model_path=str(_IDENTITY_MODEL),
        neural_input_width=32,
        neural_input_height=32,
        max_fps=120.0,
    )
    image = _solid_image(64, 48, color=(50, 150, 220))
    viewer = AIViewer(cfg, _StubClient([_frame_message(image)]))
    viewer.run_once()
    assert viewer.frames_received == 1
    saved = Image.open(viewer.last_frame_path)
    assert saved.size == (64, 48)
