"""Tests for Phase 19 viewer postprocessing and window scaffolding."""

from __future__ import annotations

import base64

import numpy as np
import pytest
from PIL import Image

from ai_viewer import (
    AIViewer,
    AIViewerConfig,
    BrightnessProcessor,
    ColorShiftProcessor,
    CompositeProcessor,
    ContrastBoostProcessor,
    FramePostProcessor,
    RuntimeClient,
    ViewerWindow,
)


def _solid_image(width: int = 6, height: int = 4, color=(120, 80, 200)) -> Image.Image:
    return Image.new("RGB", (width, height), color=color)


def _ppm_bytes(image: Image.Image) -> bytes:
    pixels = np.asarray(image.convert("RGB"), dtype=np.int32)
    h, w = pixels.shape[:2]
    body = " ".join(
        " ".join(f"{r} {g} {b}" for r, g, b in row) for row in pixels
    )
    return f"P3\n{w} {h}\n255\n{body}\n".encode("ascii")


# --- config validation includes the new fields ---


def test_config_validates_new_fields_defaults():
    AIViewerConfig().validate()


def test_config_rejects_non_positive_max_fps():
    cfg = AIViewerConfig(max_fps=0.0)
    with pytest.raises(ValueError):
        cfg.validate()
    cfg = AIViewerConfig(max_fps=-1.0)
    with pytest.raises(ValueError):
        cfg.validate()


def test_config_accepts_window_and_postprocess_toggles():
    cfg = AIViewerConfig(enable_window=False, enable_postprocess=False)
    cfg.validate()


# --- base class is a pass-through ---


def test_base_processor_is_passthrough():
    image = _solid_image()
    out = FramePostProcessor().process(image)
    assert out is image


# --- ContrastBoostProcessor ---


def test_contrast_processor_preserves_dimensions():
    image = _solid_image(8, 5)
    out = ContrastBoostProcessor(factor=1.5).process(image)
    assert out.size == image.size


def test_contrast_processor_rejects_non_positive_factor():
    with pytest.raises(ValueError):
        ContrastBoostProcessor(factor=0.0)
    with pytest.raises(ValueError):
        ContrastBoostProcessor(factor=-1.0)


# --- BrightnessProcessor ---


def test_brightness_processor_preserves_dimensions():
    image = _solid_image(7, 3)
    out = BrightnessProcessor(factor=1.5).process(image)
    assert out.size == image.size


def test_brightness_factor_above_one_raises_average():
    image = _solid_image(4, 4, color=(80, 80, 80))
    boosted = BrightnessProcessor(factor=1.5).process(image)
    assert np.asarray(boosted).mean() > np.asarray(image).mean()


def test_brightness_processor_rejects_negative_factor():
    with pytest.raises(ValueError):
        BrightnessProcessor(factor=-0.1)


# --- ColorShiftProcessor ---


def test_color_shift_preserves_dimensions():
    image = _solid_image(5, 5)
    out = ColorShiftProcessor(r_shift=10, g_shift=-5, b_shift=20).process(image)
    assert out.size == image.size


def test_color_shift_clamps_to_byte_range():
    image = _solid_image(3, 3, color=(250, 5, 200))
    out = ColorShiftProcessor(r_shift=50, g_shift=-20, b_shift=80).process(image)
    arr = np.asarray(out)
    assert (arr[..., 0] == 255).all()
    assert (arr[..., 1] == 0).all()
    assert (arr[..., 2] == 255).all()


def test_color_shift_zero_is_identity():
    image = _solid_image(4, 4)
    out = ColorShiftProcessor().process(image)
    np.testing.assert_array_equal(np.asarray(out), np.asarray(image))


# --- CompositeProcessor ---


def test_composite_chains_in_order():
    image = _solid_image(4, 4, color=(100, 100, 100))
    composite = CompositeProcessor(
        [
            ContrastBoostProcessor(factor=1.2),
            BrightnessProcessor(factor=1.1),
            ColorShiftProcessor(b_shift=10),
        ]
    )
    out = composite.process(image)
    assert out.size == image.size


def test_composite_empty_is_identity():
    image = _solid_image()
    out = CompositeProcessor([]).process(image)
    np.testing.assert_array_equal(np.asarray(out), np.asarray(image))


def test_composite_calls_each_processor_once():
    counter = {"n": 0}

    class Counter(FramePostProcessor):
        def process(self, image: Image.Image) -> Image.Image:
            counter["n"] += 1
            return image

    image = _solid_image()
    CompositeProcessor([Counter(), Counter(), Counter()]).process(image)
    assert counter["n"] == 3


def test_composite_handles_small_image():
    tiny = _solid_image(1, 1, color=(50, 50, 50))
    composite = CompositeProcessor(
        [
            ContrastBoostProcessor(factor=1.4),
            BrightnessProcessor(factor=0.9),
            ColorShiftProcessor(r_shift=10),
        ]
    )
    out = composite.process(tiny)
    assert out.size == (1, 1)


# --- ViewerWindow scaffolding (no display required) ---


def test_viewer_window_validates_dimensions():
    with pytest.raises(ValueError):
        ViewerWindow(0, 100)
    with pytest.raises(ValueError):
        ViewerWindow(100, -1)


def test_viewer_window_close_is_safe_when_never_opened():
    window = ViewerWindow(64, 64)
    window.close()  # must not raise even before _ensure_open ran
    assert window._closed is True


def test_viewer_window_show_frame_no_op_when_unavailable():
    window = ViewerWindow(64, 64)
    window.available = False
    # Should not raise even though no Tk root exists.
    window.show_frame(_solid_image())
    window.update()


# --- AIViewer integration with the new pipeline ---


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


def test_viewer_postprocessor_runs_when_enabled(tmp_path):
    image = _solid_image(8, 8, color=(80, 80, 80))
    config = AIViewerConfig(
        output_directory=str(tmp_path),
        enable_window=False,
        enable_postprocess=True,
        max_fps=120.0,
    )
    composite = CompositeProcessor(
        [
            BrightnessProcessor(factor=1.5),
            ColorShiftProcessor(b_shift=20),
        ]
    )
    viewer = AIViewer(
        config,
        _StubClient([_frame_message(image)]),
        postprocessor=composite,
    )
    viewer.run_once()
    assert viewer.frames_received == 1
    saved = np.asarray(Image.open(viewer.last_frame_path).convert("RGB"))
    # original average was 80; postprocessed should be brighter and
    # blue-shifted relative to the input.
    assert saved.mean() > 80.0
    assert saved[..., 2].mean() > saved[..., 0].mean()


def test_viewer_skips_postprocess_when_disabled(tmp_path):
    image = _solid_image(8, 8, color=(80, 80, 80))
    config = AIViewerConfig(
        output_directory=str(tmp_path),
        enable_window=False,
        enable_postprocess=False,
        max_fps=120.0,
    )
    viewer = AIViewer(
        config,
        _StubClient([_frame_message(image)]),
        postprocessor=BrightnessProcessor(factor=10.0),
    )
    viewer.run_once()
    saved = np.asarray(Image.open(viewer.last_frame_path).convert("RGB"))
    assert int(saved.mean()) == 80


def test_viewer_fps_returns_zero_with_no_frames():
    config = AIViewerConfig(enable_window=False, output_directory="outputs/x")
    viewer = AIViewer(config, _StubClient([]))
    assert viewer.fps() == 0.0


def test_viewer_handles_invalid_frame_payload_gracefully(tmp_path):
    config = AIViewerConfig(
        output_directory=str(tmp_path),
        enable_window=False,
    )
    bad_message = {
        "type": "frame",
        "data": base64.b64encode(b"not a ppm").decode("ascii"),
    }
    viewer = AIViewer(config, _StubClient([bad_message]))
    viewer.run_once()
    assert viewer.frames_received == 0
    assert viewer.last_frame_path is None
