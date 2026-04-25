"""Tests for Phase 18 AI Viewer client."""

from __future__ import annotations

import base64
import json
from pathlib import Path

import numpy as np
import pytest

from ai_viewer import AIViewer, AIViewerConfig, FrameBuffer, RuntimeClient


def _make_ppm_bytes(width: int = 4, height: int = 3) -> bytes:
    pixels = []
    for y in range(height):
        for x in range(width):
            pixels.append((x * 30, y * 60, (x + y) * 20))
    body = " ".join(f"{r} {g} {b}" for (r, g, b) in pixels)
    return f"P3\n{width} {height}\n255\n{body}\n".encode("ascii")


# --- AIViewerConfig validation ---


def test_config_defaults_validate():
    AIViewerConfig().validate()


@pytest.mark.parametrize(
    "overrides",
    [
        {"server_host": ""},
        {"server_port": 0},
        {"server_port": 70_000},
        {"server_port": -1},
        {"width": 0},
        {"width": -1},
        {"height": 0},
        {"output_directory": ""},
    ],
)
def test_config_rejects_bad_fields(overrides):
    cfg = AIViewerConfig()
    for k, v in overrides.items():
        setattr(cfg, k, v)
    with pytest.raises(ValueError):
        cfg.validate()


# --- FrameBuffer ---


def test_frame_buffer_clear_resets_pixels():
    fb = FrameBuffer(4, 3)
    fb.pixels = np.full((3, 4, 3), 200, dtype=np.uint8)
    fb.clear()
    assert (fb.pixels == 0).all()
    assert fb.pixels.shape == (3, 4, 3)


def test_frame_buffer_load_ppm_bytes_round_trip(tmp_path: Path):
    fb = FrameBuffer()
    fb.load_ppm_bytes(_make_ppm_bytes(4, 3))
    assert fb.width == 4
    assert fb.height == 3
    assert fb.pixels.shape == (3, 4, 3)
    # Save and reload via disk to confirm the writer matches the parser.
    out = tmp_path / "frame.ppm"
    fb.save_ppm(str(out))
    fb2 = FrameBuffer()
    fb2.load_ppm_bytes(out.read_bytes())
    np.testing.assert_array_equal(fb2.pixels, fb.pixels)


def test_frame_buffer_load_ppm_handles_comments():
    payload = (
        b"P3\n# this is a comment\n4 3\n# another comment\n255\n"
        + b"0 0 0 " * (4 * 3) + b"\n"
    )
    fb = FrameBuffer()
    fb.load_ppm_bytes(payload)
    assert fb.width == 4
    assert fb.height == 3


def test_frame_buffer_rejects_non_p3():
    fb = FrameBuffer()
    with pytest.raises(ValueError):
        fb.load_ppm_bytes(b"P6\n4 3\n255\n\x00" * 36)


def test_frame_buffer_rejects_truncated_body():
    fb = FrameBuffer()
    with pytest.raises(ValueError):
        fb.load_ppm_bytes(b"P3\n4 3\n255\n0 0 0\n")


def test_frame_buffer_save_rejects_empty_buffer(tmp_path: Path):
    fb = FrameBuffer()
    with pytest.raises(ValueError):
        fb.save_ppm(str(tmp_path / "empty.ppm"))


def test_frame_buffer_ascii_preview_is_non_empty():
    fb = FrameBuffer()
    fb.load_ppm_bytes(_make_ppm_bytes(8, 4))
    text = fb.to_ascii_preview(max_width=4)
    assert text
    rows = text.split("\n")
    assert all(len(r) <= 4 for r in rows)


def test_frame_buffer_ascii_preview_handles_empty_buffer():
    assert FrameBuffer().to_ascii_preview() == ""


# --- AIViewer ---


class _StubClient:
    """Fake RuntimeClient that returns a queued list of messages."""

    def __init__(self, messages: list[dict | None]) -> None:
        self.messages = list(messages)
        self.connected = False
        self.disconnected = False

    def connect(self) -> None:
        self.connected = True

    def disconnect(self) -> None:
        self.disconnected = True

    def receive_message(self) -> dict | None:
        if not self.messages:
            return None
        return self.messages.pop(0)


def _scene_state_message() -> dict:
    return {
        "type": "scene_state",
        "data": {
            "julian_date": 2_451_545.0,
            "total_objects": 3,
            "active_objects": 3,
            "object_type_counts": {"star": 2, "galaxy": 1},
            "truth_level_counts": {"catalog_imported": 3},
            "source_counts": {"gaia": 2, "desi": 1},
            "physics_backend": "none",
            "perception_enabled": True,
            "ai_warp_enabled": False,
            "notes": ["frame ok"],
        },
    }


def _frame_message() -> dict:
    return {
        "type": "frame",
        "data": base64.b64encode(_make_ppm_bytes(4, 3)).decode("ascii"),
    }


def test_viewer_handles_missing_frame_gracefully():
    config = AIViewerConfig(output_directory="outputs/viewer-test")
    client = _StubClient([None])
    viewer = AIViewer(config, client)
    assert viewer.run_once() is None
    assert viewer.frames_received == 0
    assert viewer.last_frame_path is None


def test_viewer_processes_scene_state():
    config = AIViewerConfig(output_directory="outputs/viewer-test")
    viewer = AIViewer(config, _StubClient([_scene_state_message()]))
    viewer.run_once()
    assert viewer.last_scene_state is not None
    assert viewer.last_scene_state["total_objects"] == 3
    assert viewer.scene_states_received == 1


def test_viewer_renders_scene_state_text():
    config = AIViewerConfig()
    viewer = AIViewer(config, _StubClient([]))
    text = viewer.render_scene_state_text(_scene_state_message()["data"])
    assert "julian_date" in text
    assert "total_objects" in text
    assert "physics" in text


def test_viewer_saves_frame_to_output_directory(tmp_path: Path):
    config = AIViewerConfig(output_directory=str(tmp_path))
    viewer = AIViewer(config, _StubClient([_frame_message()]))
    viewer.run_once()
    assert viewer.frames_received == 1
    assert viewer.last_frame_path is not None
    assert Path(viewer.last_frame_path).is_file()
    assert Path(viewer.last_frame_path).parent == tmp_path


def test_viewer_run_loop_drains_queue_and_disconnects():
    client = _StubClient([_scene_state_message(), _scene_state_message(), None])
    config = AIViewerConfig(output_directory="outputs/viewer-test")
    viewer = AIViewer(config, client)
    processed = viewer.run_loop(max_frames=10)
    assert processed == 2
    assert client.connected is True
    assert client.disconnected is True


def test_optional_ai_postprocess_preserves_dimensions(tmp_path: Path):
    config = AIViewerConfig(output_directory=str(tmp_path))
    viewer = AIViewer(config, _StubClient([]))
    fb = FrameBuffer()
    fb.load_ppm_bytes(_make_ppm_bytes(5, 4))
    boosted = viewer.optional_ai_postprocess(fb)
    assert boosted.width == fb.width
    assert boosted.height == fb.height
    assert boosted.pixels.shape == fb.pixels.shape
    # gamma 0.8 brightens midtones — sum should not decrease (sanity)
    assert boosted.pixels.sum() >= fb.pixels.sum() * 0.99


def test_optional_ai_postprocess_handles_empty_buffer():
    config = AIViewerConfig()
    viewer = AIViewer(config, _StubClient([]))
    fb = FrameBuffer()
    out = viewer.optional_ai_postprocess(fb)
    assert out is fb  # no-op for empty buffer


def test_viewer_with_ai_postprocess_enabled_writes_boosted_frame(tmp_path: Path):
    config = AIViewerConfig(
        output_directory=str(tmp_path), enable_ai_postprocess=True
    )
    viewer = AIViewer(config, _StubClient([_frame_message()]))
    viewer.run_once()
    assert viewer.frames_received == 1
    assert viewer.last_frame_path is not None
    assert Path(viewer.last_frame_path).is_file()
