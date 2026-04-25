"""High-level orchestration for the AI Viewer client."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from ai_viewer.client import RuntimeClient
from ai_viewer.config import AIViewerConfig
from ai_viewer.frame_buffer import FrameBuffer


class AIViewer:
    """Receive runtime messages, save frames, render SceneState text.

    The "AI" postprocess is currently a deterministic gamma adjustment;
    a real neural step will swap in here without changing the public
    API.
    """

    def __init__(self, config: AIViewerConfig, client: RuntimeClient) -> None:
        config.validate()
        self.config = config
        self.client = client
        self.frame_buffer = FrameBuffer()
        self.last_scene_state: dict | None = None
        self.last_frame_path: str | None = None
        self.frames_received: int = 0
        self.scene_states_received: int = 0

    # --- scene-state rendering -------------------------------------------

    def render_scene_state_text(self, scene_state: dict) -> str:
        """Format a :class:`SceneState` dict as a short readable block."""
        type_counts = scene_state.get("object_type_counts", {})
        source_counts = scene_state.get("source_counts", {})
        notes = scene_state.get("notes", []) or []
        lines = [
            f"julian_date    : {scene_state.get('julian_date'):.6f}",
            f"total_objects  : {scene_state.get('total_objects')}",
            f"active_objects : {scene_state.get('active_objects')}",
            f"physics        : {scene_state.get('physics_backend')}",
            f"perception     : {scene_state.get('perception_enabled')}",
            f"types          : {json.dumps(type_counts, sort_keys=True)}",
            f"sources        : {json.dumps(source_counts, sort_keys=True)}",
        ]
        if notes:
            lines.append(f"notes          : {notes}")
        return "\n".join(lines)

    # --- AI postprocess (placeholder) -------------------------------------

    def optional_ai_postprocess(self, frame_buffer: FrameBuffer) -> FrameBuffer:
        """Deterministic gamma 0.8 + clamp; placeholder for a real neural step.

        Preserves dimensions; idempotent on shape; always writes to a new
        :class:`FrameBuffer`.
        """
        if frame_buffer.width <= 0 or frame_buffer.height <= 0:
            return frame_buffer
        normalized = frame_buffer.pixels.astype(np.float64) / 255.0
        boosted = np.power(normalized, 0.8) * 255.0
        result = FrameBuffer(frame_buffer.width, frame_buffer.height)
        result.pixels = np.clip(boosted, 0.0, 255.0).astype(np.uint8)
        return result

    # --- per-message dispatch --------------------------------------------

    def run_once(self) -> dict | None:
        """Receive and process one message; returns the raw message or ``None``."""
        message = self.client.receive_message()
        if message is None:
            return None
        kind = message.get("type")
        if kind == "scene_state":
            self.last_scene_state = message.get("data")
            self.scene_states_received += 1
        elif kind == "frame":
            data = message.get("data", "")
            if data:
                import base64

                try:
                    self.frame_buffer.load_ppm_bytes(base64.b64decode(data))
                except Exception:
                    return message
                if self.config.enable_ai_postprocess:
                    self.frame_buffer = self.optional_ai_postprocess(
                        self.frame_buffer
                    )
                self._save_frame()
                self.frames_received += 1
        return message

    def run_loop(self, max_frames: int | None = None) -> int:
        """Connect, drain ``max_frames`` messages, disconnect.

        Returns the number of messages actually processed.
        """
        self.client.connect()
        processed = 0
        try:
            while max_frames is None or processed < max_frames:
                message = self.run_once()
                if message is None:
                    break
                processed += 1
        finally:
            self.client.disconnect()
        return processed

    # --- helpers ----------------------------------------------------------

    def _save_frame(self) -> None:
        out_dir = Path(self.config.output_directory)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"frame_{int(time.time() * 1_000)}.ppm"
        self.frame_buffer.save_ppm(str(path))
        self.last_frame_path = str(path)
