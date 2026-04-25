"""High-level orchestration for the AI Viewer client."""

from __future__ import annotations

import base64
import json
import time
from collections import deque
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image

from ai_viewer.client import RuntimeClient
from ai_viewer.config import AIViewerConfig
from ai_viewer.frame_buffer import FrameBuffer
from ai_viewer.postprocess import (
    FramePostProcessor,
    SafeProcessor,
    build_postprocessor_from_config,
)
from ai_viewer.window import ViewerWindow


class AIViewer:
    """Receive runtime messages, postprocess frames, display + save them.

    Phase 19: an optional :class:`ViewerWindow` displays each frame
    in real time and an optional :class:`FramePostProcessor` runs
    contrast / brightness / color shifts on the way past. ``run_once``
    processes one message; ``run_loop`` drains a bounded number of
    messages and respects ``config.max_fps`` by sleeping between
    frame renders.
    """

    def __init__(
        self,
        config: AIViewerConfig,
        client: RuntimeClient,
        window: ViewerWindow | None = None,
        postprocessor: FramePostProcessor | None = None,
    ) -> None:
        config.validate()
        self.config = config
        self.client = client
        self.window = window
        if config.render_mode == "gaussian":
            pipeline = (
                "GPU mock (GaussianSplatPipeline)"
                if config.use_gpu_pipeline
                else "CPU (GaussianSplatRenderer)"
            )
            print(
                f"AIViewer note: render_mode='gaussian' bypasses server "
                f"PPM frames; drive {pipeline} directly for that path"
            )
        if postprocessor is None:
            if config.render_mode == "gaussian":
                # In gaussian mode, image postprocess is meaningless because
                # the viewer is not the one rendering pixels.
                postprocessor = FramePostProcessor()
            elif config.use_photon_warp:
                # Photon-space warping happens at the source; the viewer
                # should not double-process the resulting image.
                if not config.photon_warp_model_path:
                    print(
                        "AIViewer warning: use_photon_warp=True but no "
                        "photon_warp_model_path provided; image postprocess "
                        "still bypassed (deterministic perception assumed)"
                    )
                postprocessor = FramePostProcessor()
            else:
                postprocessor = build_postprocessor_from_config(config)
        self.postprocessor: FramePostProcessor = postprocessor
        self.frame_buffer = FrameBuffer()
        self.last_scene_state: dict | None = None
        self.last_frame_path: str | None = None
        self.frames_received: int = 0
        self.scene_states_received: int = 0
        self._frame_times: deque[float] = deque(maxlen=30)
        self._min_frame_interval: float = 1.0 / max(config.max_fps, 1.0e-6)
        self._last_render_time: float = 0.0
        self._postprocessor_warned: bool = False
        print(f"AIViewer postprocessor: {self._describe_postprocessor()}")

    def _describe_postprocessor(self) -> str:
        """One-line label for the active processor (handles SafeProcessor wrap)."""
        proc = self.postprocessor
        if isinstance(proc, SafeProcessor):
            return f"SafeProcessor({type(proc.wrapped).__name__})"
        return type(proc).__name__

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

    # --- AI postprocess (legacy gamma curve) ------------------------------

    def optional_ai_postprocess(self, frame_buffer: FrameBuffer) -> FrameBuffer:
        """Deterministic gamma 0.8 + clamp; placeholder for a real neural step.

        Preserves dimensions; always writes to a new :class:`FrameBuffer`.
        Kept around for backwards-compat with ``config.enable_ai_postprocess``;
        Phase 19 prefers the modular :class:`FramePostProcessor` chain.
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
            self._handle_frame(message)
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

    def fps(self) -> float:
        """Approximate frames-per-second over the recent window."""
        if len(self._frame_times) < 2:
            return 0.0
        span = self._frame_times[-1] - self._frame_times[0]
        if span <= 0.0:
            return 0.0
        return (len(self._frame_times) - 1) / span

    # --- frame pipeline ---------------------------------------------------

    def _handle_frame(self, message: dict) -> None:
        encoded = message.get("data", "")
        if not encoded:
            return
        try:
            ppm_bytes = base64.b64decode(encoded)
        except Exception:
            return

        image = self._load_image(ppm_bytes)
        if image is None:
            return

        # Mirror to FrameBuffer so ASCII previews / legacy paths still work.
        self._sync_frame_buffer(image)

        if self.config.enable_postprocess and self.postprocessor is not None:
            try:
                image = self.postprocessor.process(image)
            except Exception:
                pass
            self._maybe_warn_postprocessor()
        elif self.config.enable_ai_postprocess:
            self.frame_buffer = self.optional_ai_postprocess(self.frame_buffer)
            image = Image.fromarray(self.frame_buffer.pixels, mode="RGB")

        self._save_image(image)
        self._display(image)

        now = time.perf_counter()
        self._frame_times.append(now)
        self.frames_received += 1
        # FPS print every 5 frames as required by Phase 19.
        if self.frames_received % 5 == 0:
            print(f"FPS: {self.fps():.1f}")

        elapsed_since_last = now - self._last_render_time
        if elapsed_since_last < self._min_frame_interval:
            time.sleep(self._min_frame_interval - elapsed_since_last)
        self._last_render_time = time.perf_counter()

    def _load_image(self, ppm_bytes: bytes) -> Image.Image | None:
        try:
            image = Image.open(BytesIO(ppm_bytes))
            image.load()
            if image.mode != "RGB":
                image = image.convert("RGB")
            return image
        except Exception:
            try:
                self.frame_buffer.load_ppm_bytes(ppm_bytes)
            except Exception:
                return None
            return Image.fromarray(self.frame_buffer.pixels, mode="RGB")

    def _sync_frame_buffer(self, image: Image.Image) -> None:
        rgb = image if image.mode == "RGB" else image.convert("RGB")
        self.frame_buffer = FrameBuffer(rgb.width, rgb.height)
        self.frame_buffer.pixels = np.asarray(rgb, dtype=np.uint8)

    def _save_image(self, image: Image.Image) -> None:
        out_dir = Path(self.config.output_directory)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"frame_{int(time.time() * 1_000)}.ppm"
        try:
            image.save(path, format="PPM")
        except Exception:
            rgb = image if image.mode == "RGB" else image.convert("RGB")
            buf = FrameBuffer(rgb.width, rgb.height)
            buf.pixels = np.asarray(rgb, dtype=np.uint8)
            buf.save_ppm(str(path))
        self.last_frame_path = str(path)

    def _display(self, image: Image.Image) -> None:
        if not self.config.enable_window or self.window is None:
            return
        try:
            self.window.show_frame(image)
            self.window.update()
        except Exception:
            pass

    def _maybe_warn_postprocessor(self) -> None:
        """Print a one-line warning if the postprocessor reports an error."""
        if self._postprocessor_warned:
            return
        proc = self.postprocessor
        error = getattr(proc, "last_error", None)
        if not error and isinstance(proc, SafeProcessor):
            error = getattr(proc.wrapped, "last_error", None)
        if not error:
            return
        print(
            f"AIViewer postprocessor warning: {error} "
            "(falling back to passthrough for this frame)"
        )
        self._postprocessor_warned = True
