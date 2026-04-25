"""Neural perception in photon space (Phase 21).

Unlike :class:`cosmic_engine.ai.onnx_warp.ONNXWarpModel`, this model
operates **before** rendering: callers feed it per-photon
direction / brightness / color samples and observer state, and it
returns warped versions that the existing photon renderer then
projects to pixels. This is "true AI perception" — the camera does
not see a deterministic photon field that gets touched up
afterwards; it sees a photon field the model already shaped.

Input vector (9 float32):

    [0] dir.x
    [1] dir.y
    [2] dir.z
    [3] brightness
    [4] color_r   (0..255)
    [5] color_g   (0..255)
    [6] color_b   (0..255)
    [7] observer_beta
    [8] warp_factor

Output vector (7 float32):

    [0..2]  new direction (will be renormalized by caller)
    [3]     new brightness
    [4..6]  new color (clamped to 0..255 by caller)

If the model fails to load or inference raises, every prediction
silently falls back to the deterministic
:mod:`cosmic_engine.perception.transform` helpers and ``last_error``
records the failure.
"""

from __future__ import annotations

import math

import numpy as np

from cosmic_engine.ai.base import AIWarpModel
from cosmic_engine.ai.onnx_model import ONNXModelWrapper
from cosmic_engine.core.vector import Vector3
from cosmic_engine.perception.observer import ObserverState
from cosmic_engine.perception.transform import (
    apply_brightness_warp,
    apply_color_warp,
    apply_direction_warp,
)


_BRIGHTNESS_CEILING = 1.0e18


def _normalize(v: Vector3) -> Vector3:
    n = math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z)
    if n == 0.0:
        return v
    return Vector3(v.x / n, v.y / n, v.z / n)


class ONNXPhotonWarpModel(AIWarpModel):
    """Neural photon-space warper backed by an ONNX session."""

    def __init__(self, model_path: str) -> None:
        self.model_path = model_path
        self.last_error: str | None = None
        self._wrapper: ONNXModelWrapper | None = None
        try:
            self._wrapper = ONNXModelWrapper(model_path)
        except FileNotFoundError as e:
            self.last_error = f"model not found: {e}"
        except Exception as e:
            self.last_error = f"failed to load model: {e}"

    # --- internals --------------------------------------------------------

    def _build_input(
        self,
        direction: Vector3,
        brightness: float,
        color_rgb: tuple[int, int, int],
        observer: ObserverState,
    ) -> list[float]:
        return [
            float(direction.x),
            float(direction.y),
            float(direction.z),
            float(brightness),
            float(color_rgb[0]),
            float(color_rgb[1]),
            float(color_rgb[2]),
            float(observer.beta()),
            float(observer.warp_factor),
        ]

    def _infer(self, vec: list[float]) -> list[float] | None:
        if self._wrapper is None or self._wrapper.session is None:
            return None
        try:
            arr = np.asarray([vec], dtype=np.float32)
            out = self._wrapper.session.run(
                [self._wrapper.output_name],
                {self._wrapper.input_name: arr},
            )[0]
            return np.asarray(out).flatten().tolist()
        except Exception as e:
            self.last_error = f"inference failed: {e}"
            return None

    # --- AIWarpModel interface -------------------------------------------

    def predict_direction(
        self,
        direction: Vector3,
        observer: ObserverState,
    ) -> Vector3:
        out = self._infer(self._build_input(direction, 1.0, (0, 0, 0), observer))
        if out is None or len(out) < 3:
            return apply_direction_warp(direction, observer)
        return _normalize(Vector3(out[0], out[1], out[2]))

    def predict_brightness(
        self,
        brightness: float,
        direction: Vector3,
        observer: ObserverState,
    ) -> float:
        out = self._infer(
            self._build_input(direction, brightness, (0, 0, 0), observer)
        )
        if out is None or len(out) < 4:
            return apply_brightness_warp(brightness, direction, observer)
        return min(max(float(out[3]), 0.0), _BRIGHTNESS_CEILING)

    def predict_color(
        self,
        color_rgb: tuple[int, int, int],
        direction: Vector3,
        observer: ObserverState,
    ) -> tuple[int, int, int]:
        out = self._infer(
            self._build_input(direction, 1.0, color_rgb, observer)
        )
        if out is None or len(out) < 7:
            return apply_color_warp(color_rgb, direction, observer)
        return (
            max(0, min(255, int(round(float(out[4]))))),
            max(0, min(255, int(round(float(out[5]))))),
            max(0, min(255, int(round(float(out[6]))))),
        )

    def confidence(self) -> float:
        return 0.3 if self.last_error else 0.85
