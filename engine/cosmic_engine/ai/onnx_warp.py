"""AI warp model backed by an ONNX inference session.

The bundled mock model at ``data/mock_warp_model.onnx`` has:
    input  shape [1, 9] = [dx, dy, dz, beta, warp_factor, brightness, r, g, b]
    output shape [1, 7] = [dx', dy', dz', brightness', r', g', b']

:class:`ONNXWarpModel` packs each ``predict_*`` call into that 9-vector
(zero-filling the fields the call doesn't use) and slices the output
accordingly. A real trained model can be dropped in simply by matching
these shapes.
"""

from __future__ import annotations

import math

from cosmic_engine.ai.base import AIWarpModel
from cosmic_engine.ai.onnx_model import ONNXModelWrapper
from cosmic_engine.core.vector import Vector3
from cosmic_engine.perception.observer import ObserverState


_BRIGHTNESS_CEILING = 1.0e18


def _normalize(v: Vector3) -> Vector3:
    n = math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z)
    if n == 0.0:
        return v
    return Vector3(v.x / n, v.y / n, v.z / n)


class ONNXWarpModel(AIWarpModel):
    """Real-inference variant of :class:`AIWarpModel`."""

    def __init__(self, model_path: str) -> None:
        self.model = ONNXModelWrapper(model_path)

    def _base_vector(
        self,
        direction: Vector3,
        observer: ObserverState,
    ) -> list[float]:
        return [
            direction.x,
            direction.y,
            direction.z,
            observer.beta(),
            observer.warp_factor,
            0.0,  # brightness slot
            0.0,  # r slot
            0.0,  # g slot
            0.0,  # b slot
        ]

    def predict_direction(
        self,
        direction: Vector3,
        observer: ObserverState,
    ) -> Vector3:
        out = self.model.predict(self._base_vector(direction, observer))
        return _normalize(Vector3(out[0], out[1], out[2]))

    def predict_brightness(
        self,
        brightness: float,
        direction: Vector3,
        observer: ObserverState,
    ) -> float:
        vec = self._base_vector(direction, observer)
        vec[5] = float(brightness)
        out = self.model.predict(vec)
        return min(max(float(out[3]), 0.0), _BRIGHTNESS_CEILING)

    def predict_color(
        self,
        color_rgb: tuple[int, int, int],
        direction: Vector3,
        observer: ObserverState,
    ) -> tuple[int, int, int]:
        r, g, b = color_rgb
        vec = self._base_vector(direction, observer)
        vec[6] = float(r)
        vec[7] = float(g)
        vec[8] = float(b)
        out = self.model.predict(vec)
        return (
            max(0, min(255, int(round(out[4])))),
            max(0, min(255, int(round(out[5])))),
            max(0, min(255, int(round(out[6])))),
        )

    def confidence(self) -> float:
        return 0.8
