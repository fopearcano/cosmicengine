"""Deterministic stand-in for a future neural perception model.

:class:`SimpleNeuralWarp` is not a neural network. It is a hand-written
function with the shape of an AI surrogate so the rest of the pipeline
can be wired up and tested. When a real model arrives it replaces this
class without changing anything downstream.
"""

from __future__ import annotations

import math

from cosmic_engine.ai.base import AIWarpModel
from cosmic_engine.core.vector import Vector3
from cosmic_engine.perception.observer import ObserverState


_BRIGHTNESS_CEILING = 1.0e18


def _normalize(v: Vector3) -> Vector3:
    n = math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z)
    if n == 0.0:
        return v
    return Vector3(v.x / n, v.y / n, v.z / n)


class SimpleNeuralWarp(AIWarpModel):
    """Deterministic pseudo-AI remapping. Truth level: AI_SURROGATE."""

    def predict_direction(
        self,
        direction: Vector3,
        observer: ObserverState,
    ) -> Vector3:
        forward = _normalize(observer.forward)
        amount = math.sqrt(observer.warp_factor) * abs(direction.z)
        warped = Vector3(
            direction.x + forward.x * amount,
            direction.y + forward.y * amount,
            direction.z + forward.z * amount,
        )
        return _normalize(warped)

    def predict_brightness(
        self,
        brightness: float,
        direction: Vector3,
        observer: ObserverState,
    ) -> float:
        scaled = brightness * (1.0 + observer.warp_factor * 0.1)
        return min(max(scaled, 0.0), _BRIGHTNESS_CEILING)

    def predict_color(
        self,
        color_rgb: tuple[int, int, int],
        direction: Vector3,
        observer: ObserverState,
    ) -> tuple[int, int, int]:
        r, g, b = color_rgb
        mix = min(observer.warp_factor / 50.0, 1.0)
        new_r = (1.0 - mix) * r + mix * g
        new_g = (1.0 - mix) * g + mix * b
        new_b = (1.0 - mix) * b + mix * r
        return (
            max(0, min(255, int(round(new_r)))),
            max(0, min(255, int(round(new_g)))),
            max(0, min(255, int(round(new_b)))),
        )

    def confidence(self) -> float:
        return 0.5
