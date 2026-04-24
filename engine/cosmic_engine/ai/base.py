"""Abstract interface every AI warp model must implement."""

from __future__ import annotations

from cosmic_engine.core.vector import Vector3
from cosmic_engine.perception.observer import ObserverState


class AIWarpModel:
    """Interface for models that remap perception at the photon-sample level.

    Subclasses replace one or more of the deterministic transforms in
    :mod:`cosmic_engine.perception.transform`. All methods receive the
    current :class:`ObserverState` so the model can condition on pose,
    velocity, and ``warp_factor``.

    The base implementation raises :class:`NotImplementedError` on every
    prediction method so forgetting to override one is a hard error,
    not a silent identity passthrough.
    """

    def predict_direction(
        self,
        direction: Vector3,
        observer: ObserverState,
    ) -> Vector3:
        """Return a remapped unit-length direction."""
        raise NotImplementedError

    def predict_brightness(
        self,
        brightness: float,
        direction: Vector3,
        observer: ObserverState,
    ) -> float:
        """Return a remapped apparent brightness (non-negative, finite)."""
        raise NotImplementedError

    def predict_color(
        self,
        color_rgb: tuple[int, int, int],
        direction: Vector3,
        observer: ObserverState,
    ) -> tuple[int, int, int]:
        """Return a remapped RGB tint with each channel in ``[0, 255]``."""
        raise NotImplementedError

    def confidence(self) -> float:
        """Return a scalar in ``[0, 1]`` describing the model's self-rated trust."""
        raise NotImplementedError
