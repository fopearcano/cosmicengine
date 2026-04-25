"""Discrete event record (luminosity spike, supernova, signal, …)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class Event:
    """A pointlike, time-stamped occurrence in the universe.

    ``time_t`` is global *coordinate* time in seconds — the same clock
    every observer's coordinate_time_t advances on. Each observer
    decides what they actually perceive based on the past light cone.
    """

    id: str
    position_m: np.ndarray
    time_t: float
    payload: dict[str, Any] = field(default_factory=dict)
    source: str | None = None

    def __post_init__(self) -> None:
        # Canonicalize to a plain float64 array of shape (3,) so equality
        # / hashing across the rest of the engine never cares whether the
        # caller handed in a list, tuple, or numpy view.
        arr = np.asarray(self.position_m, dtype=np.float64)
        if arr.shape != (3,):
            raise ValueError(
                f"Event.position_m must have shape (3,); got {arr.shape}"
            )
        self.position_m = arr

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly dict (numpy → list)."""
        return {
            "id": self.id,
            "position_m": self.position_m.tolist(),
            "time_t": float(self.time_t),
            "payload": dict(self.payload),
            "source": self.source,
        }
