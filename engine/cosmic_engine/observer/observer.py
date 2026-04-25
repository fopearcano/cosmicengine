"""Observer state for the multi-observer reality system."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from cosmic_engine.ai.base import AIWarpModel
from cosmic_engine.ai.spacetime_field import SpacetimeFieldModel
from cosmic_engine.core.units import SPEED_OF_LIGHT_M_S
from cosmic_engine.core.vector import Vector3
from cosmic_engine.time.proper_time import (
    advance_proper_time,
    gravitational_potential_weak,
)


_ZERO = Vector3.zero()


@dataclass
class Observer:
    """Per-observer pose, motion, perception, and model bindings.

    Mirrors :class:`cosmic_engine.perception.ObserverState` (kept around
    for the single-observer perception pipeline) but adds an ``id``,
    optional ``spacetime_model`` / ``ai_warp_model`` slots, and a free
    ``config`` dict so callers can attach renderer / AI options without
    growing the dataclass.

    Phase 34: also carries its own ``proper_time_tau`` and
    ``coordinate_time_t`` so each observer has a subjective timeline.
    """

    id: str
    position_m: Vector3
    velocity_m_s: Vector3
    forward: Vector3
    up: Vector3
    warp_factor: float = 1.0
    spacetime_model: SpacetimeFieldModel | None = None
    ai_warp_model: AIWarpModel | None = None
    config: dict[str, Any] = field(default_factory=dict)
    proper_time_tau: float = 0.0
    coordinate_time_t: float = 0.0

    def speed_magnitude(self) -> float:
        """Return |velocity| in m/s."""
        v = self.velocity_m_s
        return math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z)

    def beta(self) -> float:
        """Return v/c, dimensionless."""
        return self.speed_magnitude() / SPEED_OF_LIGHT_M_S

    def advance_time(
        self,
        delta_t: float,
        masses_for_potential: list[tuple[np.ndarray, float]] | None = None,
    ) -> None:
        """Advance both clocks by a coordinate-time step ``delta_t``.

        ``masses_for_potential`` is an optional list of
        ``(position_xyz, mass_kg)`` pairs used to compute the local
        gravitational potential for weak-field time dilation. Pass
        ``None`` to skip the GR factor and use SR-only.
        """
        if delta_t < 0.0:
            raise ValueError("delta_t must be non-negative")
        self.coordinate_time_t = float(self.coordinate_time_t) + delta_t
        beta = self.beta()
        phi = None
        if masses_for_potential:
            position = np.array(
                [self.position_m.x, self.position_m.y, self.position_m.z],
                dtype=np.float64,
            )
            phi = gravitational_potential_weak(position, masses_for_potential)
        self.proper_time_tau = advance_proper_time(
            self.proper_time_tau, delta_t, beta, gravitational_potential=phi
        )

    def validate(self) -> None:
        """Raise :class:`ValueError` if any field violates the invariants."""
        if not self.id:
            raise ValueError("Observer.id must be a non-empty string")
        if self.warp_factor < 1.0:
            raise ValueError(
                f"warp_factor must be >= 1.0; got {self.warp_factor}"
            )
        if self.beta() >= 1.0:
            raise ValueError("observer velocity must be subluminal (beta < 1)")
        if self.forward == _ZERO:
            raise ValueError("forward must not be the zero vector")
        if self.up == _ZERO:
            raise ValueError("up must not be the zero vector")
