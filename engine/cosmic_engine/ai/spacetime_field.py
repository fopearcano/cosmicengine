"""Pluggable spacetime curvature model.

Phase 30 contribution: an interface that future learned models can
implement to provide acceleration (or curvature) at a given point in
space — replacing the analytical Schwarzschild formula in
:func:`cosmic_engine.physics.nbody.GRAVITATIONAL_CONSTANT`-driven code
paths.

Default is :class:`NotImplementedError` so subclasses can't silently
no-op. The deterministic analytical path remains the fallback for any
caller that wants a guaranteed result.
"""

from __future__ import annotations

import numpy as np


class SpacetimeFieldModel:
    """Abstract neural spacetime curvature query interface."""

    def query_acceleration(
        self,
        position: np.ndarray,
        direction: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return the acceleration vector at ``position`` (3 floats)."""
        raise NotImplementedError

    def confidence(self) -> float:
        """Self-rated trust in ``[0, 1]``."""
        raise NotImplementedError
