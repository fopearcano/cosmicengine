"""Light-cone visibility and retarded-time helpers."""

from __future__ import annotations

import numpy as np

from cosmic_engine.core.units import SPEED_OF_LIGHT_M_S
from cosmic_engine.time.event import Event


def is_event_visible(
    event: Event,
    observer_position: np.ndarray,
    observer_time_t: float,
) -> bool:
    """Return ``True`` if ``event`` is in the past light cone of the observer.

    Condition: ``c · (t_obs − t_event) >= |x_obs − x_event|``, with the
    additional constraint ``t_obs >= t_event`` (no perceiving the
    future). Numerically, ``c · Δt − distance`` is allowed to be very
    slightly negative to absorb floating-point drift on the cone.
    """
    p = np.asarray(observer_position, dtype=np.float64)
    if p.shape != (3,):
        raise ValueError("observer_position must have shape (3,)")
    dt = float(observer_time_t) - float(event.time_t)
    if dt < 0.0:
        return False
    distance = float(np.linalg.norm(event.position_m - p))
    # 1 ns of slack on the boundary so an event placed exactly on the
    # cone counts as visible despite float roundoff.
    slack = 1.0e-9
    return SPEED_OF_LIGHT_M_S * dt + slack >= distance


def compute_retarded_time(
    observer_position: np.ndarray,
    observer_time_t: float,
    source_position: np.ndarray,
) -> float:
    """Return ``t_emit ≈ t_obs − distance / c`` (static-source approx).

    For sources moving slowly compared to the observed light delay
    this is the correct retarded time to first order; we don't iterate
    on a moving emitter to keep the math closed-form.
    """
    p = np.asarray(observer_position, dtype=np.float64)
    s = np.asarray(source_position, dtype=np.float64)
    if p.shape != (3,) or s.shape != (3,):
        raise ValueError("positions must have shape (3,)")
    distance = float(np.linalg.norm(s - p))
    return float(observer_time_t) - distance / SPEED_OF_LIGHT_M_S
