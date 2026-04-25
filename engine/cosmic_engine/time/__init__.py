"""Subjective time and causality.

Phase 34 contribution: per-observer proper time, an event store, and a
light-cone-based visibility filter so each observer sees a causally
consistent slice of the universe.

Approximations:
- special-relativistic time dilation via gamma_from_beta.
- weak-field gravitational time dilation: dτ ≈ dt * (1 + Φ/c²),
  with Φ a Newtonian potential.
- retarded time computed as ``t_emit ≈ t_obs - distance / c`` —
  no iterative root-finding for moving sources.
"""

from cosmic_engine.time.causality import (
    compute_retarded_time,
    is_event_visible,
)
from cosmic_engine.time.event import Event
from cosmic_engine.time.event_store import EventStore
from cosmic_engine.time.proper_time import (
    advance_proper_time,
    gamma_from_beta,
    gravitational_potential_weak,
)

__all__ = [
    "Event",
    "EventStore",
    "advance_proper_time",
    "compute_retarded_time",
    "gamma_from_beta",
    "gravitational_potential_weak",
    "is_event_visible",
]
