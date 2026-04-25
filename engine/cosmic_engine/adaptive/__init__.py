"""Self-improving / adaptive feedback layer.

Phase 37 contribution: an opt-in subsystem that measures
discrepancies between analytical physics and neural approximations,
logs structured feedback, and *suggests* model / rule adjustments.

Hard guarantees:
- No automatic mutation of core physics, neural models, or rules.
- Deterministic. No randomness anywhere.
- Lightweight: feedback log is a Python list capped at the policy's
  ``window_size``.
- Fully auditable: every emitted suggestion records the feedback ids
  that motivated it.
"""

from cosmic_engine.adaptive.adaptive_engine import AdaptiveEngine
from cosmic_engine.adaptive.feedback import FeedbackRecord
from cosmic_engine.adaptive.metrics import (
    compute_acceleration_error,
    compute_brightness_error,
    compute_direction_error,
)
from cosmic_engine.adaptive.policies import AdaptivePolicy

__all__ = [
    "AdaptiveEngine",
    "AdaptivePolicy",
    "FeedbackRecord",
    "compute_acceleration_error",
    "compute_brightness_error",
    "compute_direction_error",
]
