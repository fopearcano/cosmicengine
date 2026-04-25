"""Discrepancy / feedback record."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class FeedbackRecord:
    """One observed discrepancy between an analytical reference and a model.

    ``metric_value`` is what the model produced; ``expected_value`` is
    the analytical reference (or ``None`` when only the metric matters,
    e.g. a non-comparative quality metric). ``deviation`` is the
    normalized (typically relative-L2) error returned by one of the
    helpers in :mod:`cosmic_engine.adaptive.metrics`.

    ``source`` identifies which model produced the discrepancy
    (``"spacetime_model"``, ``"ai_warp"``, …). ``context`` is a free
    dict for whatever the runtime knows at the time (rule preset,
    observer pose, etc.).
    """

    id: str
    observer_id: str
    timestamp_t: float
    context: dict[str, Any]
    metric_name: str
    metric_value: float
    expected_value: float | None
    deviation: float
    source: str
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("FeedbackRecord.id must be a non-empty string")
        if not self.metric_name:
            raise ValueError("FeedbackRecord.metric_name must be non-empty")
        if not self.source:
            raise ValueError("FeedbackRecord.source must be non-empty")
        if self.deviation < 0.0:
            raise ValueError(
                f"FeedbackRecord.deviation must be non-negative; "
                f"got {self.deviation}"
            )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly dict copy."""
        return {
            "id": self.id,
            "observer_id": self.observer_id,
            "timestamp_t": float(self.timestamp_t),
            "context": dict(self.context),
            "metric_name": self.metric_name,
            "metric_value": float(self.metric_value),
            "expected_value": (
                None if self.expected_value is None else float(self.expected_value)
            ),
            "deviation": float(self.deviation),
            "source": self.source,
            "notes": self.notes,
        }
