"""The adaptive engine: collects feedback, suggests bounded updates."""

from __future__ import annotations

from typing import Any

from cosmic_engine.adaptive.feedback import FeedbackRecord
from cosmic_engine.adaptive.policies import AdaptivePolicy


class AdaptiveEngine:
    """Aggregate :class:`FeedbackRecord` instances and emit suggestions.

    The engine never executes any change itself. Every call returns a
    list of suggestion dicts; humans / external tools decide whether
    to act on them. Suggestions cite the feedback ids that motivated
    them so the audit trail is closed.
    """

    def __init__(self, policy: AdaptivePolicy | None = None) -> None:
        self.policy: AdaptivePolicy = policy or AdaptivePolicy()
        self.feedback_log: list[FeedbackRecord] = []
        # Cap the in-memory log at 4× window so we don't grow unbounded
        # over a long run; older records get dropped FIFO.
        self._log_capacity: int = max(128, 4 * self.policy.window_size)
        # Counter of suggestions emitted this "run" (resets when the
        # caller invokes :meth:`reset_run`).
        self._updates_this_run: int = 0

    # --- recording --------------------------------------------------------

    def record_feedback(self, record: FeedbackRecord) -> bool:
        """Append ``record`` if the policy says so. Returns whether it was kept."""
        if not self.policy.should_record(record):
            return False
        self.feedback_log.append(record)
        if len(self.feedback_log) > self._log_capacity:
            # Drop the oldest record; FIFO bound on memory.
            self.feedback_log = self.feedback_log[-self._log_capacity :]
        return True

    def reset_run(self) -> None:
        """Reset the per-run suggestion counter (call between distinct sessions)."""
        self._updates_this_run = 0

    # --- evaluation -------------------------------------------------------

    def evaluate(self) -> list[dict[str, Any]]:
        """Apply the policy to the log and return any suggested actions.

        Suggestions are deterministic given the same feedback log:
        they are emitted in a fixed order (model first, rules second)
        and capped at ``policy.max_updates_per_run``.
        """
        suggestions: list[dict[str, Any]] = []
        budget = self.policy.max_updates_per_run - self._updates_this_run
        if budget <= 0:
            return suggestions

        # Model retraining suggestion (default source: spacetime_model).
        if self.policy.should_update_model(self.feedback_log):
            window = self._window_for_source("spacetime_model")
            mean_dev = sum(r.deviation for r in window) / max(len(window), 1)
            suggestions.append({
                "action": "retrain_spacetime_model",
                "source": "spacetime_model",
                "reason": (
                    f"sustained deviation across {len(window)} samples; "
                    f"mean deviation = {mean_dev:.4f} "
                    f"(threshold = {self.policy.error_threshold})"
                ),
                "feedback_ids": [r.id for r in window],
                "mean_deviation": mean_dev,
                "window_size": len(window),
            })
            self._updates_this_run += 1
            budget -= 1
            if budget <= 0:
                return suggestions

        # Rule adjustment suggestion (default source: ai_warp).
        if self.policy.should_update_rules(self.feedback_log):
            window = self._window_for_source("ai_warp")
            mean_dev = sum(r.deviation for r in window) / max(len(window), 1)
            suggestions.append({
                "action": "adjust_rule",
                "source": "ai_warp",
                "reason": (
                    f"sustained deviation across {len(window)} samples; "
                    f"mean deviation = {mean_dev:.4f}"
                ),
                "feedback_ids": [r.id for r in window],
                "mean_deviation": mean_dev,
                "window_size": len(window),
            })
            self._updates_this_run += 1
        return suggestions

    def summary(self) -> dict[str, Any]:
        """Return a small dict describing the current log state."""
        if not self.feedback_log:
            return {
                "record_count": 0,
                "by_source": {},
                "max_deviation": 0.0,
                "mean_deviation": 0.0,
            }
        by_source: dict[str, int] = {}
        for r in self.feedback_log:
            by_source[r.source] = by_source.get(r.source, 0) + 1
        max_dev = max(r.deviation for r in self.feedback_log)
        mean_dev = sum(r.deviation for r in self.feedback_log) / len(
            self.feedback_log
        )
        return {
            "record_count": len(self.feedback_log),
            "by_source": by_source,
            "max_deviation": float(max_dev),
            "mean_deviation": float(mean_dev),
        }

    # --- helpers ----------------------------------------------------------

    def _window_for_source(self, source: str) -> list[FeedbackRecord]:
        filtered = [r for r in self.feedback_log if r.source == source]
        return filtered[-self.policy.window_size :]
