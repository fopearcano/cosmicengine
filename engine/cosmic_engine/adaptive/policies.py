"""Decision policy that gates feedback recording and update suggestions."""

from __future__ import annotations

from cosmic_engine.adaptive.feedback import FeedbackRecord


class AdaptivePolicy:
    """Bounded, deterministic policy for the adaptive engine.

    Defaults are tuned to be "noisy enough to surface real drift, but
    quiet under normal noise":

    - ``error_threshold = 0.10`` — ignore deviations below 10 %.
    - ``window_size = 32`` — only the last 32 records count toward an
      update suggestion.
    - ``sustained_fraction = 0.5`` — at least half of the windowed
      records must be over-threshold to suggest an update.
    - ``max_updates_per_run = 1`` — at most one suggestion per
      :meth:`AdaptiveEngine.evaluate` call.

    Every threshold is a constructor parameter so tests / demos can
    tune behaviour without monkey-patching.
    """

    def __init__(
        self,
        *,
        error_threshold: float = 0.10,
        window_size: int = 32,
        sustained_fraction: float = 0.5,
        max_updates_per_run: int = 1,
    ) -> None:
        if error_threshold < 0.0:
            raise ValueError("error_threshold must be non-negative")
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        if not (0.0 < sustained_fraction <= 1.0):
            raise ValueError("sustained_fraction must be in (0, 1]")
        if max_updates_per_run <= 0:
            raise ValueError("max_updates_per_run must be positive")
        self.error_threshold = float(error_threshold)
        self.window_size = int(window_size)
        self.sustained_fraction = float(sustained_fraction)
        self.max_updates_per_run = int(max_updates_per_run)

    def should_record(self, feedback: FeedbackRecord) -> bool:
        """Record any feedback whose deviation exceeds the threshold."""
        return feedback.deviation >= self.error_threshold

    def _windowed(
        self,
        feedback_records: list[FeedbackRecord],
        source: str,
    ) -> list[FeedbackRecord]:
        """Return the most-recent ``window_size`` records matching ``source``."""
        filtered = [r for r in feedback_records if r.source == source]
        if not filtered:
            return []
        return filtered[-self.window_size :]

    def should_update_model(
        self,
        feedback_records: list[FeedbackRecord],
        source: str = "spacetime_model",
    ) -> bool:
        """``True`` iff a sustained fraction of the window is over threshold."""
        window = self._windowed(feedback_records, source)
        if len(window) < self.window_size:
            return False
        n_bad = sum(1 for r in window if r.deviation >= self.error_threshold)
        return (n_bad / len(window)) >= self.sustained_fraction

    def should_update_rules(
        self,
        feedback_records: list[FeedbackRecord],
        source: str = "ai_warp",
    ) -> bool:
        """Same heuristic as ``should_update_model`` but for rule sources."""
        window = self._windowed(feedback_records, source)
        if len(window) < self.window_size:
            return False
        n_bad = sum(1 for r in window if r.deviation >= self.error_threshold)
        return (n_bad / len(window)) >= self.sustained_fraction
