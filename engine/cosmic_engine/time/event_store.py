"""Tiny list-backed event registry."""

from __future__ import annotations

from cosmic_engine.time.event import Event


class EventStore:
    """In-memory event log. List-based for now — fine up to ~1e6 events."""

    def __init__(self) -> None:
        self.events: list[Event] = []

    def add_event(self, event: Event) -> None:
        """Append ``event`` to the log."""
        self.events.append(event)

    def list_events(self) -> list[Event]:
        """Return all events in insertion order (defensive copy)."""
        return list(self.events)

    def query_time_window(
        self,
        t_min: float,
        t_max: float,
    ) -> list[Event]:
        """Return events whose ``time_t`` lies in ``[t_min, t_max]``."""
        if t_max < t_min:
            raise ValueError("t_max must be >= t_min")
        return [e for e in self.events if t_min <= e.time_t <= t_max]

    def __len__(self) -> int:
        return len(self.events)
