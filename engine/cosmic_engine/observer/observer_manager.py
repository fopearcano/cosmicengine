"""Registry of named :class:`Observer` instances."""

from __future__ import annotations

from cosmic_engine.observer.observer import Observer


class ObserverManager:
    """Hold all active observers; lookups are O(1) by id."""

    def __init__(self) -> None:
        self.observers: dict[str, Observer] = {}

    def add_observer(self, observer: Observer) -> None:
        """Register ``observer``. Raises :class:`ValueError` on duplicate id."""
        observer.validate()
        if observer.id in self.observers:
            raise ValueError(
                f"Observer id {observer.id!r} is already registered"
            )
        self.observers[observer.id] = observer

    def remove_observer(self, observer_id: str) -> None:
        """Drop ``observer_id``. Raises :class:`KeyError` if unknown."""
        if observer_id not in self.observers:
            raise KeyError(observer_id)
        del self.observers[observer_id]

    def get_observer(self, observer_id: str) -> Observer:
        """Return the observer registered under ``observer_id``."""
        if observer_id not in self.observers:
            raise KeyError(observer_id)
        return self.observers[observer_id]

    def list_observers(self) -> list[Observer]:
        """Return all observers, sorted by id for determinism."""
        return [self.observers[i] for i in sorted(self.observers)]

    def __len__(self) -> int:
        return len(self.observers)

    def __contains__(self, observer_id: object) -> bool:
        return observer_id in self.observers
