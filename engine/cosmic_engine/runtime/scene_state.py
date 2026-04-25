"""Snapshot of what the runtime knows after a frame."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


@dataclass
class SceneState:
    """A serializable summary of the current registry + runtime state."""

    julian_date: float
    total_objects: int
    active_objects: int
    object_type_counts: dict[str, int] = field(default_factory=dict)
    truth_level_counts: dict[str, int] = field(default_factory=dict)
    source_counts: dict[str, int] = field(default_factory=dict)
    physics_backend: str = "none"
    perception_enabled: bool = False
    ai_warp_enabled: bool = False
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain dict (only primitives, lists, and dicts)."""
        return {
            "julian_date": self.julian_date,
            "total_objects": self.total_objects,
            "active_objects": self.active_objects,
            "object_type_counts": dict(self.object_type_counts),
            "truth_level_counts": dict(self.truth_level_counts),
            "source_counts": dict(self.source_counts),
            "physics_backend": self.physics_backend,
            "perception_enabled": self.perception_enabled,
            "ai_warp_enabled": self.ai_warp_enabled,
            "notes": list(self.notes),
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        """Return a JSON string of :meth:`to_dict`."""
        return json.dumps(self.to_dict(), indent=indent)
