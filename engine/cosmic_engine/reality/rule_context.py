"""Immutable-by-convention context object passed through the rule chain."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

from cosmic_engine.core.vector import Vector3


@dataclass
class RuleContext:
    """Snapshot of everything a rule may inspect or modify.

    The runtime fills this in once per render, then hands it to
    :meth:`RealityRuleEngine.evaluate` which threads it through every
    enabled rule. Rules return a fresh context (via :meth:`clone`) so
    the original observer / runtime state is never mutated.
    """

    observer_id: str
    observer_position_m: Vector3
    observer_velocity_m_s: Vector3
    warp_factor: float
    coordinate_time_t: float
    proper_time_tau: float
    scale_zone: str | None = None
    truth_level_counts: dict[str, int] = field(default_factory=dict)
    active_rule_ids: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def clone(self) -> "RuleContext":
        """Return a deep copy. Mutating the result never touches the original."""
        return RuleContext(
            observer_id=self.observer_id,
            observer_position_m=Vector3(
                self.observer_position_m.x,
                self.observer_position_m.y,
                self.observer_position_m.z,
            ),
            observer_velocity_m_s=Vector3(
                self.observer_velocity_m_s.x,
                self.observer_velocity_m_s.y,
                self.observer_velocity_m_s.z,
            ),
            warp_factor=float(self.warp_factor),
            coordinate_time_t=float(self.coordinate_time_t),
            proper_time_tau=float(self.proper_time_tau),
            scale_zone=self.scale_zone,
            truth_level_counts=dict(self.truth_level_counts),
            active_rule_ids=list(self.active_rule_ids),
            metadata=copy.deepcopy(self.metadata),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly dict (Vector3 → list)."""
        return {
            "observer_id": self.observer_id,
            "observer_position_m": [
                self.observer_position_m.x,
                self.observer_position_m.y,
                self.observer_position_m.z,
            ],
            "observer_velocity_m_s": [
                self.observer_velocity_m_s.x,
                self.observer_velocity_m_s.y,
                self.observer_velocity_m_s.z,
            ],
            "warp_factor": float(self.warp_factor),
            "coordinate_time_t": float(self.coordinate_time_t),
            "proper_time_tau": float(self.proper_time_tau),
            "scale_zone": self.scale_zone,
            "truth_level_counts": dict(self.truth_level_counts),
            "active_rule_ids": list(self.active_rule_ids),
            "metadata": copy.deepcopy(self.metadata),
        }
