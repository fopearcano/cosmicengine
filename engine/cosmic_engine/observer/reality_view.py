"""Per-observer perceived reality snapshot."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from cosmic_engine.runtime.scene_state import SceneState


@dataclass
class RealityView:
    """What a single observer perceives at one moment in time.

    Pairs the runtime's :class:`SceneState` (which is shared truth) with
    everything that's specific to *this* observer — the
    representation_type chosen for their scale / view, the rendered
    pixel buffer (or ``None`` if rendering was skipped), and an open
    metadata dict for anything else (warp factor, model labels, blend
    info, etc.).

    Phase 34 added explicit ``proper_time_tau`` / ``coordinate_time_t``
    fields and a ``visible_event_count`` so the per-observer subjective
    timeline is first-class on the wire.
    """

    observer_id: str
    scene_state: SceneState
    representation_type: str
    frame_data: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    proper_time_tau: float = 0.0
    coordinate_time_t: float = 0.0
    visible_event_count: int = 0
    active_rule_ids: list[str] = field(default_factory=list)
    reality_metadata: dict[str, Any] = field(default_factory=dict)
    provenance_summary: dict[str, Any] = field(default_factory=dict)
    audit_warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly dict (frame data is reported by shape)."""
        if self.frame_data is None:
            frame_info: Any = None
        else:
            frame_info = {
                "shape": list(self.frame_data.shape),
                "dtype": str(self.frame_data.dtype),
            }
        return {
            "observer_id": self.observer_id,
            "scene_state": self.scene_state.to_dict(),
            "representation_type": self.representation_type,
            "frame": frame_info,
            "metadata": dict(self.metadata),
            "proper_time_tau": float(self.proper_time_tau),
            "coordinate_time_t": float(self.coordinate_time_t),
            "visible_event_count": int(self.visible_event_count),
            "active_rule_ids": list(self.active_rule_ids),
            "reality_metadata": dict(self.reality_metadata),
            "provenance_summary": dict(self.provenance_summary),
            "audit_warnings": list(self.audit_warnings),
        }

    def summary(self) -> str:
        """One-line human-readable summary."""
        meta = self.metadata
        warp = meta.get("warp_factor")
        model = meta.get("spacetime_model", "none")
        n_obj = meta.get("object_count", "?")
        warp_str = f"warp={warp}" if warp is not None else "warp=?"
        rules_str = ",".join(self.active_rule_ids) or "-"
        n_warn = len(self.audit_warnings)
        return (
            f"observer={self.observer_id} rep={self.representation_type} "
            f"objects={n_obj} {warp_str} spacetime={model} "
            f"τ={self.proper_time_tau:.3f} t={self.coordinate_time_t:.3f} "
            f"events={self.visible_event_count} rules={rules_str} "
            f"warnings={n_warn}"
        )
