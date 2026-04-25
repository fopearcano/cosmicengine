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
    """

    observer_id: str
    scene_state: SceneState
    representation_type: str
    frame_data: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

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
        }

    def summary(self) -> str:
        """One-line human-readable summary."""
        meta = self.metadata
        warp = meta.get("warp_factor")
        model = meta.get("spacetime_model", "none")
        n_obj = meta.get("object_count", "?")
        warp_str = f"warp={warp}" if warp is not None else "warp=?"
        return (
            f"observer={self.observer_id} rep={self.representation_type} "
            f"objects={n_obj} {warp_str} spacetime={model}"
        )
