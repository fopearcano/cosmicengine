"""Single-Gaussian point primitive for the splat renderer."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class GaussianPoint:
    """A 3D isotropic Gaussian point in world coordinates.

    ``position`` is in meters (or any consistent world unit).
    ``color`` is RGB in ``[0, 255]``. ``intensity`` is a scalar that
    multiplies the Gaussian footprint at render time. ``sigma`` is the
    world-space spread; the renderer projects it to screen-space sigma
    using the camera's distance to the point.

    The optional ``object_id`` / ``truth_level`` / ``metadata`` fields
    let warp pipelines preserve provenance through the field — useful
    for debugging which galaxy a splat came from after a hyper-warp
    moved its apparent position.
    """

    position: np.ndarray  # shape (3,)
    color: np.ndarray     # shape (3,)
    intensity: float
    sigma: float
    object_id: str | None = None
    truth_level: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.position = np.asarray(self.position, dtype=np.float64).reshape(3)
        self.color = np.asarray(self.color, dtype=np.float64).reshape(3)
        self.intensity = float(self.intensity)
        self.sigma = float(self.sigma)
