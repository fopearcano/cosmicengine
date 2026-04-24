"""Observer-side perception transforms.

Phase 4 contribution: deterministic, approximate "relativistic-flavored"
reshaping of an existing photon field. Covers direction aberration,
brightness beaming, and a simple color Doppler tint, with a
``warp_factor`` knob that extends each effect beyond its physical
regime for visualization.

Not a full relativistic renderer. No AI. Pure Python math.
"""

from cosmic_engine.perception.observer import ObserverState
from cosmic_engine.perception.transform import (
    apply_brightness_warp,
    apply_color_warp,
    apply_direction_warp,
    transform_photon_field,
    transform_photon_sample,
)
from cosmic_engine.perception.vectorized_transform import (
    transform_photon_field_batch,
)

__all__ = [
    "ObserverState",
    "apply_brightness_warp",
    "apply_color_warp",
    "apply_direction_warp",
    "transform_photon_field",
    "transform_photon_field_batch",
    "transform_photon_sample",
]
