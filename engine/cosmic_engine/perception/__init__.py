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
from cosmic_engine.perception.vectorized_ai_transform import (
    apply_batch_ai_warp,
    build_photon_warp_input,
    validate_warp_output,
)
from cosmic_engine.perception.vectorized_transform import (
    transform_photon_field_batch,
)

__all__ = [
    "ObserverState",
    "apply_batch_ai_warp",
    "apply_brightness_warp",
    "apply_color_warp",
    "apply_direction_warp",
    "build_photon_warp_input",
    "transform_photon_field",
    "transform_photon_field_batch",
    "transform_photon_sample",
    "validate_warp_output",
]
