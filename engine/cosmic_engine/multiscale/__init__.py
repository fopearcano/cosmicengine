"""Hierarchical multi-scale universe management.

Phase 32 contribution: a scale-aware layer that picks the right
representation for the observer's current zoom level — galaxy
field at intergalactic scales, star field at interstellar scales,
N-body at stellar-system scales, density / neural fields at the
sub-stellar scale — and blends adjacent zones at boundaries so
navigation feels continuous instead of stepwise.
"""

from cosmic_engine.multiscale.representation import (
    REPRESENTATION_TYPES,
    get_representation_for_zone,
)
from cosmic_engine.multiscale.scale_manager import (
    DEFAULT_ZONES,
    ScaleManager,
)
from cosmic_engine.multiscale.scale_zone import ScaleZone
from cosmic_engine.multiscale.transition import (
    blend_representations,
    compute_transition_alpha,
)

__all__ = [
    "DEFAULT_ZONES",
    "REPRESENTATION_TYPES",
    "ScaleManager",
    "ScaleZone",
    "blend_representations",
    "compute_transition_alpha",
    "get_representation_for_zone",
]
