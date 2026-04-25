"""General-relativity-inspired rendering effects.

Phase 28 contribution: weak-field gravitational lensing and a
Schwarzschild-style black-hole renderer. None of this is real GR
geodesic integration — it's the ``α = 4GM/(c²b)`` deflection angle
applied as a small rotation toward the lens, plus event-horizon
absorption — but it is enough for real-time-capable splat rendering
that visibly bends light.
"""

from ai_viewer.neural_field.gr.black_hole import (
    BlackHole,
    apply_black_hole_to_points,
)
from ai_viewer.neural_field.gr.lensing import (
    apply_lensing,
    apply_lensing_to_points,
    compute_deflection_angle,
)

__all__ = [
    "BlackHole",
    "apply_black_hole_to_points",
    "apply_lensing",
    "apply_lensing_to_points",
    "compute_deflection_angle",
]
