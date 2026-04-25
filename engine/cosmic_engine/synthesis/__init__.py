"""Reality synthesis.

Phase 38 contribution: a deterministic, seeded generator that builds an
entire :class:`CosmicRuntime` from a declarative :class:`UniverseSpec`.
Generated content is explicitly tagged
``TruthLevel.SYNTHETIC_GENERATED`` so audit consumers can never
confuse it with observed or catalog-imported data.

Hard guarantees:
- Same seed → same generated state, byte for byte.
- Real-data pipelines (``CosmicRuntime.load_sample_data``) are never
  touched by anything in this package.
- Constraints are enforced at generation time and again at registry
  install — invariant checks log adjustments rather than failing
  silently.
"""

from cosmic_engine.synthesis.constraints import (
    Constraint,
    MaxMassConstraint,
    MaxVelocityConstraint,
    apply_constraints,
)
from cosmic_engine.synthesis.generator import (
    UniverseGenerator,
    create_runtime_from_spec,
)
from cosmic_engine.synthesis.presets import (
    create_hyperwarp_universe,
    create_standard_physics_universe,
    create_symbolic_universe,
)
from cosmic_engine.synthesis.universe_spec import UniverseSpec

__all__ = [
    "Constraint",
    "MaxMassConstraint",
    "MaxVelocityConstraint",
    "UniverseGenerator",
    "UniverseSpec",
    "apply_constraints",
    "create_hyperwarp_universe",
    "create_runtime_from_spec",
    "create_standard_physics_universe",
    "create_symbolic_universe",
]
