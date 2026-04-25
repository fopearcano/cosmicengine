"""Emergent reality layer.

Phase 35 contribution: a deterministic, inspectable rule engine that
can modify perception / rendering / interpretation per observer or
region without mutating the underlying universe state.

Every rule declares:
- a *domain* (which subsystem it touches)
- a *kind* (physical / perceptual / symbolic / experimental)
- a *priority* (deterministic application order)
- whether it's *enabled*
- *parameters* and a free-form *description*

The runtime evaluates the rule engine inside ``render_for_observer``
and threads the resulting :class:`RuleContext` through the rest of
the render. Every applied rule's id is recorded on the produced
:class:`RealityView` so consumers can audit exactly what shaped the
view.
"""

from cosmic_engine.reality.presets import (
    create_blackhole_perception_preset,
    create_hypertravel_reality_preset,
    create_scientific_reality_preset,
)
from cosmic_engine.reality.rule import (
    CausalityRelaxationRule,
    NeuralRealityRule,
    RealityRule,
    RealityRuleDomain,
    RealityRuleKind,
    RedshiftSymbolicColorRule,
    WarpAmplificationRule,
)
from cosmic_engine.reality.rule_context import RuleContext
from cosmic_engine.reality.rule_engine import RealityRuleEngine

__all__ = [
    "CausalityRelaxationRule",
    "NeuralRealityRule",
    "RealityRule",
    "RealityRuleDomain",
    "RealityRuleEngine",
    "RealityRuleKind",
    "RedshiftSymbolicColorRule",
    "RuleContext",
    "WarpAmplificationRule",
    "create_blackhole_perception_preset",
    "create_hypertravel_reality_preset",
    "create_scientific_reality_preset",
]
