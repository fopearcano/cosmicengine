"""Rule taxonomy + the four built-in rule subclasses."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from cosmic_engine.reality.rule_context import RuleContext


class RealityRuleDomain(str, Enum):
    """Which subsystem a rule touches."""

    PHYSICS = "physics"
    PERCEPTION = "perception"
    RENDERING = "rendering"
    CAUSALITY = "causality"
    DATA_INTERPRETATION = "data_interpretation"
    SYMBOLIC = "symbolic"


class RealityRuleKind(str, Enum):
    """Whether the rule alters truth, perception, or symbolic overlay."""

    PHYSICAL = "physical"
    PERCEPTUAL = "perceptual"
    SYMBOLIC = "symbolic"
    EXPERIMENTAL = "experimental"


@dataclass(kw_only=True)
class RealityRule:
    """Base class for a single deterministic reality modification.

    Subclasses override :meth:`apply`. Default :meth:`applies_to_context`
    returns ``self.enabled`` so subclasses only need to override it
    when they want spatial / temporal / observer-scoped activation.

    The dataclass uses ``kw_only=True`` so subclasses can introduce
    sensible defaults for ``id`` / ``name`` / ``domain`` / ``kind``
    without breaking field order.
    """

    id: str
    name: str
    domain: RealityRuleDomain
    kind: RealityRuleKind
    priority: int = 0
    enabled: bool = True
    parameters: dict[str, Any] = field(default_factory=dict)
    source: str | None = None
    description: str = ""

    def applies_to_context(self, context: RuleContext) -> bool:
        """Default: apply iff the rule is enabled.

        Subclasses can use ``self.parameters['scope']`` or
        ``context.observer_id`` / ``context.scale_zone`` to scope
        themselves further.
        """
        return bool(self.enabled)

    def apply(self, context: RuleContext) -> RuleContext:
        """Default: clone the context and echo this rule's parameters.

        Parameters land under ``metadata['rules'][self.id]`` so every
        rule — including base-class instances used as declarative
        markers — is fully traceable downstream without needing a
        subclass.
        """
        new_ctx = context.clone()
        if self.parameters:
            rule_meta = new_ctx.metadata.setdefault("rules", {})
            rule_meta[self.id] = dict(self.parameters)
        return new_ctx


# --- A. WarpAmplificationRule ------------------------------------------


@dataclass(kw_only=True)
class WarpAmplificationRule(RealityRule):
    """Multiply the perceived ``warp_factor`` by ``parameters['factor']``.

    Modifies only the rule context; the observer's stored
    ``warp_factor`` is untouched, so this affects one render only.
    """

    id: str = "warp_amplification"
    name: str = "Warp Amplification"
    domain: RealityRuleDomain = RealityRuleDomain.PERCEPTION
    kind: RealityRuleKind = RealityRuleKind.PERCEPTUAL

    def __post_init__(self) -> None:
        # Default factor of 1.0 == no-op; rejects non-positive values
        # so we never invert / zero out the warp.
        factor = float(self.parameters.get("factor", 1.0))
        if factor <= 0.0:
            raise ValueError(
                f"WarpAmplificationRule factor must be positive; got {factor}"
            )
        self.parameters["factor"] = factor

    def apply(self, context: RuleContext) -> RuleContext:
        new_ctx = context.clone()
        factor = float(self.parameters.get("factor", 1.0))
        new_ctx.warp_factor = max(1.0, float(new_ctx.warp_factor) * factor)
        new_ctx.metadata["warp_amplification_factor"] = factor
        return new_ctx


# --- B. RedshiftSymbolicColorRule --------------------------------------


@dataclass(kw_only=True)
class RedshiftSymbolicColorRule(RealityRule):
    """Mark the view as using symbolic-redshift coloring."""

    id: str = "redshift_symbolic_color"
    name: str = "Symbolic Redshift Coloring"
    domain: RealityRuleDomain = RealityRuleDomain.DATA_INTERPRETATION
    kind: RealityRuleKind = RealityRuleKind.SYMBOLIC

    def apply(self, context: RuleContext) -> RuleContext:
        new_ctx = context.clone()
        new_ctx.metadata["color_mapping"] = "symbolic_redshift"
        return new_ctx


# --- C. CausalityRelaxationRule ----------------------------------------


@dataclass(kw_only=True)
class CausalityRelaxationRule(RealityRule):
    """Declare a *relaxed* causality mode without altering visibility yet.

    Future phases can read ``metadata['causality_mode']`` to skip the
    light-cone filter for the affected observer; for now this is a
    declarative marker only.
    """

    id: str = "causality_relaxation"
    name: str = "Causality Relaxation"
    domain: RealityRuleDomain = RealityRuleDomain.CAUSALITY
    kind: RealityRuleKind = RealityRuleKind.EXPERIMENTAL

    def apply(self, context: RuleContext) -> RuleContext:
        new_ctx = context.clone()
        new_ctx.metadata["causality_mode"] = "relaxed"
        return new_ctx


# --- D. NeuralRealityRule ----------------------------------------------


@dataclass(kw_only=True)
class NeuralRealityRule(RealityRule):
    """Mark the view as using a neural perception model.

    Optional ``parameters['model_id']`` is propagated into the
    metadata so the renderer or AI viewer can resolve the right
    model.
    """

    id: str = "neural_reality"
    name: str = "Neural Reality"
    domain: RealityRuleDomain = RealityRuleDomain.PERCEPTION
    kind: RealityRuleKind = RealityRuleKind.PERCEPTUAL

    def apply(self, context: RuleContext) -> RuleContext:
        new_ctx = context.clone()
        new_ctx.metadata["use_neural_perception"] = True
        model_id = self.parameters.get("model_id")
        if model_id is not None:
            new_ctx.metadata["model_id"] = str(model_id)
        return new_ctx
