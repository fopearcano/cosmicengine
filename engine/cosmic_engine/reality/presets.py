"""Built-in :class:`RealityRule` presets."""

from __future__ import annotations

from cosmic_engine.reality.rule import (
    CausalityRelaxationRule,
    NeuralRealityRule,
    RealityRule,
    RealityRuleDomain,
    RealityRuleKind,
    RedshiftSymbolicColorRule,
    WarpAmplificationRule,
)


def create_scientific_reality_preset() -> list[RealityRule]:
    """Strict default: empty rule set so render uses raw, deterministic state.

    No symbolic distortion, no causality relaxation, no perception
    overrides. The :class:`RealityView` produced under this preset
    contains no active rule ids, which is the inspectable signal that
    the underlying truth metadata has *not* been altered.
    """
    return []


def create_hypertravel_reality_preset() -> list[RealityRule]:
    """Stylized "hypertravel" preset: amplified warp + symbolic + neural.

    Returns 4 rules in a deterministic order. None of them mutate
    truth or provenance; every effect lands in the rule context's
    metadata or in the per-render ``warp_factor``.
    """
    return [
        WarpAmplificationRule(
            priority=10,
            parameters={"factor": 4.0},
            description="boost warp_factor x4 for this render only",
        ),
        RedshiftSymbolicColorRule(
            priority=5,
            description="recolor objects by symbolic redshift bucket",
        ),
        NeuralRealityRule(
            priority=3,
            parameters={"model_id": "neural_perception_v1"},
            description="declare neural perception override",
        ),
        CausalityRelaxationRule(
            priority=1,
            description="declare relaxed causality (visibility unchanged)",
        ),
    ]


def create_blackhole_perception_preset() -> list[RealityRule]:
    """Stylized "near a black hole" preset: neural spacetime + spectral shift.

    The lensing-emphasis multiplier is dropped into rule metadata so
    the renderer can pick it up without the rule object knowing about
    renderer internals.
    """
    return [
        NeuralRealityRule(
            id="neural_blackhole_perception",
            name="Neural Black-Hole Perception",
            priority=10,
            parameters={"model_id": "neural_spacetime_blackhole"},
            description="enable neural spacetime field for perception",
        ),
        RealityRule(
            id="lensing_emphasis",
            name="Lensing Emphasis",
            domain=RealityRuleDomain.RENDERING,
            kind=RealityRuleKind.PERCEPTUAL,
            priority=5,
            parameters={"emphasis": 2.0},
            description="multiplier for gravitational lensing weight",
        ),
        RedshiftSymbolicColorRule(
            id="spectral_shift_blackhole",
            name="Spectral Shift",
            priority=3,
            description="symbolic spectral shift overlay near horizon",
        ),
    ]
