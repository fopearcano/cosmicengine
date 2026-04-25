"""Built-in :class:`UniverseSpec` presets + rule-id resolver."""

from __future__ import annotations

from cosmic_engine.reality.rule import (
    CausalityRelaxationRule,
    NeuralRealityRule,
    RealityRule,
    RedshiftSymbolicColorRule,
    WarpAmplificationRule,
)
from cosmic_engine.synthesis.universe_spec import UniverseSpec


def create_standard_physics_universe(seed: int) -> UniverseSpec:
    """Strict, scientific-mode universe.

    No symbolic rules. Bounded by physically-motivated mass /
    velocity caps.
    """
    return UniverseSpec(
        id=f"standard_physics_{seed}",
        seed=seed,
        description="ΛCDM-like seeded universe with strict caps and no rules",
        scale_limits=(1.0e10, 1.0e22),
        initial_conditions={
            "object_count": 64,
            "mass_range_kg": (1.0e26, 1.0e30),
            "velocity_dispersion_m_s": 1.0e4,
            "include_central_mass": True,
            "central_mass_kg": 1.989e30,
            "object_type": "star",
        },
        physics_model="lambda_cdm",
        spacetime_model="analytical",
        rule_ids=[],
        constraints={
            "max_mass_kg": 1.0e35,
            "max_velocity_m_s": 1.0e6,  # 1 000 km/s
        },
    )


def create_hyperwarp_universe(seed: int) -> UniverseSpec:
    """Stylized universe with neural spacetime + warp amplification."""
    return UniverseSpec(
        id=f"hyperwarp_{seed}",
        seed=seed,
        description="High warp_factor, neural spacetime, perception rules",
        scale_limits=(1.0e9, 1.0e21),
        initial_conditions={
            "object_count": 96,
            "mass_range_kg": (1.0e25, 5.0e29),
            "velocity_dispersion_m_s": 1.0e5,
            "include_central_mass": True,
            "central_mass_kg": 4.0e30,
            "object_type": "star",
        },
        physics_model="newtonian",
        spacetime_model="neural",
        rule_ids=[
            "warp_amplification",
            "neural_reality",
            "redshift_symbolic_color",
        ],
        constraints={
            "max_mass_kg": 1.0e40,
            "max_velocity_m_s": 5.0e7,  # 0.17 c
        },
    )


def create_symbolic_universe(seed: int) -> UniverseSpec:
    """Experimental universe: relaxed causality + symbolic redshift."""
    return UniverseSpec(
        id=f"symbolic_{seed}",
        seed=seed,
        description="Symbolic / experimental rules, relaxed causality",
        scale_limits=(1.0e11, 1.0e22),
        initial_conditions={
            "object_count": 128,
            "mass_range_kg": (1.0e22, 1.0e29),
            "velocity_dispersion_m_s": 5.0e4,
            "include_central_mass": False,
            "object_type": "galaxy",
        },
        physics_model="none",
        spacetime_model="none",
        rule_ids=[
            "causality_relaxation",
            "redshift_symbolic_color",
            "neural_reality",
        ],
        constraints={
            "max_velocity_m_s": 2.0e8,  # 0.67 c
        },
    )


# --- rule-id → RealityRule -------------------------------------------------


_RULE_FACTORIES = {
    "warp_amplification": lambda: WarpAmplificationRule(
        priority=10, parameters={"factor": 6.0},
    ),
    "neural_reality": lambda: NeuralRealityRule(
        priority=5, parameters={"model_id": "synthetic_neural_v1"},
    ),
    "redshift_symbolic_color": lambda: RedshiftSymbolicColorRule(priority=4),
    "causality_relaxation": lambda: CausalityRelaxationRule(priority=2),
}


def _resolve_rule(rule_id: str) -> RealityRule | None:
    """Return a :class:`RealityRule` for a known id, or ``None``."""
    factory = _RULE_FACTORIES.get(rule_id)
    if factory is None:
        return None
    return factory()
