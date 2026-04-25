"""Tests for the Phase 38 reality synthesis layer."""

from __future__ import annotations

import pytest

from cosmic_engine.core.truth import TruthLevel
from cosmic_engine.synthesis import (
    Constraint,
    MaxMassConstraint,
    MaxVelocityConstraint,
    UniverseGenerator,
    UniverseSpec,
    apply_constraints,
    create_hyperwarp_universe,
    create_runtime_from_spec,
    create_standard_physics_universe,
    create_symbolic_universe,
)


# --- UniverseSpec ------------------------------------------------------


def test_spec_defaults_validate():
    spec = UniverseSpec(id="u1", seed=0)
    spec.validate()


def test_spec_rejects_empty_id():
    with pytest.raises(ValueError):
        UniverseSpec(id="", seed=0).validate()


def test_spec_rejects_negative_seed():
    with pytest.raises(ValueError):
        UniverseSpec(id="u", seed=-1).validate()


def test_spec_rejects_inverted_scale_limits():
    with pytest.raises(ValueError):
        UniverseSpec(id="u", seed=0, scale_limits=(1.0e22, 1.0e10)).validate()


def test_spec_rejects_unknown_physics_model():
    with pytest.raises(ValueError):
        UniverseSpec(id="u", seed=0, physics_model="quantum_gravity").validate()


def test_spec_rejects_unknown_spacetime_model():
    with pytest.raises(ValueError):
        UniverseSpec(id="u", seed=0, spacetime_model="loop").validate()


def test_spec_rejects_negative_object_count():
    with pytest.raises(ValueError):
        UniverseSpec(
            id="u", seed=0,
            initial_conditions={"object_count": -1},
        ).validate()


def test_spec_rejects_invalid_mass_range():
    with pytest.raises(ValueError):
        UniverseSpec(
            id="u", seed=0,
            initial_conditions={"mass_range_kg": (1e30, 1e25)},
        ).validate()


# --- Constraints ------------------------------------------------------


def test_max_mass_constraint_rejects_invalid_limit():
    with pytest.raises(ValueError):
        MaxMassConstraint(0.0)


def test_max_mass_constraint_clamps_and_logs():
    state = {"objects": [
        {"id": "a", "mass_kg": 1.0e40},
        {"id": "b", "mass_kg": 1.0e30},
    ]}
    c = MaxMassConstraint(1.0e35)
    assert c.check(state) is False
    state, notes = c.enforce(state)
    assert state["objects"][0]["mass_kg"] == 1.0e35
    assert state["objects"][1]["mass_kg"] == 1.0e30  # untouched
    assert any("clamped a" in n for n in notes)


def test_max_velocity_constraint_rejects_superluminal():
    from cosmic_engine.core.units import SPEED_OF_LIGHT_M_S
    with pytest.raises(ValueError):
        MaxVelocityConstraint(SPEED_OF_LIGHT_M_S)
    with pytest.raises(ValueError):
        MaxVelocityConstraint(-1.0)


def test_max_velocity_constraint_clamps_proportionally():
    state = {"objects": [
        {"id": "a", "velocity_m_s": [3.0e8, 0.0, 0.0]},  # 1c
    ]}
    c = MaxVelocityConstraint(1.5e8)  # 0.5c
    assert c.check(state) is False
    state, notes = c.enforce(state)
    v = state["objects"][0]["velocity_m_s"]
    speed = (v[0] ** 2 + v[1] ** 2 + v[2] ** 2) ** 0.5
    assert speed == pytest.approx(1.5e8)
    assert notes


def test_apply_constraints_aggregates_notes():
    state = {"objects": [
        {"id": "a", "mass_kg": 1.0e40, "velocity_m_s": [3.0e8, 0.0, 0.0]},
    ]}
    state = apply_constraints(state, [
        MaxMassConstraint(1.0e35),
        MaxVelocityConstraint(1.0e8),
    ])
    assert len(state["constraint_notes"]) == 2
    assert state["objects"][0]["mass_kg"] == 1.0e35


# --- Generator: deterministic output ----------------------------------


def test_generator_deterministic_state_for_same_seed():
    spec = UniverseSpec(
        id="u", seed=99,
        initial_conditions={"object_count": 16, "include_central_mass": True},
    )
    a = UniverseGenerator(spec).generate_initial_state()
    b = UniverseGenerator(spec).generate_initial_state()
    assert [o["id"] for o in a["objects"]] == [o["id"] for o in b["objects"]]
    for ao, bo in zip(a["objects"], b["objects"]):
        assert ao["position_m"] == bo["position_m"]
        assert ao["velocity_m_s"] == bo["velocity_m_s"]
        assert ao["mass_kg"] == bo["mass_kg"]


def test_generator_different_seed_diverges():
    a = UniverseGenerator(
        UniverseSpec(id="u", seed=1, initial_conditions={"object_count": 8})
    ).generate_initial_state()
    b = UniverseGenerator(
        UniverseSpec(id="u", seed=2, initial_conditions={"object_count": 8})
    ).generate_initial_state()
    # Same number of objects but different positions.
    assert len(a["objects"]) == len(b["objects"])
    assert a["objects"][1]["position_m"] != b["objects"][1]["position_m"]


def test_generator_zero_objects_only_central_mass():
    spec = UniverseSpec(
        id="u", seed=0,
        initial_conditions={"object_count": 0, "include_central_mass": True},
    )
    state = UniverseGenerator(spec).generate_initial_state()
    assert len(state["objects"]) == 1
    assert state["objects"][0]["id"].endswith("__center")


def test_generator_no_central_mass_option():
    spec = UniverseSpec(
        id="u", seed=0,
        initial_conditions={"object_count": 4, "include_central_mass": False},
    )
    state = UniverseGenerator(spec).generate_initial_state()
    assert len(state["objects"]) == 4
    assert all(not o["id"].endswith("__center") for o in state["objects"])


def test_generator_applies_spec_constraints():
    spec = UniverseSpec(
        id="u", seed=0,
        initial_conditions={
            "object_count": 4,
            "mass_range_kg": (1.0e40, 2.0e40),  # all over the cap
        },
        constraints={"max_mass_kg": 1.0e35},
    )
    state = UniverseGenerator(spec).generate_initial_state()
    assert all(o["mass_kg"] <= 1.0e35 for o in state["objects"])
    assert state["constraint_notes"]


# --- Generator: runtime + provenance ----------------------------------


def test_generated_runtime_has_synthetic_objects():
    spec = create_standard_physics_universe(seed=42)
    runtime = create_runtime_from_spec(spec)
    objs = runtime.registry.list_objects()
    assert len(objs) > 0
    for obj in objs:
        assert obj.truth_level == TruthLevel.SYNTHETIC_GENERATED
        assert obj.source == spec.id
        assert obj.metadata.get("synthetic") is True
        assert obj.metadata.get("synth_seed") == spec.seed


def test_generated_runtime_attaches_rules_when_specified():
    spec = create_hyperwarp_universe(seed=7)
    runtime = create_runtime_from_spec(spec)
    assert runtime.reality_rule_engine is not None
    rule_ids = {r.id for r in runtime.reality_rule_engine.list_rules()}
    # Hyperwarp preset uses warp / neural / symbolic-redshift.
    assert "warp_amplification" in rule_ids
    assert "neural_reality" in rule_ids


def test_generated_runtime_no_rules_for_standard():
    spec = create_standard_physics_universe(seed=7)
    runtime = create_runtime_from_spec(spec)
    assert runtime.reality_rule_engine is None


def test_generator_provenance_records_synthesized_label():
    spec = create_symbolic_universe(seed=1)
    runtime = create_runtime_from_spec(spec)
    label = f"synthesized:{spec.id}"
    sample = runtime.registry.list_objects()[0]
    rec = runtime.truth_tracker.get_record(sample.id)
    assert rec is not None
    assert rec.truth_level == "synthetic_generated"
    assert label in rec.transformations


def test_create_runtime_from_spec_attaches_state_and_spec():
    spec = create_standard_physics_universe(seed=3)
    runtime = create_runtime_from_spec(spec)
    assert runtime.synthesis_spec is spec
    state = runtime.synthesis_state
    assert state["spec_id"] == spec.id
    assert state["seed"] == spec.seed
    assert isinstance(state["objects"], list)


def test_unknown_rule_ids_silently_skipped():
    spec = UniverseSpec(
        id="u", seed=0,
        initial_conditions={"object_count": 1},
        rule_ids=["warp_amplification", "totally_made_up_rule"],
    )
    runtime = create_runtime_from_spec(spec)
    rule_ids = {r.id for r in runtime.reality_rule_engine.list_rules()}
    assert rule_ids == {"warp_amplification"}


# --- presets -----------------------------------------------------------


def test_all_three_presets_validate():
    for factory in (
        create_standard_physics_universe,
        create_hyperwarp_universe,
        create_symbolic_universe,
    ):
        spec = factory(seed=42)
        spec.validate()


def test_preset_ids_carry_seed():
    spec = create_standard_physics_universe(seed=99)
    assert "99" in spec.id


def test_standard_preset_has_no_rules():
    spec = create_standard_physics_universe(seed=0)
    assert spec.rule_ids == []


def test_hyperwarp_preset_has_warp_and_neural():
    spec = create_hyperwarp_universe(seed=0)
    assert "warp_amplification" in spec.rule_ids
    assert "neural_reality" in spec.rule_ids


def test_symbolic_preset_has_relaxed_causality():
    spec = create_symbolic_universe(seed=0)
    assert "causality_relaxation" in spec.rule_ids


# --- isolation: synth never touches real-data path --------------------


def test_synthesis_does_not_call_load_sample_data():
    spec = create_standard_physics_universe(seed=0)
    runtime = create_runtime_from_spec(spec)
    sources = {o.source for o in runtime.registry.list_objects()}
    # Real-data sources are absent.
    assert sources == {spec.id}
    assert not any(s in sources for s in ("gaia", "sdss", "desi", "jpl"))
