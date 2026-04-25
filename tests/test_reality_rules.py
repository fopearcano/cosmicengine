"""Tests for the Phase 35 emergent reality / rule layer."""

from __future__ import annotations

import pytest

from cosmic_engine.core.units import LIGHTYEAR_IN_METERS
from cosmic_engine.core.vector import Vector3
from cosmic_engine.observer import Observer
from cosmic_engine.reality import (
    CausalityRelaxationRule,
    NeuralRealityRule,
    RealityRule,
    RealityRuleDomain,
    RealityRuleEngine,
    RealityRuleKind,
    RedshiftSymbolicColorRule,
    RuleContext,
    WarpAmplificationRule,
    create_blackhole_perception_preset,
    create_hypertravel_reality_preset,
    create_scientific_reality_preset,
)
from cosmic_engine.rendering import SimpleCamera
from cosmic_engine.runtime import CosmicRuntime, RuntimeConfig


def _ctx(**overrides) -> RuleContext:
    base = {
        "observer_id": "test",
        "observer_position_m": Vector3.zero(),
        "observer_velocity_m_s": Vector3.zero(),
        "warp_factor": 1.0,
        "coordinate_time_t": 0.0,
        "proper_time_tau": 0.0,
    }
    base.update(overrides)
    return RuleContext(**base)


def _camera() -> SimpleCamera:
    return SimpleCamera(
        position_m=Vector3.zero(),
        forward=Vector3(0.0, 1.0, 0.0),
        up=Vector3(0.0, 0.0, 1.0),
        fov_degrees=90.0,
        image_width=32,
        image_height=32,
    )


# --- RuleContext -------------------------------------------------------


def test_context_clone_is_independent():
    a = _ctx(metadata={"k": "v"}, active_rule_ids=["x"])
    b = a.clone()
    b.metadata["k"] = "changed"
    b.active_rule_ids.append("y")
    b.warp_factor = 99.0
    # Mutating the clone leaves the original untouched.
    assert a.metadata == {"k": "v"}
    assert a.active_rule_ids == ["x"]
    assert a.warp_factor == 1.0


def test_context_to_dict_round_trip():
    ctx = _ctx(
        metadata={"a": 1},
        active_rule_ids=["r1"],
        scale_zone="interstellar",
    )
    d = ctx.to_dict()
    assert d["observer_id"] == "test"
    assert d["scale_zone"] == "interstellar"
    assert d["metadata"] == {"a": 1}
    assert d["active_rule_ids"] == ["r1"]


# --- RealityRule basics ------------------------------------------------


def test_default_apply_clones_and_echoes_parameters():
    rule = RealityRule(
        id="r1",
        name="test",
        domain=RealityRuleDomain.PHYSICS,
        kind=RealityRuleKind.PHYSICAL,
        parameters={"k": 7},
    )
    out = rule.apply(_ctx())
    assert out.metadata["rules"]["r1"] == {"k": 7}


def test_disabled_rule_skipped_by_engine():
    rule = WarpAmplificationRule(parameters={"factor": 4.0}, enabled=False)
    engine = RealityRuleEngine([rule])
    out = engine.evaluate(_ctx())
    assert out.warp_factor == 1.0
    assert out.active_rule_ids == []


def test_warp_amplification_rejects_nonpositive_factor():
    with pytest.raises(ValueError):
        WarpAmplificationRule(parameters={"factor": 0.0})
    with pytest.raises(ValueError):
        WarpAmplificationRule(parameters={"factor": -1.0})


def test_warp_amplification_floors_at_one():
    # Even with factor < 1, the engine never lets warp_factor drop
    # below 1.0 (the documented physical floor).
    rule = WarpAmplificationRule(parameters={"factor": 0.25})
    out = rule.apply(_ctx(warp_factor=2.0))
    assert out.warp_factor == 1.0


def test_warp_amplification_does_not_mutate_input():
    ctx = _ctx(warp_factor=1.5)
    rule = WarpAmplificationRule(parameters={"factor": 4.0})
    out = rule.apply(ctx)
    assert ctx.warp_factor == 1.5
    assert out.warp_factor == 6.0


def test_redshift_rule_sets_color_mapping():
    out = RedshiftSymbolicColorRule().apply(_ctx())
    assert out.metadata["color_mapping"] == "symbolic_redshift"


def test_causality_rule_sets_mode():
    out = CausalityRelaxationRule().apply(_ctx())
    assert out.metadata["causality_mode"] == "relaxed"


def test_neural_rule_sets_flag_and_optional_model_id():
    out_no_model = NeuralRealityRule().apply(_ctx())
    assert out_no_model.metadata["use_neural_perception"] is True
    assert "model_id" not in out_no_model.metadata
    out_with_model = NeuralRealityRule(parameters={"model_id": "m1"}).apply(_ctx())
    assert out_with_model.metadata["model_id"] == "m1"


# --- RealityRuleEngine -------------------------------------------------


def test_engine_rejects_duplicate_ids():
    engine = RealityRuleEngine()
    engine.add_rule(WarpAmplificationRule(id="x", parameters={"factor": 2.0}))
    with pytest.raises(ValueError):
        engine.add_rule(WarpAmplificationRule(id="x", parameters={"factor": 3.0}))


def test_engine_remove_unknown_raises():
    engine = RealityRuleEngine()
    with pytest.raises(KeyError):
        engine.remove_rule("missing")


def test_engine_priority_ordering_is_higher_first():
    """Higher priority applies first; a later (lower-priority) factor
    multiplies on top of the already-amplified warp_factor."""
    engine = RealityRuleEngine([
        WarpAmplificationRule(id="b", priority=1, parameters={"factor": 2.0}),
        WarpAmplificationRule(id="a", priority=10, parameters={"factor": 3.0}),
    ])
    out = engine.evaluate(_ctx(warp_factor=1.0))
    # First "a" applies (1*3=3), then "b" (3*2=6).
    assert out.warp_factor == 6.0
    assert out.active_rule_ids == ["a", "b"]


def test_engine_list_rules_enabled_only():
    a = WarpAmplificationRule(id="a", parameters={"factor": 2.0}, enabled=True)
    b = WarpAmplificationRule(id="b", parameters={"factor": 2.0}, enabled=False)
    engine = RealityRuleEngine([a, b])
    assert {r.id for r in engine.list_rules(enabled_only=False)} == {"a", "b"}
    assert {r.id for r in engine.list_rules(enabled_only=True)} == {"a"}


def test_engine_evaluate_does_not_mutate_input_context():
    ctx = _ctx(warp_factor=2.0)
    engine = RealityRuleEngine([WarpAmplificationRule(parameters={"factor": 3.0})])
    out = engine.evaluate(ctx)
    assert ctx.warp_factor == 2.0
    assert out.warp_factor == 6.0


def test_engine_appends_rule_id_only_once():
    engine = RealityRuleEngine([WarpAmplificationRule(parameters={"factor": 2.0})])
    out = engine.evaluate(_ctx())
    assert out.active_rule_ids == ["warp_amplification"]


# --- presets -----------------------------------------------------------


def test_scientific_preset_is_empty():
    rules = create_scientific_reality_preset()
    assert rules == []


def test_hypertravel_preset_has_expected_rules():
    rules = create_hypertravel_reality_preset()
    ids = {r.id for r in rules}
    assert "warp_amplification" in ids
    assert "redshift_symbolic_color" in ids
    assert "neural_reality" in ids
    assert "causality_relaxation" in ids
    # Every rule is enabled and carries non-empty name + description.
    for r in rules:
        assert r.enabled
        assert r.name
        assert r.description


def test_blackhole_preset_has_expected_rules():
    rules = create_blackhole_perception_preset()
    ids = {r.id for r in rules}
    assert ids == {
        "neural_blackhole_perception",
        "lensing_emphasis",
        "spectral_shift_blackhole",
    }
    # Lensing emphasis is a base RealityRule with rendering domain.
    lensing = next(r for r in rules if r.id == "lensing_emphasis")
    assert lensing.domain == RealityRuleDomain.RENDERING
    assert lensing.parameters["emphasis"] == 2.0


def test_hypertravel_preset_evaluates_to_expected_metadata():
    engine = RealityRuleEngine(create_hypertravel_reality_preset())
    out = engine.evaluate(_ctx(warp_factor=1.0))
    assert out.warp_factor == 4.0
    assert out.metadata["color_mapping"] == "symbolic_redshift"
    assert out.metadata["use_neural_perception"] is True
    assert out.metadata["causality_mode"] == "relaxed"


# --- Runtime integration -----------------------------------------------


def _basic_observer(**overrides) -> Observer:
    kwargs = {
        "id": "obs",
        "position_m": Vector3(0.0, -1.0 * LIGHTYEAR_IN_METERS, 0.0),
        "velocity_m_s": Vector3.zero(),
        "forward": Vector3(0.0, 1.0, 0.0),
        "up": Vector3(0.0, 0.0, 1.0),
        "warp_factor": 1.0,
    }
    kwargs.update(overrides)
    return Observer(**kwargs)


def _make_runtime() -> CosmicRuntime:
    return CosmicRuntime(
        config=RuntimeConfig(
            enable_physics=False,
            enable_perception=False,
            active_radius_m=1.0e30,
            max_active_objects=100,
        )
    )


def test_runtime_render_without_engine_has_no_active_rules(tmp_path):
    runtime = _make_runtime()
    obs = _basic_observer(config={"output_ppm_path": tmp_path / "a.ppm"})
    runtime.observer_manager.add_observer(obs)
    view = runtime.render_for_observer("obs", _camera())
    assert view.active_rule_ids == []
    assert view.reality_metadata == {}


def test_runtime_render_with_hypertravel_preset_attaches_rules(tmp_path):
    runtime = _make_runtime()
    obs = _basic_observer(config={"output_ppm_path": tmp_path / "a.ppm"})
    runtime.observer_manager.add_observer(obs)
    runtime.reality_rule_engine = RealityRuleEngine(
        create_hypertravel_reality_preset()
    )
    view = runtime.render_for_observer("obs", _camera())
    assert "warp_amplification" in view.active_rule_ids
    assert "neural_reality" in view.active_rule_ids
    # Effective warp captured in metadata, but the observer's stored
    # warp is untouched.
    assert view.metadata["effective_warp_factor"] == pytest.approx(4.0)
    assert obs.warp_factor == 1.0  # unchanged
    assert view.reality_metadata["color_mapping"] == "symbolic_redshift"


def test_runtime_render_does_not_mutate_observer_warp(tmp_path):
    runtime = _make_runtime()
    obs = _basic_observer(
        warp_factor=2.0,
        config={"output_ppm_path": tmp_path / "a.ppm"},
    )
    runtime.observer_manager.add_observer(obs)
    runtime.reality_rule_engine = RealityRuleEngine([
        WarpAmplificationRule(parameters={"factor": 5.0})
    ])
    runtime.render_for_observer("obs", _camera())
    runtime.render_for_observer("obs", _camera())
    # Two renders: each should see the same starting warp (no compounding).
    assert obs.warp_factor == 2.0


def test_runtime_render_includes_active_rules_in_to_dict(tmp_path):
    runtime = _make_runtime()
    obs = _basic_observer(config={"output_ppm_path": tmp_path / "a.ppm"})
    runtime.observer_manager.add_observer(obs)
    runtime.reality_rule_engine = RealityRuleEngine(
        create_blackhole_perception_preset()
    )
    view = runtime.render_for_observer("obs", _camera())
    out = view.to_dict()
    assert isinstance(out["active_rule_ids"], list)
    assert "neural_blackhole_perception" in out["active_rule_ids"]
    assert isinstance(out["reality_metadata"], dict)


def test_runtime_render_preserves_universe_state(tmp_path):
    runtime = _make_runtime()
    runtime.load_sample_data()
    n_before = len(runtime.registry.list_objects())
    truth_before = sum(
        1 for o in runtime.registry.list_objects() if o.truth_level
    )
    obs = _basic_observer(config={"output_ppm_path": tmp_path / "a.ppm"})
    runtime.observer_manager.add_observer(obs)
    runtime.reality_rule_engine = RealityRuleEngine(
        create_hypertravel_reality_preset()
    )
    runtime.render_for_observer("obs", _camera())
    n_after = len(runtime.registry.list_objects())
    truth_after = sum(
        1 for o in runtime.registry.list_objects() if o.truth_level
    )
    assert n_before == n_after
    assert truth_before == truth_after
