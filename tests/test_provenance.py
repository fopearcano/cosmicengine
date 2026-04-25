"""Tests for the Phase 36 provenance / audit layer."""

from __future__ import annotations

import pytest

from cosmic_engine.ai.base import AIWarpModel
from cosmic_engine.core.object_types import CosmicObjectType
from cosmic_engine.core.truth import TruthLevel
from cosmic_engine.core.universe_object import UniverseObject
from cosmic_engine.core.vector import Vector3
from cosmic_engine.observer import Observer
from cosmic_engine.observer.reality_view import RealityView
from cosmic_engine.provenance import (
    ProvenanceRecord,
    TruthTracker,
    audit_reality_view,
    detect_truth_mixing,
)
from cosmic_engine.reality import (
    RealityRuleEngine,
    create_hypertravel_reality_preset,
)
from cosmic_engine.rendering import SimpleCamera
from cosmic_engine.runtime import CosmicRuntime, RuntimeConfig
from cosmic_engine.runtime.scene_state import SceneState


def _star(obj_id: str, position: Vector3, **overrides) -> UniverseObject:
    kwargs = {
        "id": obj_id,
        "name": obj_id,
        "object_type": CosmicObjectType.STAR,
        "position_m": position,
        "velocity_m_s": Vector3.zero(),
        "truth_level": TruthLevel.OBSERVED,
        "source": "gaia_dr3",
        "spectral_class": "G",
        "metadata": {"apparent_magnitude": 6.0},
    }
    kwargs.update(overrides)
    return UniverseObject(**kwargs)


def _camera() -> SimpleCamera:
    return SimpleCamera(
        position_m=Vector3.zero(),
        forward=Vector3(0.0, 1.0, 0.0),
        up=Vector3(0.0, 0.0, 1.0),
        fov_degrees=90.0,
        image_width=32,
        image_height=32,
    )


def _basic_observer(**overrides) -> Observer:
    kwargs = {
        "id": "obs",
        "position_m": Vector3.zero(),
        "velocity_m_s": Vector3.zero(),
        "forward": Vector3(0.0, 1.0, 0.0),
        "up": Vector3(0.0, 0.0, 1.0),
        "warp_factor": 1.0,
    }
    kwargs.update(overrides)
    return Observer(**kwargs)


# --- ProvenanceRecord --------------------------------------------------


def test_record_to_dict_roundtrip():
    rec = ProvenanceRecord(
        entity_id="e",
        source="gaia",
        truth_level="observed",
        transformations=["physics_nbody"],
        timestamp=1.5,
        observer_id="o1",
    )
    out = rec.to_dict()
    assert out == {
        "entity_id": "e",
        "source": "gaia",
        "truth_level": "observed",
        "transformations": ["physics_nbody"],
        "timestamp": 1.5,
        "observer_id": "o1",
    }


def test_record_add_transformation_appends_in_order():
    rec = ProvenanceRecord(entity_id="e", source="x", truth_level="observed")
    rec.add_transformation("a")
    rec.add_transformation("b")
    rec.add_transformation("c")
    assert rec.transformations == ["a", "b", "c"]


def test_record_rejects_empty_transformation():
    rec = ProvenanceRecord(entity_id="e", source="x", truth_level="observed")
    with pytest.raises(ValueError):
        rec.add_transformation("")


# --- TruthTracker ------------------------------------------------------


def test_tracker_register_and_get():
    t = TruthTracker()
    assert len(t) == 0
    t.register_entity("e1", "gaia", "observed")
    assert len(t) == 1
    rec = t.get_record("e1")
    assert rec is not None
    assert rec.source == "gaia"
    assert "e1" in t


def test_tracker_register_idempotent():
    t = TruthTracker()
    a = t.register_entity("e", "src", "observed")
    a.add_transformation("step1")
    b = t.register_entity("e", "src", "observed")
    # Second register must not blow away the existing transformation list.
    assert b is a
    assert b.transformations == ["step1"]


def test_tracker_register_rejects_empty_id():
    t = TruthTracker()
    with pytest.raises(ValueError):
        t.register_entity("", "src", "observed")


def test_tracker_add_transformation_returns_bool():
    t = TruthTracker()
    t.register_entity("e", "src", "observed")
    assert t.add_transformation("e", "step") is True
    assert t.add_transformation("ghost", "step") is False
    rec = t.get_record("e")
    assert rec.transformations == ["step"]


def test_tracker_add_transformation_to_all():
    t = TruthTracker()
    t.register_entity("a", "x", "observed")
    t.register_entity("b", "x", "observed")
    n = t.add_transformation_to_all("global_step")
    assert n == 2
    assert t.get_record("a").transformations == ["global_step"]
    assert t.get_record("b").transformations == ["global_step"]


def test_tracker_list_records_sorted_by_id():
    t = TruthTracker()
    for oid in ("c", "a", "b"):
        t.register_entity(oid, "x", "observed")
    assert [r.entity_id for r in t.list_records()] == ["a", "b", "c"]


# --- audit / detect_truth_mixing --------------------------------------


def _make_view(
    *,
    truth_counts=None,
    transformations=None,
    active_rules=None,
    reality_metadata=None,
    visible_event_count=0,
) -> RealityView:
    scene = SceneState(
        julian_date=2_451_545.0,
        total_objects=0,
        active_objects=0,
        truth_level_counts=dict(truth_counts or {}),
    )
    return RealityView(
        observer_id="obs",
        scene_state=scene,
        representation_type="flat",
        metadata={
            "provenance_summary": {
                "transformations": list(transformations or []),
            },
        },
        active_rule_ids=list(active_rules or []),
        reality_metadata=dict(reality_metadata or {}),
        visible_event_count=visible_event_count,
    )


def test_audit_returns_expected_keys():
    view = _make_view()
    audit = audit_reality_view(view)
    for k in (
        "observer_id",
        "truth_distribution",
        "transformations",
        "active_rules",
        "warnings",
    ):
        assert k in audit


def test_audit_combines_pipeline_and_rule_transforms():
    view = _make_view(
        transformations=["perception_warp"],
        active_rules=["warp_amplification"],
    )
    audit = audit_reality_view(view)
    assert "perception_warp" in audit["transformations"]
    assert "reality_rule:warp_amplification" in audit["transformations"]
    # Order is "pipeline then rules", and no duplicates appear.
    assert audit["transformations"].count("perception_warp") == 1


def test_detect_warns_on_ai_step_with_observed_data_no_neural_marker():
    view = _make_view(
        truth_counts={"observed": 5},
        transformations=["perception_warp", "ai_warp"],
    )
    warnings = detect_truth_mixing(view)
    assert any("AI transformation" in w for w in warnings)


def test_detect_no_warning_when_neural_use_declared():
    view = _make_view(
        truth_counts={"observed": 5},
        transformations=["perception_warp", "ai_warp"],
        reality_metadata={"use_neural_perception": True},
    )
    warnings = detect_truth_mixing(view)
    assert all("AI transformation" not in w for w in warnings)


def test_detect_warns_on_symbolic_rule_without_color_mapping():
    # Forge a view whose reality_metadata is empty even though a
    # symbolic rule is active. (In normal flow the rule would set
    # color_mapping; this is the audit catching the mismatch.)
    view = _make_view(active_rules=["redshift_symbolic_color"])
    warnings = detect_truth_mixing(view)
    assert any("symbolic rule" in w for w in warnings)


def test_detect_warns_on_relaxed_causality_with_visible_events():
    view = _make_view(
        reality_metadata={"causality_mode": "relaxed"},
        visible_event_count=3,
    )
    warnings = detect_truth_mixing(view)
    assert any("relaxed" in w for w in warnings)


def test_detect_no_warning_for_clean_scientific_view():
    view = _make_view(truth_counts={"observed": 5})
    assert detect_truth_mixing(view) == []


# --- Runtime integration ----------------------------------------------


def test_runtime_has_truth_tracker_by_default():
    rt = CosmicRuntime()
    assert isinstance(rt.truth_tracker, TruthTracker)
    assert len(rt.truth_tracker) == 0


def test_runtime_registers_added_objects():
    rt = CosmicRuntime()
    rt.add_objects([_star("s1", Vector3(0.0, 1.0e16, 0.0))])
    rec = rt.truth_tracker.get_record("s1")
    assert rec is not None
    assert rec.source == "gaia_dr3"
    assert rec.truth_level == "observed"


def test_runtime_render_attaches_provenance_summary(tmp_path):
    rt = CosmicRuntime(
        config=RuntimeConfig(
            enable_physics=False,
            max_active_objects=10,
            active_radius_m=1.0e30,
        )
    )
    rt.add_objects([_star("s1", Vector3(0.0, 1.0e16, 0.0))])
    obs = _basic_observer(config={"output_ppm_path": tmp_path / "a.ppm"})
    rt.observer_manager.add_observer(obs)
    view = rt.render_for_observer("obs", _camera())
    summary = view.provenance_summary
    assert summary["transformations"] == ["perception_warp"]
    assert summary["truth_level_counts"] == {"observed": 1}
    assert summary["source_counts"] == {"gaia_dr3": 1}
    # Per-entity record now shows the transformation we just applied.
    rec = rt.truth_tracker.get_record("s1")
    assert "perception_warp" in rec.transformations


class _IdentityAIWarp(AIWarpModel):
    def predict_direction(self, direction, observer):  # noqa: ARG002
        return direction

    def predict_brightness(self, brightness, direction, observer):  # noqa: ARG002
        return brightness

    def predict_color(self, color_rgb, direction, observer):  # noqa: ARG002
        return color_rgb

    def confidence(self) -> float:
        return 1.0


def test_runtime_render_with_ai_warp_records_ai_step(tmp_path):
    rt = CosmicRuntime(
        config=RuntimeConfig(
            enable_physics=False,
            max_active_objects=10,
            active_radius_m=1.0e30,
        )
    )
    rt.add_objects([_star("s1", Vector3(0.0, 1.0e16, 0.0))])
    obs = _basic_observer(
        ai_warp_model=_IdentityAIWarp(),
        config={"output_ppm_path": tmp_path / "a.ppm"},
    )
    rt.observer_manager.add_observer(obs)
    view = rt.render_for_observer("obs", _camera())
    assert "ai_warp" in view.provenance_summary["transformations"]
    # Audit should warn since observed data was passed through AI
    # without a neural-perception marker.
    assert any("AI transformation" in w for w in view.audit_warnings)


def test_runtime_render_with_rules_records_rule_transformations(tmp_path):
    rt = CosmicRuntime(
        config=RuntimeConfig(
            enable_physics=False,
            max_active_objects=10,
            active_radius_m=1.0e30,
        )
    )
    rt.add_objects([_star("s1", Vector3(0.0, 1.0e16, 0.0))])
    obs = _basic_observer(config={"output_ppm_path": tmp_path / "a.ppm"})
    rt.observer_manager.add_observer(obs)
    rt.reality_rule_engine = RealityRuleEngine(create_hypertravel_reality_preset())
    view = rt.render_for_observer("obs", _camera())
    # Reality rule transformations land in the provenance summary as
    # ``reality_rule:<id>`` labels.
    summary_tx = view.provenance_summary["transformations"]
    assert any(t.startswith("reality_rule:") for t in summary_tx)


def test_runtime_render_does_not_crash_with_empty_registry(tmp_path):
    rt = CosmicRuntime(
        config=RuntimeConfig(
            enable_physics=False,
            max_active_objects=10,
            active_radius_m=1.0e30,
        )
    )
    obs = _basic_observer(config={"output_ppm_path": tmp_path / "a.ppm"})
    rt.observer_manager.add_observer(obs)
    view = rt.render_for_observer("obs", _camera())
    summary = view.provenance_summary
    assert summary["tracked_entities"] == 0
    assert summary["total_records"] == 0


def test_view_to_dict_surfaces_audit_fields(tmp_path):
    rt = CosmicRuntime(
        config=RuntimeConfig(
            enable_physics=False,
            max_active_objects=10,
            active_radius_m=1.0e30,
        )
    )
    rt.add_objects([_star("s1", Vector3(0.0, 1.0e16, 0.0))])
    obs = _basic_observer(config={"output_ppm_path": tmp_path / "a.ppm"})
    rt.observer_manager.add_observer(obs)
    view = rt.render_for_observer("obs", _camera())
    out = view.to_dict()
    assert "provenance_summary" in out
    assert "audit_warnings" in out
