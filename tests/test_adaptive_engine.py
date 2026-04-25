"""Tests for the Phase 37 self-improving / adaptive feedback layer."""

from __future__ import annotations

import numpy as np
import pytest

from cosmic_engine.adaptive import (
    AdaptiveEngine,
    AdaptivePolicy,
    FeedbackRecord,
    compute_acceleration_error,
    compute_brightness_error,
    compute_direction_error,
)
from cosmic_engine.ai.base import AIWarpModel
from cosmic_engine.ai.spacetime_field import SpacetimeFieldModel
from cosmic_engine.core.object_types import CosmicObjectType
from cosmic_engine.core.truth import TruthLevel
from cosmic_engine.core.universe_object import UniverseObject
from cosmic_engine.core.vector import Vector3
from cosmic_engine.observer import Observer
from cosmic_engine.rendering import SimpleCamera
from cosmic_engine.runtime import CosmicRuntime, RuntimeConfig


def _star(obj_id: str, position: Vector3) -> UniverseObject:
    return UniverseObject(
        id=obj_id,
        name=obj_id,
        object_type=CosmicObjectType.STAR,
        position_m=position,
        velocity_m_s=Vector3.zero(),
        truth_level=TruthLevel.OBSERVED,
        source="gaia_dr3",
        spectral_class="G",
        metadata={"apparent_magnitude": 6.0},
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


def _camera() -> SimpleCamera:
    return SimpleCamera(
        position_m=Vector3.zero(),
        forward=Vector3(0.0, 1.0, 0.0),
        up=Vector3(0.0, 0.0, 1.0),
        fov_degrees=90.0,
        image_width=32,
        image_height=32,
    )


# --- metrics -----------------------------------------------------------


def test_acceleration_error_zero_when_identical():
    a = np.array([1.0, 2.0, 3.0])
    assert compute_acceleration_error(a, a) == pytest.approx(0.0)


def test_acceleration_error_relative_l2():
    a = np.array([1.0, 0.0, 0.0])
    p = np.array([1.5, 0.0, 0.0])
    assert compute_acceleration_error(a, p) == pytest.approx(0.5)


def test_acceleration_error_shape_mismatch_raises():
    with pytest.raises(ValueError):
        compute_acceleration_error(np.array([1.0]), np.array([1.0, 2.0]))


def test_direction_error_zero_for_identical():
    a = np.array([1.0, 0.0, 0.0])
    assert compute_direction_error(a, a) == pytest.approx(0.0)


def test_direction_error_one_for_orthogonal():
    a = np.array([1.0, 0.0, 0.0])
    p = np.array([0.0, 1.0, 0.0])
    assert compute_direction_error(a, p) == pytest.approx(1.0)


def test_direction_error_two_for_antiparallel():
    a = np.array([1.0, 0.0, 0.0])
    p = np.array([-1.0, 0.0, 0.0])
    assert compute_direction_error(a, p) == pytest.approx(2.0)


def test_direction_error_handles_degenerate_input():
    a = np.array([0.0, 0.0, 0.0])
    p = np.array([1.0, 0.0, 0.0])
    # Returns max error rather than NaN.
    assert compute_direction_error(a, p) == 2.0


def test_brightness_error_zero_when_identical():
    assert compute_brightness_error(2.0, 2.0) == pytest.approx(0.0)


def test_brightness_error_relative():
    assert compute_brightness_error(2.0, 2.5) == pytest.approx(0.25)


def test_brightness_error_handles_zero_truth():
    # Falls back to eps in the denominator; finite, very large value.
    err = compute_brightness_error(0.0, 1.0)
    assert err > 1.0e10
    assert np.isfinite(err)


# --- FeedbackRecord ----------------------------------------------------


def _record(**overrides) -> FeedbackRecord:
    kwargs = {
        "id": "f1",
        "observer_id": "obs",
        "timestamp_t": 0.0,
        "context": {},
        "metric_name": "deviation",
        "metric_value": 1.0,
        "expected_value": 1.0,
        "deviation": 0.5,
        "source": "spacetime_model",
    }
    kwargs.update(overrides)
    return FeedbackRecord(**kwargs)


def test_feedback_record_to_dict_roundtrip():
    rec = _record(notes="test")
    out = rec.to_dict()
    assert out["id"] == "f1"
    assert out["deviation"] == 0.5
    assert out["notes"] == "test"
    assert out["source"] == "spacetime_model"


def test_feedback_record_rejects_negative_deviation():
    with pytest.raises(ValueError):
        _record(deviation=-0.1)


def test_feedback_record_rejects_empty_id():
    with pytest.raises(ValueError):
        _record(id="")


def test_feedback_record_rejects_empty_metric_name():
    with pytest.raises(ValueError):
        _record(metric_name="")


def test_feedback_record_rejects_empty_source():
    with pytest.raises(ValueError):
        _record(source="")


# --- AdaptivePolicy ----------------------------------------------------


def test_policy_should_record_above_threshold():
    p = AdaptivePolicy(error_threshold=0.1)
    assert p.should_record(_record(deviation=0.2)) is True
    assert p.should_record(_record(deviation=0.05)) is False


def test_policy_rejects_invalid_thresholds():
    with pytest.raises(ValueError):
        AdaptivePolicy(error_threshold=-0.1)
    with pytest.raises(ValueError):
        AdaptivePolicy(window_size=0)
    with pytest.raises(ValueError):
        AdaptivePolicy(sustained_fraction=1.5)
    with pytest.raises(ValueError):
        AdaptivePolicy(max_updates_per_run=0)


def test_policy_should_update_requires_full_window():
    p = AdaptivePolicy(error_threshold=0.1, window_size=5)
    # Only 3 records, but all above threshold -> still NOT enough.
    records = [_record(id=f"f{i}", deviation=0.5) for i in range(3)]
    assert p.should_update_model(records) is False


def test_policy_should_update_when_sustained():
    p = AdaptivePolicy(
        error_threshold=0.1, window_size=5, sustained_fraction=0.5
    )
    # 5 records, 3 above threshold -> 60 % >= 50 % -> True.
    records = [
        _record(id=f"f{i}", deviation=0.5 if i < 3 else 0.0)
        for i in range(5)
    ]
    assert p.should_update_model(records) is True


def test_policy_filters_by_source():
    p = AdaptivePolicy(
        error_threshold=0.1, window_size=2, sustained_fraction=0.5
    )
    records = [
        _record(id="f1", deviation=0.5, source="ai_warp"),
        _record(id="f2", deviation=0.5, source="ai_warp"),
        _record(id="f3", deviation=0.5, source="spacetime_model"),
    ]
    # Only one spacetime record; window not full.
    assert p.should_update_model(records, source="spacetime_model") is False
    # Two ai_warp records, both bad; suggestion possible.
    assert p.should_update_rules(records, source="ai_warp") is True


# --- AdaptiveEngine ----------------------------------------------------


def test_engine_does_not_record_below_threshold():
    eng = AdaptiveEngine(policy=AdaptivePolicy(error_threshold=0.1))
    kept = eng.record_feedback(_record(deviation=0.05))
    assert kept is False
    assert len(eng.feedback_log) == 0


def test_engine_evaluate_emits_no_suggestion_under_window():
    eng = AdaptiveEngine(policy=AdaptivePolicy(
        error_threshold=0.1, window_size=4
    ))
    for i in range(2):
        eng.record_feedback(_record(id=f"f{i}", deviation=0.5))
    assert eng.evaluate() == []


def test_engine_evaluate_emits_retrain_suggestion_when_sustained():
    eng = AdaptiveEngine(policy=AdaptivePolicy(
        error_threshold=0.1, window_size=4, sustained_fraction=0.5
    ))
    for i in range(4):
        eng.record_feedback(_record(id=f"f{i}", deviation=0.5))
    suggestions = eng.evaluate()
    assert len(suggestions) == 1
    s = suggestions[0]
    assert s["action"] == "retrain_spacetime_model"
    assert s["window_size"] == 4
    assert len(s["feedback_ids"]) == 4
    # Mean deviation reported.
    assert s["mean_deviation"] == pytest.approx(0.5)


def test_engine_respects_max_updates_per_run():
    eng = AdaptiveEngine(policy=AdaptivePolicy(
        error_threshold=0.1, window_size=2, sustained_fraction=0.5,
        max_updates_per_run=1,
    ))
    for i in range(2):
        eng.record_feedback(
            _record(id=f"st{i}", deviation=0.5, source="spacetime_model")
        )
    for i in range(2):
        eng.record_feedback(
            _record(id=f"ai{i}", deviation=0.5, source="ai_warp")
        )
    suggestions = eng.evaluate()
    # Only one suggestion (model) — rule suggestion is gated by the run cap.
    assert len(suggestions) == 1
    assert suggestions[0]["action"] == "retrain_spacetime_model"
    # Second call within the same run still bumps the counter; no more.
    assert eng.evaluate() == []


def test_engine_reset_run_re_enables_suggestions():
    eng = AdaptiveEngine(policy=AdaptivePolicy(
        error_threshold=0.1, window_size=2, sustained_fraction=0.5,
        max_updates_per_run=1,
    ))
    for i in range(2):
        eng.record_feedback(_record(id=f"st{i}", deviation=0.5))
    eng.evaluate()
    eng.reset_run()
    # Same log, but fresh run -> a new suggestion can fire again.
    suggestions = eng.evaluate()
    assert len(suggestions) == 1


def test_engine_summary_reports_log_state():
    eng = AdaptiveEngine(policy=AdaptivePolicy(error_threshold=0.05))
    eng.record_feedback(_record(deviation=0.3, source="ai_warp"))
    eng.record_feedback(_record(id="f2", deviation=0.7, source="spacetime_model"))
    s = eng.summary()
    assert s["record_count"] == 2
    assert s["max_deviation"] == pytest.approx(0.7)
    assert s["by_source"] == {"ai_warp": 1, "spacetime_model": 1}


# --- Runtime integration ----------------------------------------------


class _BiasedSpacetime(SpacetimeFieldModel):
    """Returns 50 % stronger acceleration than analytical."""

    def query_acceleration(self, position, direction=None):  # noqa: ARG002
        import numpy as _np

        position = _np.asarray(position, dtype=_np.float64)
        r = float(_np.linalg.norm(position))
        if r == 0.0:
            return _np.zeros(3, dtype=_np.float64)
        # Analytical Schwarzschild magnitude * 1.5 to deliberately
        # introduce sustained drift.
        analytical = (
            -2.0 * 6.67430e-11 * 1.989e30 / (r ** 3)
        ) * position
        return analytical * 1.5

    def confidence(self) -> float:
        return 0.8


class _BiasedAIWarp(AIWarpModel):
    """Returns a direction rotated 90° from the input — large drift."""

    def predict_direction(self, direction, observer):  # noqa: ARG002
        return Vector3(direction.y, -direction.x, direction.z)

    def predict_brightness(self, brightness, direction, observer):  # noqa: ARG002
        return brightness

    def predict_color(self, color_rgb, direction, observer):  # noqa: ARG002
        return color_rgb

    def confidence(self) -> float:
        return 0.5


def test_runtime_without_adaptive_engine_returns_empty(tmp_path):
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
    assert view.feedback_summary == {}
    assert view.adaptive_suggestions == []


def test_runtime_collects_feedback_and_suggests(tmp_path):
    rt = CosmicRuntime(
        config=RuntimeConfig(
            enable_physics=False,
            max_active_objects=10,
            active_radius_m=1.0e30,
        )
    )
    rt.add_objects([_star("s1", Vector3(0.0, 1.0e16, 0.0))])
    obs = _basic_observer(
        spacetime_model=_BiasedSpacetime(),
        config={
            "output_ppm_path": tmp_path / "a.ppm",
            "adaptive_reference_mass_kg": 1.989e30,
            "adaptive_probe_radius_m": 1.0e9,
        },
    )
    rt.observer_manager.add_observer(obs)
    rt.adaptive_engine = AdaptiveEngine(policy=AdaptivePolicy(
        error_threshold=0.1, window_size=4, sustained_fraction=0.5,
        max_updates_per_run=4,
    ))
    last_view = None
    for _ in range(6):
        last_view = rt.render_for_observer("obs", _camera())
    summary = last_view.feedback_summary
    assert summary["record_count"] >= 4
    assert "spacetime_model" in summary["by_source"]
    assert summary["max_deviation"] > 0.1
    suggestions = last_view.adaptive_suggestions
    assert any(s["action"] == "retrain_spacetime_model" for s in suggestions)


def test_runtime_does_not_mutate_models_or_rules(tmp_path):
    """No automatic mutation: the observer's spacetime_model identity
    and the runtime's rule engine are both unchanged after suggestions
    fire."""
    rt = CosmicRuntime(
        config=RuntimeConfig(
            enable_physics=False,
            max_active_objects=10,
            active_radius_m=1.0e30,
        )
    )
    rt.add_objects([_star("s1", Vector3(0.0, 1.0e16, 0.0))])
    spacetime = _BiasedSpacetime()
    obs = _basic_observer(
        spacetime_model=spacetime,
        config={"output_ppm_path": tmp_path / "a.ppm"},
    )
    rt.observer_manager.add_observer(obs)
    rt.adaptive_engine = AdaptiveEngine(policy=AdaptivePolicy(
        error_threshold=0.1, window_size=2, sustained_fraction=0.5,
        max_updates_per_run=4,
    ))
    for _ in range(4):
        rt.render_for_observer("obs", _camera())
    # Same instance, same warp factor, same rule engine reference.
    assert obs.spacetime_model is spacetime
    assert rt.reality_rule_engine is None
    assert obs.warp_factor == 1.0


def test_runtime_ai_warp_drift_records_ai_feedback(tmp_path):
    rt = CosmicRuntime(
        config=RuntimeConfig(
            enable_physics=False,
            max_active_objects=10,
            active_radius_m=1.0e30,
        )
    )
    rt.add_objects([_star("s1", Vector3(0.0, 1.0e16, 0.0))])
    obs = _basic_observer(
        ai_warp_model=_BiasedAIWarp(),
        velocity_m_s=Vector3(0.5 * 299_792_458.0, 0.0, 0.0),
        config={"output_ppm_path": tmp_path / "a.ppm"},
    )
    rt.observer_manager.add_observer(obs)
    rt.adaptive_engine = AdaptiveEngine(policy=AdaptivePolicy(
        error_threshold=0.1, window_size=2, sustained_fraction=0.5,
        max_updates_per_run=4,
    ))
    last_view = None
    for _ in range(3):
        last_view = rt.render_for_observer("obs", _camera())
    assert last_view.feedback_summary["by_source"].get("ai_warp", 0) >= 2


def test_view_to_dict_surfaces_feedback_fields(tmp_path):
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
    rt.adaptive_engine = AdaptiveEngine()
    view = rt.render_for_observer("obs", _camera())
    out = view.to_dict()
    assert "feedback_summary" in out
    assert "adaptive_suggestions" in out
