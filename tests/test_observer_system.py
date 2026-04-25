"""Tests for the Phase 33 observer-dependent reality system."""

from __future__ import annotations

import numpy as np
import pytest

from cosmic_engine.ai.spacetime_field import SpacetimeFieldModel
from cosmic_engine.core.object_types import CosmicObjectType
from cosmic_engine.core.truth import TruthLevel
from cosmic_engine.core.universe_object import UniverseObject
from cosmic_engine.core.units import SPEED_OF_LIGHT_M_S
from cosmic_engine.core.vector import Vector3
from cosmic_engine.observer import Observer, ObserverManager, RealityView
from cosmic_engine.rendering import SimpleCamera
from cosmic_engine.runtime import CosmicRuntime, RuntimeConfig
from cosmic_engine.runtime.scene_state import SceneState


def _star(obj_id: str, position: Vector3) -> UniverseObject:
    return UniverseObject(
        id=obj_id,
        name=obj_id,
        object_type=CosmicObjectType.STAR,
        position_m=position,
        velocity_m_s=Vector3.zero(),
        truth_level=TruthLevel.OBSERVED,
        spectral_class="G",
        metadata={"apparent_magnitude": 6.0},
    )


def _basic_observer(obj_id: str = "obs1", **overrides) -> Observer:
    kwargs = {
        "id": obj_id,
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
        fov_degrees=120.0,
        image_width=64,
        image_height=64,
    )


# --- Observer ----------------------------------------------------------


def test_observer_validate_accepts_basic():
    obs = _basic_observer()
    obs.validate()


def test_observer_rejects_empty_id():
    with pytest.raises(ValueError):
        _basic_observer(obj_id="").validate()


def test_observer_rejects_warp_below_one():
    with pytest.raises(ValueError):
        _basic_observer(warp_factor=0.5).validate()


def test_observer_rejects_superluminal_velocity():
    with pytest.raises(ValueError):
        _basic_observer(
            velocity_m_s=Vector3(SPEED_OF_LIGHT_M_S, 0.0, 0.0)
        ).validate()


def test_observer_rejects_zero_forward():
    with pytest.raises(ValueError):
        _basic_observer(forward=Vector3.zero()).validate()


def test_observer_rejects_zero_up():
    with pytest.raises(ValueError):
        _basic_observer(up=Vector3.zero()).validate()


def test_observer_beta():
    obs = _basic_observer(
        velocity_m_s=Vector3(0.6 * SPEED_OF_LIGHT_M_S, 0.0, 0.0)
    )
    assert obs.beta() == pytest.approx(0.6)
    obs.validate()  # subluminal: ok


# --- ObserverManager ---------------------------------------------------


def test_manager_add_get_remove():
    mgr = ObserverManager()
    assert len(mgr) == 0
    obs = _basic_observer("a")
    mgr.add_observer(obs)
    assert len(mgr) == 1
    assert "a" in mgr
    assert mgr.get_observer("a") is obs
    mgr.remove_observer("a")
    assert "a" not in mgr


def test_manager_rejects_duplicate_id():
    mgr = ObserverManager()
    mgr.add_observer(_basic_observer("a"))
    with pytest.raises(ValueError):
        mgr.add_observer(_basic_observer("a"))


def test_manager_rejects_invalid_observer_on_add():
    mgr = ObserverManager()
    bad = _basic_observer(warp_factor=0.0)
    with pytest.raises(ValueError):
        mgr.add_observer(bad)


def test_manager_get_unknown_raises():
    mgr = ObserverManager()
    with pytest.raises(KeyError):
        mgr.get_observer("missing")


def test_manager_remove_unknown_raises():
    mgr = ObserverManager()
    with pytest.raises(KeyError):
        mgr.remove_observer("missing")


def test_manager_list_is_sorted_by_id():
    mgr = ObserverManager()
    for oid in ("c", "a", "b"):
        mgr.add_observer(_basic_observer(oid))
    assert [o.id for o in mgr.list_observers()] == ["a", "b", "c"]


# --- RealityView -------------------------------------------------------


def test_reality_view_to_dict_handles_no_frame():
    state = SceneState(julian_date=2_451_545.0, total_objects=0, active_objects=0)
    view = RealityView(
        observer_id="obs",
        scene_state=state,
        representation_type="flat",
        metadata={"k": "v"},
    )
    out = view.to_dict()
    assert out["observer_id"] == "obs"
    assert out["frame"] is None
    assert out["metadata"] == {"k": "v"}
    assert out["representation_type"] == "flat"


def test_reality_view_to_dict_reports_frame_shape():
    state = SceneState(julian_date=2_451_545.0, total_objects=0, active_objects=0)
    pixels = np.zeros((4, 4, 3), dtype=np.uint8)
    view = RealityView(
        observer_id="obs",
        scene_state=state,
        representation_type="flat",
        frame_data=pixels,
    )
    out = view.to_dict()
    assert out["frame"]["shape"] == [4, 4, 3]
    assert out["frame"]["dtype"] == "uint8"


def test_reality_view_summary_is_string():
    state = SceneState(julian_date=2_451_545.0, total_objects=0, active_objects=0)
    view = RealityView(
        observer_id="obs",
        scene_state=state,
        representation_type="flat",
        metadata={"warp_factor": 2.0, "spacetime_model": "MockField",
                  "object_count": 3},
    )
    s = view.summary()
    assert "obs" in s
    assert "warp=2.0" in s
    assert "MockField" in s


# --- Runtime integration ----------------------------------------------


def test_runtime_has_observer_manager_by_default():
    runtime = CosmicRuntime()
    assert isinstance(runtime.observer_manager, ObserverManager)
    assert len(runtime.observer_manager) == 0


def test_runtime_render_for_observer_returns_reality_view(tmp_path):
    runtime = CosmicRuntime(
        config=RuntimeConfig(
            enable_physics=False,
            enable_perception=False,
            active_radius_m=1.0e30,
            max_active_objects=100,
        )
    )
    runtime.add_objects([
        _star("s1", Vector3(0.0, 1.0e16, 0.0)),
        _star("s2", Vector3(1.0e15, 1.0e16, 0.0)),
    ])
    obs = _basic_observer("obs1", config={
        "output_ppm_path": tmp_path / "out.ppm",
    })
    runtime.observer_manager.add_observer(obs)

    view = runtime.render_for_observer("obs1", _camera())
    assert isinstance(view, RealityView)
    assert view.observer_id == "obs1"
    assert view.representation_type == "flat"
    assert view.metadata["warp_factor"] == 1.0
    assert view.metadata["sample_count"] >= 1
    assert view.frame_data is not None
    assert view.frame_data.shape == (64, 64, 3)


def test_runtime_render_for_unknown_observer_raises():
    runtime = CosmicRuntime()
    with pytest.raises(KeyError):
        runtime.render_for_observer("ghost", _camera())


def test_runtime_step_all_observers_returns_one_view_per_observer(tmp_path):
    runtime = CosmicRuntime(
        config=RuntimeConfig(
            enable_physics=False,
            enable_perception=False,
            active_radius_m=1.0e30,
            max_active_objects=100,
        )
    )
    runtime.add_objects([_star("s1", Vector3(0.0, 1.0e16, 0.0))])

    runtime.observer_manager.add_observer(_basic_observer(
        "a",
        config={"output_ppm_path": tmp_path / "a.ppm"},
    ))
    runtime.observer_manager.add_observer(_basic_observer(
        "b",
        velocity_m_s=Vector3(0.5 * SPEED_OF_LIGHT_M_S, 0.0, 0.0),
        config={"output_ppm_path": tmp_path / "b.ppm"},
    ))
    views = runtime.step_all_observers(1.0)
    assert len(views) == 2
    ids = sorted(v.observer_id for v in views)
    assert ids == ["a", "b"]


def test_runtime_render_for_two_observers_no_conflict(tmp_path):
    runtime = CosmicRuntime(
        config=RuntimeConfig(
            enable_physics=False,
            enable_perception=False,
            active_radius_m=1.0e30,
            max_active_objects=100,
        )
    )
    runtime.add_objects([_star("s1", Vector3(0.0, 1.0e16, 0.0))])
    runtime.observer_manager.add_observer(_basic_observer(
        "a",
        config={"output_ppm_path": tmp_path / "a.ppm"},
    ))
    runtime.observer_manager.add_observer(_basic_observer(
        "b",
        config={"output_ppm_path": tmp_path / "b.ppm"},
    ))
    view_a = runtime.render_for_observer("a", _camera())
    view_b = runtime.render_for_observer("b", _camera())
    assert (tmp_path / "a.ppm").is_file()
    assert (tmp_path / "b.ppm").is_file()
    assert view_a.metadata["output_ppm_path"].endswith("a.ppm")
    assert view_b.metadata["output_ppm_path"].endswith("b.ppm")


class _BrokenSpacetime(SpacetimeFieldModel):
    def query_acceleration(self, position, direction=None):  # noqa: ARG002
        raise RuntimeError("intentional")

    def confidence(self) -> float:
        raise RuntimeError("intentional")


def test_runtime_render_falls_back_when_models_explode(tmp_path):
    runtime = CosmicRuntime(
        config=RuntimeConfig(
            enable_physics=False,
            enable_perception=False,
            active_radius_m=1.0e30,
            max_active_objects=100,
        )
    )
    runtime.add_objects([_star("s1", Vector3(0.0, 1.0e16, 0.0))])
    obs = _basic_observer(
        "obs",
        spacetime_model=_BrokenSpacetime(),
        config={"output_ppm_path": tmp_path / "obs.ppm"},
    )
    runtime.observer_manager.add_observer(obs)
    view = runtime.render_for_observer("obs", _camera())
    assert view.metadata["spacetime_model"] == "fallback"
    # Render must still succeed.
    assert view.frame_data is not None


def test_runtime_render_with_multiscale_uses_zone_objects(tmp_path):
    from cosmic_engine.multiscale import DEFAULT_ZONES, ScaleManager

    runtime = CosmicRuntime(
        config=RuntimeConfig(
            enable_physics=False,
            enable_perception=False,
            active_radius_m=1.0e30,
            max_active_objects=100,
        )
    )
    runtime.add_objects([_star("s1", Vector3(0.0, 1.0e16, 0.0))])
    runtime.enable_multiscale(ScaleManager(list(DEFAULT_ZONES)))
    obs = _basic_observer("obs", config={
        "output_ppm_path": tmp_path / "obs.ppm",
    })
    runtime.observer_manager.add_observer(obs)
    view = runtime.render_for_observer("obs", _camera())
    # representation type comes from the scale zone, not the flat path
    assert view.representation_type != "flat"


# --- AIViewer config integration --------------------------------------


def test_viewer_config_observer_id_default_is_none():
    from ai_viewer import AIViewerConfig
    cfg = AIViewerConfig()
    assert cfg.observer_id is None
    cfg.validate()


def test_viewer_config_observer_id_validates_non_empty():
    from ai_viewer import AIViewerConfig
    cfg = AIViewerConfig(observer_id="")
    with pytest.raises(ValueError):
        cfg.validate()
