"""Tests for Phase 16 unified runtime."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cosmic_engine.core.object_types import CosmicObjectType
from cosmic_engine.core.registry import UniverseRegistry
from cosmic_engine.core.time import SimulationClock
from cosmic_engine.core.truth import TruthLevel
from cosmic_engine.core.units import SPEED_OF_LIGHT_M_S
from cosmic_engine.core.universe_object import UniverseObject
from cosmic_engine.core.vector import Vector3
from cosmic_engine.perception.observer import ObserverState
from cosmic_engine.rendering.simple_camera import SimpleCamera
from cosmic_engine.runtime import (
    CosmicRuntime,
    RuntimeConfig,
    SceneState,
    run_headless_frame,
)


def _make_camera() -> SimpleCamera:
    return SimpleCamera(
        position_m=Vector3.zero(),
        forward=Vector3(0.0, 1.0, 0.0),
        up=Vector3(0.0, 0.0, 1.0),
        fov_degrees=120.0,
        image_width=64,
        image_height=64,
    )


def _make_observer() -> ObserverState:
    return ObserverState(
        position_m=Vector3.zero(),
        velocity_m_s=Vector3(0.0, 0.1 * SPEED_OF_LIGHT_M_S, 0.0),
        forward=Vector3(0.0, 1.0, 0.0),
        up=Vector3(0.0, 0.0, 1.0),
        warp_factor=1.5,
    )


def _star(object_id: str, position: Vector3) -> UniverseObject:
    return UniverseObject(
        id=object_id,
        name=object_id,
        object_type=CosmicObjectType.STAR,
        position_m=position,
        velocity_m_s=Vector3.zero(),
        truth_level=TruthLevel.CATALOG_IMPORTED,
        source="gaia",
        spectral_class="G2V",
        mass_kg=1.0e30,
        metadata={"apparent_magnitude": 1.0},
    )


def _galaxy(object_id: str, position: Vector3, *, z: float = 0.1) -> UniverseObject:
    return UniverseObject(
        id=object_id,
        name=object_id,
        object_type=CosmicObjectType.GALAXY,
        position_m=position,
        velocity_m_s=Vector3.zero(),
        truth_level=TruthLevel.CATALOG_IMPORTED,
        source="desi",
        redshift_z=z,
        metadata={"apparent_magnitude": 18.0, "data_source": "desi"},
    )


# --- RuntimeConfig validation ---


def test_runtime_config_defaults_validate():
    RuntimeConfig().validate()


@pytest.mark.parametrize(
    "overrides",
    [
        {"physics_backend": "rk4"},
        {"physics_backend": ""},
        {"render_width": 0},
        {"render_height": -1},
        {"max_active_objects": 0},
        {"max_active_objects": -5},
        {"active_radius_m": 0.0},
        {"active_radius_m": -1.0},
    ],
)
def test_runtime_config_validation_rejects_bad_fields(overrides):
    cfg = RuntimeConfig()
    for k, v in overrides.items():
        setattr(cfg, k, v)
    with pytest.raises(ValueError):
        cfg.validate()


# --- CosmicRuntime initialization ---


def test_runtime_initializes_with_defaults():
    runtime = CosmicRuntime()
    assert runtime.registry is not None
    assert runtime.clock is not None
    assert runtime.last_scene_state is None
    assert runtime.config.physics_backend == "none"


def test_runtime_initializes_with_custom_components():
    registry = UniverseRegistry()
    config = RuntimeConfig(physics_backend="exact_nbody")
    clock = SimulationClock(current_julian_date=2_451_600.0)
    runtime = CosmicRuntime(registry=registry, config=config, clock=clock)
    assert runtime.registry is registry
    assert runtime.clock is clock
    assert runtime.config.physics_backend == "exact_nbody"


def test_runtime_invalid_config_raises():
    bad_config = RuntimeConfig(physics_backend="banana")
    with pytest.raises(ValueError):
        CosmicRuntime(config=bad_config)


# --- load_sample_data ---


def test_load_sample_data_does_not_crash():
    runtime = CosmicRuntime()
    runtime.load_sample_data()
    # The bundled sample CSVs ship 10 + 10 + 10 + JPL placeholder objects.
    assert len(runtime.registry.list_objects()) > 0


# --- step ---


def test_step_advances_clock_and_returns_scene_state():
    runtime = CosmicRuntime()
    initial_jd = runtime.clock.get_julian_date()
    state = runtime.step(86_400.0)
    assert isinstance(state, SceneState)
    assert state.julian_date > initial_jd
    assert "physics: disabled" in state.notes
    assert runtime.last_scene_state is state


def test_step_with_physics_runs_when_enabled():
    runtime = CosmicRuntime(
        config=RuntimeConfig(
            enable_physics=True, physics_backend="exact_nbody"
        )
    )
    runtime.add_objects(
        [
            _star("star_a", Vector3(0.0, 0.0, 0.0)),
            _star("star_b", Vector3(1.0e11, 0.0, 0.0)),
        ]
    )
    state = runtime.step(3600.0)
    assert any("physics:" in n and "stepped" in n for n in state.notes)


def test_step_with_physics_skips_when_too_few_bodies():
    runtime = CosmicRuntime(
        config=RuntimeConfig(
            enable_physics=True, physics_backend="exact_nbody"
        )
    )
    runtime.add_objects([_star("solo", Vector3.zero())])
    state = runtime.step(3600.0)
    assert any("physics: skipped" in n for n in state.notes)


# --- SceneState construction ---


def test_scene_state_counts_match_registry():
    runtime = CosmicRuntime()
    runtime.add_objects(
        [
            _star("sa", Vector3(1.0e10, 0.0, 0.0)),
            _star("sb", Vector3(2.0e10, 0.0, 0.0)),
            _galaxy("ga", Vector3(1.0e22, 0.0, 0.0)),
        ]
    )
    state = runtime.build_scene_state()
    assert state.total_objects == 3
    assert state.object_type_counts["star"] == 2
    assert state.object_type_counts["galaxy"] == 1
    assert state.truth_level_counts["catalog_imported"] == 3
    assert state.source_counts["gaia"] == 2
    assert state.source_counts["desi"] == 1


def test_scene_state_to_dict_and_to_json_round_trip():
    state = SceneState(
        julian_date=2_451_545.0,
        total_objects=3,
        active_objects=3,
        object_type_counts={"star": 2, "galaxy": 1},
        truth_level_counts={"catalog_imported": 3},
        source_counts={"gaia": 2, "desi": 1},
        physics_backend="exact_nbody",
        perception_enabled=True,
        ai_warp_enabled=False,
        notes=["frame 1"],
    )
    d = state.to_dict()
    assert d["object_type_counts"]["star"] == 2
    assert d["notes"] == ["frame 1"]
    assert json.loads(state.to_json()) == d


# --- select_active_objects ---


def test_select_active_objects_respects_max_active_objects():
    runtime = CosmicRuntime(config=RuntimeConfig(max_active_objects=2))
    runtime.add_objects(
        [
            _star("a", Vector3(0.0, 1.0e10, 0.0)),
            _star("b", Vector3(0.0, 2.0e10, 0.0)),
            _star("c", Vector3(0.0, 3.0e10, 0.0)),
            _star("d", Vector3(0.0, 4.0e10, 0.0)),
        ]
    )
    selected = runtime.select_active_objects(Vector3.zero())
    assert len(selected) == 2
    # deterministic: sorted by id, so first two are "a" and "b"
    assert [o.id for o in selected] == ["a", "b"]


def test_select_active_objects_respects_radius():
    runtime = CosmicRuntime(
        config=RuntimeConfig(active_radius_m=1.5e10)
    )
    runtime.add_objects(
        [
            _star("near", Vector3(0.0, 1.0e10, 0.0)),
            _star("far", Vector3(0.0, 5.0e10, 0.0)),
        ]
    )
    selected = runtime.select_active_objects(Vector3.zero())
    assert [o.id for o in selected] == ["near"]


# --- run_headless_frame ---


def test_run_headless_frame_on_empty_registry():
    runtime = CosmicRuntime()
    state = run_headless_frame(
        runtime, _make_camera(), _make_observer(), output_path=None
    )
    assert isinstance(state, SceneState)
    assert state.total_objects == 0
    assert any("active: 0" in n for n in state.notes)


def test_run_headless_frame_renders_when_path_given(tmp_path: Path):
    runtime = CosmicRuntime(
        config=RuntimeConfig(enable_perception=False)
    )
    runtime.add_objects(
        [
            _star("s1", Vector3(0.0, 1.0e15, 0.0)),
            _galaxy("g1", Vector3(0.0, 1.0e22, 1.0e22)),
        ]
    )
    out = tmp_path / "frame.ppm"
    state = run_headless_frame(
        runtime, _make_camera(), _make_observer(), output_path=str(out)
    )
    assert out.is_file()
    assert state.total_objects == 2
    assert any("rendered" in n for n in state.notes)


def test_run_headless_frame_marks_skipped_non_renderable():
    runtime = CosmicRuntime()
    planet = UniverseObject(
        id="earth",
        name="Earth",
        object_type=CosmicObjectType.PLANET,
        position_m=Vector3(1.5e11, 0.0, 0.0),
        velocity_m_s=Vector3.zero(),
        truth_level=TruthLevel.PHYSICS_SIMULATED,
    )
    runtime.add_objects([planet])
    state = run_headless_frame(
        runtime, _make_camera(), _make_observer(), output_path=None
    )
    assert any("non-renderable" in n for n in state.notes)


def test_run_headless_frame_perception_note_emitted_when_enabled():
    runtime = CosmicRuntime(
        config=RuntimeConfig(enable_perception=True)
    )
    runtime.add_objects([_star("s1", Vector3(0.0, 1.0e15, 0.0))])
    state = run_headless_frame(
        runtime, _make_camera(), _make_observer(), output_path=None
    )
    assert any("perception applied" in n for n in state.notes)
