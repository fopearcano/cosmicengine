"""Tests for the Phase 34 subjective-time / causality system."""

from __future__ import annotations

import numpy as np
import pytest

from cosmic_engine.core.units import SPEED_OF_LIGHT_M_S
from cosmic_engine.core.vector import Vector3
from cosmic_engine.observer import Observer
from cosmic_engine.runtime import CosmicRuntime, RuntimeConfig
from cosmic_engine.time import (
    Event,
    EventStore,
    advance_proper_time,
    compute_retarded_time,
    gamma_from_beta,
    gravitational_potential_weak,
    is_event_visible,
)


# --- gamma_from_beta ---------------------------------------------------


def test_gamma_at_zero_is_one():
    assert gamma_from_beta(0.0) == pytest.approx(1.0)


def test_gamma_increases_with_beta():
    last = 1.0
    for b in (0.1, 0.3, 0.5, 0.7, 0.9, 0.99):
        g = gamma_from_beta(b)
        assert g > last
        last = g


def test_gamma_handles_negative_beta():
    assert gamma_from_beta(-0.6) == pytest.approx(gamma_from_beta(0.6))


def test_gamma_clamps_at_one():
    # Should not raise / produce inf even when beta is right at 1.
    g = gamma_from_beta(1.0)
    assert g > 1.0e3


# --- advance_proper_time -----------------------------------------------


def test_proper_time_at_rest_equals_coordinate_time():
    assert advance_proper_time(0.0, 1.0, 0.0) == pytest.approx(1.0)


def test_proper_time_lt_coordinate_for_moving_observer():
    tau = advance_proper_time(0.0, 1.0, 0.6)
    assert tau == pytest.approx(0.8)
    assert tau < 1.0


def test_proper_time_rejects_negative_dt():
    with pytest.raises(ValueError):
        advance_proper_time(0.0, -1.0, 0.0)


def test_proper_time_with_gr_factor_runs_slower_in_well():
    # Negative potential -> factor < 1 -> dτ smaller than the SR-only
    # case at the same beta.
    tau_sr = advance_proper_time(0.0, 1.0, 0.0)
    tau_gr = advance_proper_time(0.0, 1.0, 0.0, gravitational_potential=-1.0e16)
    assert tau_gr < tau_sr


def test_proper_time_with_gr_factor_clamps_above_zero():
    # Absurd potential -> factor would go negative; we floor it.
    tau = advance_proper_time(0.0, 1.0, 0.0, gravitational_potential=-1.0e30)
    assert tau > 0.0


# --- gravitational_potential_weak --------------------------------------


def test_potential_empty_masses_is_zero():
    assert gravitational_potential_weak(np.zeros(3), []) == 0.0


def test_potential_negative_near_a_mass():
    phi = gravitational_potential_weak(
        np.array([1.0e9, 0.0, 0.0]),
        [(np.zeros(3), 1.989e30)],  # solar mass at origin
    )
    assert phi < 0.0


def test_potential_clamps_at_origin():
    # Coincident with a mass — without the floor we'd get -inf.
    phi = gravitational_potential_weak(
        np.zeros(3),
        [(np.zeros(3), 1.989e30)],
    )
    assert phi != float("-inf")
    assert phi < 0.0


def test_potential_skips_invalid_masses():
    phi = gravitational_potential_weak(
        np.array([1.0e9, 0.0, 0.0]),
        [(np.zeros(3), 0.0), (np.zeros(3), -1.0)],
    )
    assert phi == 0.0


def test_potential_position_shape_validated():
    with pytest.raises(ValueError):
        gravitational_potential_weak(
            np.array([1.0, 2.0]),  # wrong shape
            [(np.zeros(3), 1.0e30)],
        )


# --- Event / EventStore ------------------------------------------------


def test_event_position_normalized_to_3vector():
    e = Event(id="e", position_m=[1.0, 2.0, 3.0], time_t=0.0)
    assert isinstance(e.position_m, np.ndarray)
    assert e.position_m.shape == (3,)


def test_event_rejects_bad_position_shape():
    with pytest.raises(ValueError):
        Event(id="e", position_m=[1.0, 2.0], time_t=0.0)


def test_event_to_dict_round_trip():
    e = Event(
        id="e1",
        position_m=np.array([1.0, 2.0, 3.0]),
        time_t=4.0,
        payload={"k": "v"},
        source="testing",
    )
    out = e.to_dict()
    assert out["id"] == "e1"
    assert out["position_m"] == [1.0, 2.0, 3.0]
    assert out["time_t"] == 4.0
    assert out["payload"] == {"k": "v"}
    assert out["source"] == "testing"


def test_event_store_add_and_list():
    store = EventStore()
    assert len(store) == 0
    e1 = Event("a", np.zeros(3), 0.0)
    e2 = Event("b", np.zeros(3), 1.0)
    store.add_event(e1)
    store.add_event(e2)
    assert len(store) == 2
    listed = store.list_events()
    assert [e.id for e in listed] == ["a", "b"]
    # Defensive copy: mutating the result doesn't change the store.
    listed.clear()
    assert len(store.list_events()) == 2


def test_event_store_query_time_window():
    store = EventStore()
    for i, t in enumerate([0.0, 1.0, 2.0, 3.0]):
        store.add_event(Event(f"e{i}", np.zeros(3), t))
    inside = store.query_time_window(1.0, 2.0)
    assert [e.id for e in inside] == ["e1", "e2"]


def test_event_store_query_rejects_inverted_window():
    store = EventStore()
    with pytest.raises(ValueError):
        store.query_time_window(2.0, 1.0)


# --- is_event_visible / compute_retarded_time --------------------------


def test_event_at_origin_visible_after_zero_delay():
    e = Event("e", np.zeros(3), time_t=0.0)
    assert is_event_visible(e, np.zeros(3), 0.0) is True


def test_distant_event_not_visible_yet():
    # 1 light-second away, observer at t = 0.5 s -> not yet.
    e = Event("e", np.array([SPEED_OF_LIGHT_M_S, 0.0, 0.0]), time_t=0.0)
    assert is_event_visible(e, np.zeros(3), 0.5) is False
    assert is_event_visible(e, np.zeros(3), 1.0 + 1.0e-3) is True


def test_future_event_never_visible():
    e = Event("e", np.zeros(3), time_t=10.0)
    assert is_event_visible(e, np.zeros(3), 5.0) is False


def test_retarded_time_decreases_with_distance():
    obs_pos = np.zeros(3)
    obs_t = 100.0
    near = compute_retarded_time(obs_pos, obs_t, np.array([1.0e8, 0.0, 0.0]))
    far = compute_retarded_time(obs_pos, obs_t, np.array([1.0e10, 0.0, 0.0]))
    # Farther source -> emission time was earlier (smaller t).
    assert far < near
    assert near < obs_t


# --- Observer.advance_time --------------------------------------------


def _basic_observer(**overrides) -> Observer:
    kwargs = {
        "id": "obs",
        "position_m": Vector3.zero(),
        "velocity_m_s": Vector3.zero(),
        "forward": Vector3(0.0, 1.0, 0.0),
        "up": Vector3(0.0, 0.0, 1.0),
    }
    kwargs.update(overrides)
    return Observer(**kwargs)


def test_observer_advance_time_at_rest_matches_coordinate():
    obs = _basic_observer()
    obs.advance_time(2.0)
    assert obs.coordinate_time_t == pytest.approx(2.0)
    assert obs.proper_time_tau == pytest.approx(2.0)


def test_observer_advance_time_for_moving_observer_is_dilated():
    obs = _basic_observer(
        velocity_m_s=Vector3(0.6 * SPEED_OF_LIGHT_M_S, 0.0, 0.0)
    )
    obs.advance_time(1.0)
    assert obs.coordinate_time_t == pytest.approx(1.0)
    # gamma(0.6) = 1.25 -> dτ = 0.8
    assert obs.proper_time_tau == pytest.approx(0.8)
    assert obs.proper_time_tau < obs.coordinate_time_t


def test_observer_advance_time_rejects_negative_dt():
    obs = _basic_observer()
    with pytest.raises(ValueError):
        obs.advance_time(-0.1)


def test_observer_advance_time_uses_gr_potential():
    obs = _basic_observer()
    obs.advance_time(
        1.0,
        masses_for_potential=[
            (np.array([1.0e9, 0.0, 0.0]), 1.989e30),
        ],
    )
    # The GR factor is small but should produce dτ slightly less than 1.
    assert obs.proper_time_tau < 1.0


# --- Runtime integration ----------------------------------------------


def test_runtime_has_event_store_and_coordinate_time():
    runtime = CosmicRuntime()
    assert isinstance(runtime.event_store, EventStore)
    assert runtime.coordinate_time_t == 0.0


def test_runtime_step_advances_observers():
    runtime = CosmicRuntime(
        config=RuntimeConfig(enable_physics=False, max_active_objects=10)
    )
    runtime.observer_manager.add_observer(_basic_observer(id="a"))
    runtime.observer_manager.add_observer(_basic_observer(
        id="b",
        velocity_m_s=Vector3(0.6 * SPEED_OF_LIGHT_M_S, 0.0, 0.0),
    ))
    runtime.step(1.0)
    a = runtime.observer_manager.get_observer("a")
    b = runtime.observer_manager.get_observer("b")
    assert a.coordinate_time_t == pytest.approx(1.0)
    assert b.coordinate_time_t == pytest.approx(1.0)
    # Moving observer's proper time lags.
    assert b.proper_time_tau < a.proper_time_tau


def test_runtime_step_advances_global_coordinate_time():
    runtime = CosmicRuntime(
        config=RuntimeConfig(enable_physics=False, max_active_objects=10)
    )
    runtime.step(1.0)
    runtime.step(2.5)
    assert runtime.coordinate_time_t == pytest.approx(3.5)


def test_runtime_step_rejects_negative_delta():
    runtime = CosmicRuntime(
        config=RuntimeConfig(enable_physics=False, max_active_objects=10)
    )
    with pytest.raises(ValueError):
        runtime.step(-0.1)


def test_runtime_get_visible_events_filters_by_lightcone():
    runtime = CosmicRuntime(
        config=RuntimeConfig(enable_physics=False, max_active_objects=10)
    )
    obs = _basic_observer(id="a")
    runtime.observer_manager.add_observer(obs)
    # Local event: visible immediately.
    runtime.event_store.add_event(
        Event("local", np.zeros(3), time_t=0.0)
    )
    # 1 light-second away: not visible until 1 s.
    runtime.event_store.add_event(
        Event(
            "remote",
            np.array([SPEED_OF_LIGHT_M_S, 0.0, 0.0]),
            time_t=0.0,
        )
    )
    runtime.step(0.1)
    assert {e.id for e in runtime.get_visible_events(obs)} == {"local"}
    runtime.step(1.0)  # total t = 1.1
    assert {e.id for e in runtime.get_visible_events(obs)} == {"local", "remote"}


def test_runtime_render_for_observer_attaches_time_and_events(tmp_path):
    runtime = CosmicRuntime(
        config=RuntimeConfig(
            enable_physics=False,
            max_active_objects=10,
            active_radius_m=1.0e30,
        )
    )
    obs = _basic_observer(
        id="a",
        config={"output_ppm_path": tmp_path / "a.ppm"},
    )
    runtime.observer_manager.add_observer(obs)
    runtime.event_store.add_event(
        Event("local", np.zeros(3), time_t=0.0)
    )
    runtime.step(1.0)
    from cosmic_engine.rendering import SimpleCamera
    cam = SimpleCamera(
        position_m=Vector3.zero(),
        forward=Vector3(0.0, 1.0, 0.0),
        up=Vector3(0.0, 0.0, 1.0),
        fov_degrees=90.0,
        image_width=32,
        image_height=32,
    )
    view = runtime.render_for_observer("a", cam)
    assert view.coordinate_time_t == pytest.approx(1.0)
    assert view.proper_time_tau == pytest.approx(1.0)
    assert view.visible_event_count == 1
    out = view.to_dict()
    assert out["proper_time_tau"] == pytest.approx(1.0)
    assert out["coordinate_time_t"] == pytest.approx(1.0)
    assert out["visible_event_count"] == 1
