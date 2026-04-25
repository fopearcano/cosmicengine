"""Tests for Phase 14 N-body backend."""

from __future__ import annotations

import math

import numpy as np
import pytest

from cosmic_engine.core.object_types import CosmicObjectType
from cosmic_engine.core.truth import TruthLevel
from cosmic_engine.core.universe_object import UniverseObject
from cosmic_engine.core.vector import Vector3
from cosmic_engine.physics.nbody import (
    GRAVITATIONAL_CONSTANT,
    NBodySimulator,
    NBodyState,
    apply_nbody_state_to_objects,
    compute_accelerations,
    euler_step,
    leapfrog_step,
    objects_to_nbody_state,
)


def _massive(object_id: str, position: Vector3, *, mass: float = 1.0e24,
             velocity: Vector3 = Vector3.zero()) -> UniverseObject:
    return UniverseObject(
        id=object_id,
        name=object_id,
        object_type=CosmicObjectType.PLANET,
        position_m=position,
        velocity_m_s=velocity,
        truth_level=TruthLevel.PHYSICS_SIMULATED,
        mass_kg=mass,
    )


def _sun_earth_state() -> NBodyState:
    sun_mass = 1.989e30
    earth_mass = 5.972e24
    earth_distance = 1.496e11  # ~1 AU
    v_circ = math.sqrt(GRAVITATIONAL_CONSTANT * sun_mass / earth_distance)
    return NBodyState(
        object_ids=["sun", "earth"],
        positions_m=np.array(
            [[0.0, 0.0, 0.0], [earth_distance, 0.0, 0.0]],
            dtype=np.float64,
        ),
        velocities_m_s=np.array(
            [[0.0, 0.0, 0.0], [0.0, v_circ, 0.0]],
            dtype=np.float64,
        ),
        masses_kg=np.array([sun_mass, earth_mass], dtype=np.float64),
    )


# --- objects_to_nbody_state ---


def test_objects_to_nbody_state_filters_massless():
    objects = [
        _massive("a", Vector3(0.0, 0.0, 0.0), mass=2.0e24),
        UniverseObject(
            id="ghost",
            name="Ghost",
            object_type=CosmicObjectType.UNKNOWN,
            position_m=Vector3(1.0, 0.0, 0.0),
            velocity_m_s=Vector3.zero(),
            truth_level=TruthLevel.PROCEDURAL_APPROXIMATION,
            mass_kg=None,
        ),
        _massive("b", Vector3(1.0e8, 0.0, 0.0), mass=3.0e24),
    ]
    state = objects_to_nbody_state(objects)
    assert state.object_ids == ["a", "b"]
    assert state.masses_kg.tolist() == [2.0e24, 3.0e24]


def test_objects_to_nbody_state_requires_two_massive():
    only_one = [_massive("a", Vector3.zero())]
    with pytest.raises(ValueError):
        objects_to_nbody_state(only_one)


# --- compute_accelerations ---


def test_compute_accelerations_shape():
    pos = np.array(
        [[0.0, 0.0, 0.0], [1.0e10, 0.0, 0.0], [0.0, 1.0e10, 0.0]],
        dtype=np.float64,
    )
    masses = np.array([1.0e30, 1.0e24, 1.0e24], dtype=np.float64)
    accel = compute_accelerations(pos, masses)
    assert accel.shape == (3, 3)
    assert np.isfinite(accel).all()


def test_two_body_accelerations_are_opposite_in_direction():
    pos = np.array(
        [[0.0, 0.0, 0.0], [1.0e10, 0.0, 0.0]], dtype=np.float64
    )
    masses = np.array([1.0e30, 1.0e30], dtype=np.float64)
    accel = compute_accelerations(pos, masses)
    # equal masses → equal-magnitude, opposite-direction accelerations
    np.testing.assert_allclose(accel[0], -accel[1], rtol=1e-12)
    # body 0 pulled toward +x (toward body 1)
    assert accel[0, 0] > 0.0
    assert accel[1, 0] < 0.0


def test_softening_keeps_accelerations_finite_at_zero_separation():
    # Two coincident bodies with softening should not blow up.
    pos = np.array(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float64
    )
    masses = np.array([1.0e24, 1.0e24], dtype=np.float64)
    accel = compute_accelerations(pos, masses, softening_m=1.0e6)
    assert np.isfinite(accel).all()


def test_compute_accelerations_rejects_negative_softening():
    with pytest.raises(ValueError):
        compute_accelerations(
            np.zeros((2, 3)), np.ones(2), softening_m=-1.0
        )


# --- integrator steps ---


def test_euler_step_changes_positions():
    state = _sun_earth_state()
    new_state = euler_step(state, dt_seconds=3600.0)
    # Earth moved
    assert not np.array_equal(new_state.positions_m[1], state.positions_m[1])
    # Sun barely moves but velocity stays (almost) unchanged
    assert new_state.positions_m.shape == state.positions_m.shape


def test_leapfrog_step_changes_positions_and_velocities():
    state = _sun_earth_state()
    new_state = leapfrog_step(state, dt_seconds=3600.0)
    assert not np.array_equal(new_state.positions_m[1], state.positions_m[1])
    assert not np.array_equal(
        new_state.velocities_m_s[1], state.velocities_m_s[1]
    )
    assert np.isfinite(new_state.positions_m).all()
    assert np.isfinite(new_state.velocities_m_s).all()


@pytest.mark.parametrize("step_fn", [euler_step, leapfrog_step])
def test_step_rejects_non_positive_dt(step_fn):
    state = _sun_earth_state()
    with pytest.raises(ValueError):
        step_fn(state, dt_seconds=0.0)
    with pytest.raises(ValueError):
        step_fn(state, dt_seconds=-1.0)


# --- NBodySimulator ---


def test_simulator_rejects_invalid_integrator():
    with pytest.raises(ValueError):
        NBodySimulator(_sun_earth_state(), integrator="rk4")


def test_simulator_rejects_negative_softening():
    with pytest.raises(ValueError):
        NBodySimulator(_sun_earth_state(), softening_m=-1.0)


def test_simulator_run_preserves_object_count():
    state = _sun_earth_state()
    sim = NBodySimulator(state, integrator="leapfrog")
    final = sim.run(steps=10, dt_seconds=3600.0)
    assert len(final) == 2
    assert final.object_ids == ["sun", "earth"]


def test_simulator_get_state_returns_current():
    state = _sun_earth_state()
    sim = NBodySimulator(state, integrator="leapfrog")
    sim.step(3600.0)
    assert sim.get_state() is sim.state


def test_simulator_run_zero_steps_is_noop():
    state = _sun_earth_state()
    sim = NBodySimulator(state, integrator="leapfrog")
    out = sim.run(steps=0, dt_seconds=3600.0)
    np.testing.assert_array_equal(out.positions_m, state.positions_m)


def test_no_nans_for_sun_earth_simulation():
    state = _sun_earth_state()
    sim = NBodySimulator(state, integrator="leapfrog")
    final = sim.run(steps=200, dt_seconds=3600.0)
    assert np.isfinite(final.positions_m).all()
    assert np.isfinite(final.velocities_m_s).all()


def test_leapfrog_keeps_circular_orbit_radius_stable():
    """Leapfrog is symplectic — circular orbit radius should drift very little."""
    state = _sun_earth_state()
    initial_r = np.linalg.norm(state.positions_m[1] - state.positions_m[0])
    sim = NBodySimulator(state, integrator="leapfrog")
    sim.run(steps=240, dt_seconds=3600.0)  # 10 days
    final_r = np.linalg.norm(
        sim.state.positions_m[1] - sim.state.positions_m[0]
    )
    assert final_r == pytest.approx(initial_r, rel=1e-2)


# --- apply_nbody_state_to_objects ---


def test_apply_nbody_state_updates_matching_objects():
    obj_a = _massive("a", Vector3.zero())
    obj_b = _massive("b", Vector3(1.0e10, 0.0, 0.0))
    state = NBodyState(
        object_ids=["a", "b"],
        positions_m=np.array(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64
        ),
        velocities_m_s=np.array(
            [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]], dtype=np.float64
        ),
        masses_kg=np.array([1.0e24, 2.0e24], dtype=np.float64),
    )
    apply_nbody_state_to_objects([obj_a, obj_b], state)
    assert obj_a.position_m == Vector3(1.0, 2.0, 3.0)
    assert obj_a.velocity_m_s == Vector3(7.0, 8.0, 9.0)
    assert obj_b.position_m == Vector3(4.0, 5.0, 6.0)
    assert obj_b.velocity_m_s == Vector3(10.0, 11.0, 12.0)


def test_apply_nbody_state_preserves_metadata_and_truth():
    obj_a = _massive("a", Vector3.zero())
    obj_a.metadata["note"] = "keep me"
    state = NBodyState(
        object_ids=["a"],
        positions_m=np.array([[1.0, 2.0, 3.0]], dtype=np.float64),
        velocities_m_s=np.array([[4.0, 5.0, 6.0]], dtype=np.float64),
        masses_kg=np.array([1.0e24]),
    )
    apply_nbody_state_to_objects([obj_a], state)
    assert obj_a.metadata["note"] == "keep me"
    assert obj_a.truth_level is TruthLevel.PHYSICS_SIMULATED
    assert obj_a.mass_kg == 1.0e24


# --- NBodyState validation ---


def test_nbody_state_validate_catches_shape_mismatches():
    bad = NBodyState(
        object_ids=["a", "b"],
        positions_m=np.zeros((3, 3)),  # wrong N
        velocities_m_s=np.zeros((2, 3)),
        masses_kg=np.ones(2),
    )
    with pytest.raises(ValueError):
        bad.validate()


def test_nbody_state_validate_rejects_negative_mass():
    bad = NBodyState(
        object_ids=["a", "b"],
        positions_m=np.zeros((2, 3)),
        velocities_m_s=np.zeros((2, 3)),
        masses_kg=np.array([1.0, -1.0]),
    )
    with pytest.raises(ValueError):
        bad.validate()
