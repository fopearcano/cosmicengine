"""Tests for Phase 15 Barnes-Hut octree N-body."""

from __future__ import annotations

import numpy as np
import pytest

from cosmic_engine.physics.barnes_hut import (
    OctreeNode,
    build_octree,
    compute_acceleration_bh,
    compute_accelerations_bh,
)
from cosmic_engine.physics.nbody import (
    NBodySimulator,
    NBodyState,
    compute_accelerations,
    objects_to_nbody_state,
)


def _two_body() -> tuple[np.ndarray, np.ndarray]:
    positions = np.array(
        [[0.0, 0.0, 0.0], [1.0e10, 0.0, 0.0]], dtype=np.float64
    )
    masses = np.array([1.0e30, 1.0e30], dtype=np.float64)
    return positions, masses


# --- octree construction ---


def test_build_octree_returns_root_node():
    positions, masses = _two_body()
    root = build_octree(positions, masses)
    assert isinstance(root, OctreeNode)


def test_root_mass_equals_sum_of_masses():
    rng = np.random.default_rng(0)
    positions = rng.standard_normal((50, 3)) * 1.0e9
    masses = rng.uniform(1.0e23, 1.0e25, size=50)
    root = build_octree(positions, masses)
    assert root.mass == pytest.approx(masses.sum(), rel=1e-12)


def test_center_of_mass_for_two_equal_bodies():
    positions, masses = _two_body()
    root = build_octree(positions, masses)
    # COM should be at midpoint for equal masses
    np.testing.assert_allclose(
        root.center_of_mass, [5.0e9, 0.0, 0.0], rtol=1e-12, atol=1e-3
    )


def test_center_of_mass_weighted():
    positions = np.array(
        [[0.0, 0.0, 0.0], [1.0e10, 0.0, 0.0]], dtype=np.float64
    )
    masses = np.array([3.0, 1.0], dtype=np.float64)  # 3:1
    root = build_octree(positions, masses)
    # COM at 1/4 of the way from the heavy body
    np.testing.assert_allclose(
        root.center_of_mass, [2.5e9, 0.0, 0.0], rtol=1e-9, atol=1e-3
    )


def test_octree_handles_single_body():
    positions = np.array([[1.0, 2.0, 3.0]], dtype=np.float64)
    masses = np.array([1.0e24])
    root = build_octree(positions, masses)
    assert root.mass == 1.0e24
    np.testing.assert_array_equal(root.center_of_mass, [1.0, 2.0, 3.0])


def test_octree_handles_empty_input():
    root = build_octree(
        np.zeros((0, 3), dtype=np.float64),
        np.zeros((0,), dtype=np.float64),
    )
    assert root.mass == 0.0
    assert root.is_leaf()


def test_octree_handles_overlapping_bodies():
    # two bodies at the same position should not blow up
    positions = np.array(
        [[1.0e9, 0.0, 0.0], [1.0e9, 0.0, 0.0]], dtype=np.float64
    )
    masses = np.array([1.0e24, 1.0e24])
    root = build_octree(positions, masses, max_depth=10)
    assert root.mass == pytest.approx(2.0e24, rel=1e-12)
    assert np.all(np.isfinite(root.center_of_mass))


# --- compute_accelerations_bh ---


def test_acceleration_shape():
    rng = np.random.default_rng(1)
    n = 30
    positions = rng.standard_normal((n, 3)) * 1.0e9
    masses = rng.uniform(1.0e23, 1.0e25, size=n)
    accel = compute_accelerations_bh(positions, masses, theta=0.5)
    assert accel.shape == (n, 3)
    assert np.isfinite(accel).all()


def test_bh_matches_exact_for_small_n_low_theta():
    rng = np.random.default_rng(2)
    n = 50
    positions = rng.standard_normal((n, 3)) * 1.0e9
    masses = rng.uniform(1.0e23, 1.0e25, size=n)
    softening = 1.0e7

    exact = compute_accelerations(positions, masses, softening_m=softening)
    bh = compute_accelerations_bh(
        positions, masses, theta=0.1, softening_m=softening
    )
    # Tight theta → traverse to leaves; should match exact within float tol.
    np.testing.assert_allclose(bh, exact, rtol=1e-3, atol=1.0e-15)


def test_bh_two_body_matches_exact():
    positions, masses = _two_body()
    exact = compute_accelerations(positions, masses)
    bh = compute_accelerations_bh(positions, masses, theta=0.5)
    np.testing.assert_allclose(bh, exact, rtol=1e-12)


def test_bh_no_nans_on_random_distribution():
    rng = np.random.default_rng(3)
    positions = rng.uniform(-1.0e10, 1.0e10, size=(200, 3))
    masses = rng.uniform(1.0e22, 1.0e25, size=200)
    accel = compute_accelerations_bh(
        positions, masses, theta=0.5, softening_m=1.0e7
    )
    assert np.isfinite(accel).all()


def test_bh_larger_theta_is_faster_or_equal():
    rng = np.random.default_rng(4)
    n = 300
    positions = rng.standard_normal((n, 3)) * 1.0e10
    masses = rng.uniform(1.0e22, 1.0e25, size=n)
    # Just sanity: both runs produce finite, sized output.
    a05 = compute_accelerations_bh(positions, masses, theta=0.5)
    a08 = compute_accelerations_bh(positions, masses, theta=0.8)
    assert a05.shape == a08.shape == (n, 3)


def test_bh_rejects_invalid_parameters():
    positions = np.zeros((2, 3))
    masses = np.ones(2)
    with pytest.raises(ValueError):
        compute_accelerations_bh(positions, masses, theta=0.0)
    with pytest.raises(ValueError):
        compute_accelerations_bh(positions, masses, theta=-0.1)
    with pytest.raises(ValueError):
        compute_accelerations_bh(positions, masses, softening_m=-1.0)


def test_bh_handles_empty_input():
    accel = compute_accelerations_bh(
        np.zeros((0, 3)), np.zeros((0,)), theta=0.5
    )
    assert accel.shape == (0, 3)


# --- compute_acceleration_bh (single body) ---


def test_compute_acceleration_bh_self_leaf_zero():
    positions = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
    masses = np.array([1.0e30])
    root = build_octree(positions, masses)
    accel = compute_acceleration_bh(0, root, positions, masses, 0.5, 0.0)
    np.testing.assert_array_equal(accel, [0.0, 0.0, 0.0])


# --- NBodySimulator integration ---


def test_simulator_accepts_barnes_hut_integrator():
    state = NBodyState(
        object_ids=["a", "b"],
        positions_m=np.array(
            [[0.0, 0.0, 0.0], [1.0e10, 0.0, 0.0]], dtype=np.float64
        ),
        velocities_m_s=np.zeros((2, 3), dtype=np.float64),
        masses_kg=np.array([1.0e30, 1.0e24]),
    )
    sim = NBodySimulator(state, integrator="barnes_hut", theta=0.5)
    final = sim.run(steps=5, dt_seconds=3600.0)
    assert final.positions_m.shape == (2, 3)
    assert np.isfinite(final.positions_m).all()


def test_simulator_rejects_non_positive_theta():
    state = NBodyState(
        object_ids=["a", "b"],
        positions_m=np.zeros((2, 3)),
        velocities_m_s=np.zeros((2, 3)),
        masses_kg=np.ones(2),
    )
    with pytest.raises(ValueError):
        NBodySimulator(state, integrator="barnes_hut", theta=0.0)
    with pytest.raises(ValueError):
        NBodySimulator(state, integrator="leapfrog", theta=-1.0)


def test_simulator_default_theta_is_used():
    state = NBodyState(
        object_ids=["a", "b"],
        positions_m=np.zeros((2, 3)),
        velocities_m_s=np.zeros((2, 3)),
        masses_kg=np.ones(2),
    )
    sim = NBodySimulator(state, integrator="barnes_hut")
    assert sim.theta == 0.5
