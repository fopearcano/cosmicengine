"""Tests for Phase 29 geodesic ray marching."""

from __future__ import annotations

import math

import numpy as np
import pytest

from ai_viewer import (
    AIViewerConfig,
    BlackHole,
    GaussianPoint,
    GaussianSplatRenderer,
    GeodesicRayMarcher,
    integrate_geodesic_step,
    schwarzschild_acceleration,
    trace_points_through_geodesic,
)

from cosmic_engine.core.vector import Vector3
from cosmic_engine.rendering.simple_camera import SimpleCamera


def _camera() -> SimpleCamera:
    return SimpleCamera(
        position_m=Vector3.zero(),
        forward=Vector3(0.0, 1.0, 0.0),
        up=Vector3(0.0, 0.0, 1.0),
        fov_degrees=90.0,
        image_width=64,
        image_height=64,
    )


def _point(position=(0.0, 1.0e16, 0.0)) -> GaussianPoint:
    return GaussianPoint(
        position=np.asarray(position, dtype=np.float64),
        color=np.array([255.0, 200.0, 100.0]),
        intensity=1.0,
        sigma=2.0e14,
    )


# --- schwarzschild_acceleration ---


def test_acceleration_points_toward_origin():
    a = schwarzschild_acceleration(
        np.array([1.0e10, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        mass_kg=1.0e36,
    )
    # Origin is at the negative-x side from position (1e10, 0, 0).
    assert a[0] < 0.0
    assert a[1] == pytest.approx(0.0, abs=1e-9)
    assert a[2] == pytest.approx(0.0, abs=1e-9)


def test_acceleration_zero_for_zero_mass():
    a = schwarzschild_acceleration(
        np.array([1.0e10, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        mass_kg=0.0,
    )
    np.testing.assert_array_equal(a, np.zeros(3))


def test_acceleration_zero_at_origin():
    a = schwarzschild_acceleration(
        np.zeros(3), np.array([0.0, 1.0, 0.0]), mass_kg=1.0e36
    )
    np.testing.assert_array_equal(a, np.zeros(3))


def test_acceleration_inverse_square_with_distance():
    near = schwarzschild_acceleration(
        np.array([1.0e10, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        mass_kg=1.0e36,
    )
    far = schwarzschild_acceleration(
        np.array([2.0e10, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        mass_kg=1.0e36,
    )
    # |a| ∝ 1/r²: doubling r should drop magnitude by 4×.
    near_mag = float(np.linalg.norm(near))
    far_mag = float(np.linalg.norm(far))
    assert near_mag == pytest.approx(4.0 * far_mag, rel=1e-9)


# --- integrate_geodesic_step ---


def test_integrate_step_returns_unit_direction():
    pos = np.array([1.0e22, 1.0e23, 0.0])
    direction = np.array([0.0, 1.0, 0.0])
    new_pos, new_dir = integrate_geodesic_step(
        pos, direction, step_size=1.0e22, mass_kg=1.0e50
    )
    assert math.isclose(float(np.linalg.norm(new_dir)), 1.0, abs_tol=1e-9)


def test_integrate_step_advances_position_along_direction():
    pos = np.array([0.0, 0.0, 0.0])
    direction = np.array([0.0, 1.0, 0.0])
    new_pos, new_dir = integrate_geodesic_step(
        pos, direction, step_size=1.0e10, mass_kg=0.0
    )
    np.testing.assert_allclose(new_pos, np.array([0.0, 1.0e10, 0.0]))
    np.testing.assert_allclose(new_dir, direction)


def test_integrate_step_bends_more_near_mass():
    """A small offset → larger per-step bend than a large offset."""
    direction = np.array([0.0, 1.0, 0.0])
    near_pos = np.array([1.0e22, 0.0, 0.0])
    far_pos = np.array([1.0e25, 0.0, 0.0])
    _, near_dir = integrate_geodesic_step(
        near_pos, direction, step_size=1.0e22, mass_kg=1.0e50
    )
    _, far_dir = integrate_geodesic_step(
        far_pos, direction, step_size=1.0e22, mass_kg=1.0e50
    )
    near_bend = math.acos(max(-1.0, min(1.0, float(np.dot(direction, near_dir)))))
    far_bend = math.acos(max(-1.0, min(1.0, float(np.dot(direction, far_dir)))))
    assert near_bend > far_bend


def test_integrate_step_zero_direction_passes_through():
    pos = np.array([1.0, 2.0, 3.0])
    new_pos, new_dir = integrate_geodesic_step(
        pos, np.zeros(3), step_size=1.0e10, mass_kg=1.0e36
    )
    np.testing.assert_array_equal(new_pos, pos)
    np.testing.assert_array_equal(new_dir, np.zeros(3))


def test_integrate_step_rejects_non_positive_step_size():
    with pytest.raises(ValueError):
        integrate_geodesic_step(
            np.zeros(3), np.array([0.0, 1.0, 0.0]), 0.0, 1.0e36
        )
    with pytest.raises(ValueError):
        integrate_geodesic_step(
            np.zeros(3), np.array([0.0, 1.0, 0.0]), -1.0, 1.0e36
        )


def test_integrate_step_no_nans_at_extreme_inputs():
    new_pos, new_dir = integrate_geodesic_step(
        np.array([1.0e8, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        step_size=1.0e8,
        mass_kg=1.0e50,
    )
    assert np.isfinite(new_pos).all()
    assert np.isfinite(new_dir).all()
    assert math.isclose(float(np.linalg.norm(new_dir)), 1.0, abs_tol=1e-6)


# --- GeodesicRayMarcher ---


def test_ray_marcher_validates_construction():
    bh = BlackHole((0.0, 0.0, 0.0), 1.0e36)
    with pytest.raises(ValueError):
        GeodesicRayMarcher(bh, step_size=0.0, max_steps=8)
    with pytest.raises(ValueError):
        GeodesicRayMarcher(bh, step_size=-1.0, max_steps=8)
    with pytest.raises(ValueError):
        GeodesicRayMarcher(bh, step_size=1.0, max_steps=0)


def test_ray_marcher_terminates_within_max_steps():
    bh = BlackHole((0.0, 0.0, 0.0), 1.0e30)  # tiny mass, no horizon impact
    marcher = GeodesicRayMarcher(bh, step_size=1.0e10, max_steps=5)
    direction, absorbed = marcher.trace_ray(
        origin=np.array([0.0, -1.0e22, 0.0]),
        direction=np.array([0.0, 1.0, 0.0]),
    )
    assert absorbed is False
    assert math.isclose(float(np.linalg.norm(direction)), 1.0, abs_tol=1e-6)


def test_ray_marcher_absorbs_ray_starting_inside_event_horizon():
    bh = BlackHole((0.0, 0.0, 0.0), 4.0e36)
    marcher = GeodesicRayMarcher(bh, step_size=1.0e8, max_steps=4)
    rs = bh.schwarzschild_radius()
    direction, absorbed = marcher.trace_ray(
        origin=np.array([rs * 0.5, 0.0, 0.0]),
        direction=np.array([0.0, 1.0, 0.0]),
    )
    assert absorbed is True


def test_ray_marcher_absorbs_ray_aimed_at_black_hole():
    """A ray fired straight at the BH should hit the horizon within max_steps."""
    bh = BlackHole((0.0, 0.0, 0.0), 4.0e50)
    # Total path = 100 * 1e23 = 1e25, easily past the 5e24-m start distance.
    marcher = GeodesicRayMarcher(bh, step_size=1.0e23, max_steps=100)
    direction, absorbed = marcher.trace_ray(
        origin=np.array([0.0, -5.0e24, 0.0]),
        direction=np.array([0.0, 1.0, 0.0]),
    )
    assert absorbed is True


def test_ray_marcher_zero_direction_returns_unabsorbed():
    bh = BlackHole((0.0, 0.0, 0.0), 1.0e36)
    marcher = GeodesicRayMarcher(bh, step_size=1.0e10, max_steps=4)
    direction, absorbed = marcher.trace_ray(
        origin=np.array([1.0e10, 1.0e10, 0.0]),
        direction=np.zeros(3),
    )
    assert absorbed is False
    np.testing.assert_array_equal(direction, np.zeros(3))


def test_ray_marcher_no_nans_for_high_mass():
    bh = BlackHole((0.0, 0.0, 0.0), 1.0e52)
    marcher = GeodesicRayMarcher(bh, step_size=1.0e23, max_steps=16)
    final, _ = marcher.trace_ray(
        origin=np.array([0.0, -1.0e25, 0.0]),
        direction=np.array([1.0, 1.0, 0.0]) / math.sqrt(2.0),
    )
    assert np.isfinite(final).all()


# --- trace_points_through_geodesic ---


def test_trace_points_preserves_distance():
    bh = BlackHole((0.0, 0.0, 0.0), 1.0e36)
    marcher = GeodesicRayMarcher(bh, step_size=1.0e10, max_steps=4)
    points = [_point(position=(1.0e15, 1.0e16, 0.0))]
    # Observer is well outside the BH's event horizon (~1.5e9 m) so the
    # ray doesn't get absorbed during the trace.
    observer = np.array([0.0, -1.0e16, 0.0])
    out = trace_points_through_geodesic(points, marcher, observer)
    assert len(out) == 1
    original_distance = float(np.linalg.norm(points[0].position - observer))
    new_distance = float(np.linalg.norm(out[0].position - observer))
    assert new_distance == pytest.approx(original_distance, rel=1e-6)
    assert out[0].metadata["geodesic_traced"] is True


def test_trace_points_empty_list_safe():
    bh = BlackHole((0.0, 0.0, 0.0), 1.0e36)
    marcher = GeodesicRayMarcher(bh, step_size=1.0e10, max_steps=4)
    assert trace_points_through_geodesic([], marcher, np.zeros(3)) == []


def test_trace_points_does_not_mutate_input():
    bh = BlackHole((0.0, 0.0, 0.0), 1.0e36)
    marcher = GeodesicRayMarcher(bh, step_size=1.0e10, max_steps=4)
    p = _point()
    pos_before = p.position.copy()
    trace_points_through_geodesic([p], marcher, np.zeros(3))
    np.testing.assert_array_equal(p.position, pos_before)


def test_trace_points_drops_absorbed():
    bh = BlackHole((0.0, 0.0, 0.0), 4.0e50)
    # Total ray path = 100 * 1e23 = 1e25 m, past the 5e24-m start distance.
    marcher = GeodesicRayMarcher(bh, step_size=1.0e23, max_steps=100)
    target = _point(position=(0.0, 1.0e24, 0.0))  # ray aimed at the BH
    out = trace_points_through_geodesic(
        [target], marcher, observer_position=np.array([0.0, -5.0e24, 0.0])
    )
    assert out == []  # absorbed


# --- renderer end-to-end ---


def test_renderer_does_not_crash_with_geodesic_points():
    bh = BlackHole((0.0, 0.0, 0.0), 1.0e50)
    marcher = GeodesicRayMarcher(bh, step_size=1.0e22, max_steps=8)
    base = [_point(position=(2.0e15, 1.0e16, 0.0))]
    out = trace_points_through_geodesic(
        base, marcher, observer_position=np.array([0.0, -1.0e16, 0.0])
    )
    image = GaussianSplatRenderer(64, 64, _camera()).render(out)
    assert image.shape == (64, 64, 3)
    assert np.isfinite(image).all()


# --- AIViewerConfig: gr_mode + geodesic params ---


def test_config_gr_mode_default():
    cfg = AIViewerConfig()
    assert cfg.gr_mode == "lensing"
    assert cfg.geodesic_steps == 8
    assert cfg.geodesic_step_size == pytest.approx(1.0e9)
    cfg.validate()


@pytest.mark.parametrize("mode", ["none", "lensing", "geodesic"])
def test_config_gr_mode_valid_values(mode):
    AIViewerConfig(gr_mode=mode).validate()


def test_config_gr_mode_rejects_unknown():
    with pytest.raises(ValueError):
        AIViewerConfig(gr_mode="kerr").validate()


def test_config_geodesic_steps_must_be_positive():
    with pytest.raises(ValueError):
        AIViewerConfig(geodesic_steps=0).validate()
    with pytest.raises(ValueError):
        AIViewerConfig(geodesic_steps=-1).validate()


def test_config_geodesic_step_size_must_be_positive():
    with pytest.raises(ValueError):
        AIViewerConfig(geodesic_step_size=0.0).validate()
    with pytest.raises(ValueError):
        AIViewerConfig(geodesic_step_size=-1.0).validate()


# --- WebGPU renderer.enable_gr_effects geodesic mode ---


def test_webgpu_renderer_enable_geodesic_state():
    from ai_viewer.neural_field.gpu import (
        WebGPUDevice,
        WebGPUSplatRenderer,
    )

    device = WebGPUDevice(backend="cpu")  # no real GPU; just test state
    renderer = WebGPUSplatRenderer(device, 32, 32)
    renderer.enable_gr_effects(
        BlackHole((0.0, 0.0, 0.0), 4.0e36),
        gr_mode="geodesic",
        geodesic_steps=12,
        geodesic_step_size=2.0e9,
    )
    assert renderer._gr_mode == "geodesic"
    assert renderer._geodesic_steps == 12
    assert renderer._geodesic_step_size == pytest.approx(2.0e9)
    assert renderer._use_gr_shader is True


def test_webgpu_renderer_rejects_invalid_gr_mode():
    from ai_viewer.neural_field.gpu import (
        WebGPUDevice,
        WebGPUSplatRenderer,
    )

    device = WebGPUDevice(backend="cpu")
    renderer = WebGPUSplatRenderer(device, 32, 32)
    with pytest.raises(ValueError):
        renderer.enable_gr_effects(gr_mode="kerr")


def test_webgpu_renderer_geodesic_validates_step_params():
    from ai_viewer.neural_field.gpu import (
        WebGPUDevice,
        WebGPUSplatRenderer,
    )

    device = WebGPUDevice(backend="cpu")
    renderer = WebGPUSplatRenderer(device, 32, 32)
    with pytest.raises(ValueError):
        renderer.enable_gr_effects(geodesic_steps=0)
    with pytest.raises(ValueError):
        renderer.enable_gr_effects(geodesic_step_size=-1.0)
