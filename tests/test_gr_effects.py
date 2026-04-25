"""Tests for Phase 28 GR effects: lensing + black holes."""

from __future__ import annotations

import math

import numpy as np
import pytest

from ai_viewer import (
    AIViewerConfig,
    BlackHole,
    GaussianPoint,
    GaussianSplatRenderer,
    apply_black_hole_to_points,
    apply_lensing,
    apply_lensing_to_points,
    compute_deflection_angle,
)

from cosmic_engine.core.vector import Vector3
from cosmic_engine.rendering.simple_camera import SimpleCamera


_SOLAR_MASS_KG = 1.989e30
_R_SUN = 6.957e8


def _camera() -> SimpleCamera:
    return SimpleCamera(
        position_m=Vector3.zero(),
        forward=Vector3(0.0, 1.0, 0.0),
        up=Vector3(0.0, 0.0, 1.0),
        fov_degrees=90.0,
        image_width=64,
        image_height=64,
    )


def _point(
    position=(0.0, 1.0e16, 0.0),
    color=(255.0, 200.0, 100.0),
    *,
    intensity: float = 1.0,
    sigma: float = 2.0e14,
) -> GaussianPoint:
    return GaussianPoint(
        position=np.asarray(position, dtype=np.float64),
        color=np.asarray(color, dtype=np.float64),
        intensity=intensity,
        sigma=sigma,
    )


# --- compute_deflection_angle ---


def test_deflection_angle_for_solar_grazing_ray_matches_reference():
    """The classic 1.75″ value for light grazing the Sun's limb."""
    alpha = compute_deflection_angle(_R_SUN, _SOLAR_MASS_KG)
    arcsec = math.degrees(alpha) * 3600.0
    assert 1.5 < arcsec < 2.0  # textbook ≈ 1.75″


def test_deflection_angle_increases_with_mass():
    base = compute_deflection_angle(1.0e10, 1.0e30)
    bigger = compute_deflection_angle(1.0e10, 2.0e30)
    assert bigger > base
    assert bigger == pytest.approx(2.0 * base, rel=1e-9)


def test_deflection_angle_decreases_with_impact_parameter():
    near = compute_deflection_angle(1.0e8, 1.0e30)
    far = compute_deflection_angle(1.0e10, 1.0e30)
    assert near > far


def test_deflection_angle_zero_for_non_positive_inputs():
    assert compute_deflection_angle(0.0, 1.0e30) == 0.0
    assert compute_deflection_angle(-1.0, 1.0e30) == 0.0
    assert compute_deflection_angle(1.0e10, 0.0) == 0.0


def test_deflection_angle_capped_for_extreme_inputs():
    # Tiny impact parameter near a huge mass would otherwise overflow.
    alpha = compute_deflection_angle(1.0, 1.0e40)
    assert math.isfinite(alpha)
    assert alpha <= math.radians(85.0) + 1e-6


# --- apply_lensing ---


def test_apply_lensing_returns_unit_vector():
    direction = np.array([0.0, 1.0, 0.0])
    out = apply_lensing(
        direction,
        lens_position=np.array([1.0e10, 5.0e10, 0.0]),
        observer_position=np.array([0.0, 0.0, 0.0]),
        mass_kg=1.0e35,
    )
    assert math.isclose(float(np.linalg.norm(out)), 1.0, abs_tol=1e-9)


def test_apply_lensing_bends_toward_lens():
    """A ray running parallel to a lens off to the side should bend toward it."""
    direction = np.array([0.0, 1.0, 0.0])
    lens = np.array([1.0e10, 5.0e10, 0.0])  # to the +x side
    out = apply_lensing(
        direction, lens, np.array([0.0, 0.0, 0.0]), mass_kg=1.0e36
    )
    # Original x-component was 0; lensed component should now lean +x toward lens.
    assert out[0] > 0.0


def test_apply_lensing_returns_input_when_lens_behind_observer():
    direction = np.array([0.0, 1.0, 0.0])
    lens_behind = np.array([0.0, -5.0e10, 0.0])
    out = apply_lensing(direction, lens_behind, np.array([0.0, 0.0, 0.0]), 1.0e36)
    np.testing.assert_allclose(out, direction)


def test_apply_lensing_returns_input_when_ray_passes_through_lens():
    direction = np.array([0.0, 1.0, 0.0])
    lens_on_axis = np.array([0.0, 5.0e10, 0.0])
    out = apply_lensing(direction, lens_on_axis, np.array([0.0, 0.0, 0.0]), 1.0e36)
    np.testing.assert_allclose(out, direction)


def test_apply_lensing_zero_direction_returned_as_is():
    out = apply_lensing(
        np.array([0.0, 0.0, 0.0]),
        np.array([1.0e10, 5.0e10, 0.0]),
        np.array([0.0, 0.0, 0.0]),
        1.0e36,
    )
    np.testing.assert_array_equal(out, np.array([0.0, 0.0, 0.0]))


def test_apply_lensing_to_points_preserves_distance_and_count():
    points = [
        _point(position=(1.0e10, 1.0e16, 0.0)),
        _point(position=(-2.0e10, 5.0e15, 0.0)),
    ]
    observer = np.array([0.0, 0.0, 0.0])
    lens = np.array([0.0, 5.0e15, 0.0])
    lensed = apply_lensing_to_points(points, lens, 1.0e36, observer)
    assert len(lensed) == len(points)
    for original, warped in zip(points, lensed):
        original_distance = float(np.linalg.norm(original.position - observer))
        warped_distance = float(np.linalg.norm(warped.position - observer))
        assert warped_distance == pytest.approx(original_distance, rel=1e-9)
        assert warped is not original
        assert warped.metadata.get("lensed") is True


def test_apply_lensing_to_points_empty_returns_empty():
    assert apply_lensing_to_points(
        [], np.zeros(3), 1.0e36, np.zeros(3)
    ) == []


def test_apply_lensing_to_points_does_not_mutate_input():
    p = _point()
    pos_before = p.position.copy()
    apply_lensing_to_points([p], np.array([1.0e10, 5.0e10, 0.0]), 1.0e36, np.zeros(3))
    np.testing.assert_array_equal(p.position, pos_before)


# --- BlackHole ---


def test_black_hole_schwarzschild_radius_positive():
    bh = BlackHole((0.0, 0.0, 0.0), _SOLAR_MASS_KG)
    rs = bh.schwarzschild_radius()
    assert rs > 0.0
    assert rs == pytest.approx(2952.0, rel=0.01)  # ~2.95 km for the Sun


def test_black_hole_rejects_non_positive_mass():
    with pytest.raises(ValueError):
        BlackHole((0.0, 0.0, 0.0), 0.0)
    with pytest.raises(ValueError):
        BlackHole((0.0, 0.0, 0.0), -1.0)


def test_black_hole_inside_event_horizon():
    bh = BlackHole((0.0, 0.0, 0.0), 4.0e36)  # Rs ~ 5.9e9 m
    assert bh.is_inside_event_horizon((0.0, 0.0, 0.0)) is True
    assert bh.is_inside_event_horizon((1.0e9, 0.0, 0.0)) is True
    assert bh.is_inside_event_horizon((1.0e12, 0.0, 0.0)) is False


def test_black_hole_deflection_strength_increases_at_smaller_distance():
    bh = BlackHole((0.0, 0.0, 0.0), 4.0e36)
    rs = bh.schwarzschild_radius()
    near = bh.deflection_strength(rs * 1.1)
    far = bh.deflection_strength(rs * 100.0)
    assert near > far > 0.0


def test_black_hole_deflection_strength_finite_at_horizon():
    bh = BlackHole((0.0, 0.0, 0.0), 4.0e36)
    rs = bh.schwarzschild_radius()
    strength_at = bh.deflection_strength(rs)
    strength_inside = bh.deflection_strength(rs * 0.5)
    assert math.isfinite(strength_at)
    assert math.isfinite(strength_inside)


def test_apply_black_hole_absorbs_inside_horizon_points():
    bh = BlackHole((0.0, 0.0, 0.0), 4.0e36)
    rs = bh.schwarzschild_radius()
    inside = _point(position=(rs * 0.5, 0.0, 0.0))
    outside = _point(position=(rs * 5.0, 1.0e10, 0.0))
    surviving = apply_black_hole_to_points(
        [inside, outside], bh, observer_position=np.array([0.0, 0.0, 1.0e9])
    )
    ids = [s.metadata.get("lensed") for s in surviving]
    assert len(surviving) == 1  # the inside-horizon point was absorbed
    assert ids == [True]


def test_apply_black_hole_empty_input_returns_empty():
    bh = BlackHole((0.0, 0.0, 0.0), 4.0e36)
    assert apply_black_hole_to_points([], bh, np.zeros(3)) == []


def test_apply_black_hole_does_not_mutate_input():
    bh = BlackHole((0.0, 0.0, 0.0), 4.0e36)
    p = _point(position=(1.0e12, 0.0, 0.0))
    pos_before = p.position.copy()
    apply_black_hole_to_points([p], bh, np.array([0.0, -5.0e12, 0.0]))
    np.testing.assert_array_equal(p.position, pos_before)


# --- renderer with GR enabled (CPU path) ---


def test_renderer_does_not_crash_with_gr_inputs():
    bh = BlackHole((0.0, 0.0, 0.0), 4.0e36)
    base_points = [
        _point(position=(0.0, 1.0e16, 0.0), intensity=1.0),
        _point(position=(2.0e15, 1.5e16, 0.5e15), intensity=0.8),
    ]
    observer = np.array([0.0, -1.0e16, 0.0])
    points = apply_black_hole_to_points(base_points, bh, observer)
    image = GaussianSplatRenderer(64, 64, _camera()).render(points)
    assert image.shape == (64, 64, 3)
    assert image.dtype == np.uint8
    assert np.isfinite(image).all()


# --- AIViewerConfig: GR fields ---


def test_config_gr_defaults():
    cfg = AIViewerConfig()
    assert cfg.enable_gr is False
    assert cfg.black_hole_mass_kg is None
    assert cfg.black_hole_position is None
    cfg.validate()


def test_config_gr_enabled_with_valid_mass_validates():
    AIViewerConfig(
        enable_gr=True,
        black_hole_mass_kg=4.0e36,
        black_hole_position=(0.0, 0.0, 0.0),
    ).validate()


def test_config_rejects_non_positive_black_hole_mass():
    with pytest.raises(ValueError):
        AIViewerConfig(
            enable_gr=True, black_hole_mass_kg=0.0
        ).validate()
    with pytest.raises(ValueError):
        AIViewerConfig(
            enable_gr=True, black_hole_mass_kg=-1.0
        ).validate()


def test_config_rejects_wrong_position_shape():
    cfg = AIViewerConfig(enable_gr=True, black_hole_position=(0.0, 0.0))  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        cfg.validate()


# --- WebGPU renderer.enable_gr_effects (no GPU required) ---


def test_webgpu_renderer_enable_gr_effects_recreates_pipeline_state():
    from ai_viewer.neural_field.gpu import (
        WebGPUDevice,
        WebGPUSplatRenderer,
    )

    device = WebGPUDevice(backend="cpu")  # no real GPU; just exercise state flips
    renderer = WebGPUSplatRenderer(device, 32, 32)
    assert renderer._use_gr_shader is False
    renderer.enable_gr_effects(BlackHole((0.0, 0.0, 0.0), 4.0e36), enable_lensing=True)
    assert renderer._use_gr_shader is True
    assert renderer._enable_lensing is True
    assert renderer._black_hole is not None
    renderer.enable_gr_effects(black_hole=None, enable_lensing=False)
    assert renderer._use_gr_shader is False
    assert renderer._black_hole is None
