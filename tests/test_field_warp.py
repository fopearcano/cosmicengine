"""Tests for Phase 25 relativistic Gaussian field warp."""

from __future__ import annotations

import math

import numpy as np
import pytest

from ai_viewer import (
    AIViewerConfig,
    GaussianPoint,
    warp_gaussian_field,
    warp_gaussian_point,
)

from cosmic_engine.core.units import SPEED_OF_LIGHT_M_S
from cosmic_engine.core.vector import Vector3
from cosmic_engine.perception.observer import ObserverState


FORWARD = Vector3(0.0, 1.0, 0.0)
UP = Vector3(0.0, 0.0, 1.0)


def _observer(*, warp_factor: float = 5.0, beta: float = 0.5) -> ObserverState:
    return ObserverState(
        position_m=Vector3.zero(),
        velocity_m_s=Vector3(0.0, beta * SPEED_OF_LIGHT_M_S, 0.0),
        forward=FORWARD,
        up=UP,
        warp_factor=warp_factor,
    )


def _point(
    position=(0.0, 1.0e16, 0.0),
    color=(200.0, 180.0, 160.0),
    *,
    intensity: float = 1.5,
    sigma: float = 1.0e15,
    object_id: str | None = "g1",
    truth_level: str | None = "catalog_imported",
) -> GaussianPoint:
    return GaussianPoint(
        position=np.asarray(position, dtype=np.float64),
        color=np.asarray(color, dtype=np.float64),
        intensity=intensity,
        sigma=sigma,
        object_id=object_id,
        truth_level=truth_level,
        metadata={"note": "kept"},
    )


# --- warp_gaussian_point ---


def test_warp_returns_new_object_not_aliased():
    point = _point()
    obs = _observer()
    out = warp_gaussian_point(point, obs)
    assert out is not point
    assert out.position is not point.position
    assert out.color is not point.color


def test_warp_does_not_mutate_original():
    point = _point()
    pos_before = point.position.copy()
    color_before = point.color.copy()
    intensity_before = point.intensity
    sigma_before = point.sigma
    obs = _observer(warp_factor=10.0)
    warp_gaussian_point(point, obs)
    np.testing.assert_array_equal(point.position, pos_before)
    np.testing.assert_array_equal(point.color, color_before)
    assert point.intensity == intensity_before
    assert point.sigma == sigma_before


def test_warp_intensity_remains_non_negative_and_finite():
    obs = _observer(warp_factor=50.0, beta=0.9)
    out = warp_gaussian_point(_point(intensity=2.0), obs)
    assert out.intensity >= 0.0
    assert math.isfinite(out.intensity)


def test_warp_color_within_byte_bounds():
    obs = _observer(warp_factor=50.0, beta=0.9)
    out = warp_gaussian_point(_point(color=(250.0, 5.0, 200.0)), obs)
    for c in out.color:
        assert 0.0 <= float(c) <= 255.0


def test_warp_sigma_remains_positive():
    obs = _observer(warp_factor=50.0)
    out = warp_gaussian_point(_point(sigma=1.0e14), obs)
    assert out.sigma > 0.0
    # sigma should grow with warp_factor (gentle)
    assert out.sigma >= 1.0e14


def test_warp_preserves_distance_from_observer():
    """The warped position lies along the warped direction at the same distance."""
    obs = _observer(warp_factor=2.0, beta=0.3)
    point = _point(position=(0.0, 3.0e16, 1.0e16))
    original_distance = math.sqrt(
        sum(p ** 2 for p in point.position - np.array([0.0, 0.0, 0.0]))
    )
    out = warp_gaussian_point(point, obs)
    new_distance = math.sqrt(
        sum(p ** 2 for p in out.position - np.array([0.0, 0.0, 0.0]))
    )
    assert new_distance == pytest.approx(original_distance, rel=1e-9)


def test_warp_zero_distance_returns_safe_copy():
    """A point coincident with the observer must not blow up."""
    obs = _observer(warp_factor=10.0)
    point = _point(position=(0.0, 0.0, 0.0))
    out = warp_gaussian_point(point, obs)
    assert math.isfinite(out.position[0])
    assert out.intensity == point.intensity


def test_warp_preserves_provenance_fields():
    obs = _observer()
    out = warp_gaussian_point(_point(), obs)
    assert out.object_id == "g1"
    assert out.truth_level == "catalog_imported"
    assert out.metadata["note"] == "kept"
    assert out.metadata["warped"] is True
    assert out.metadata["warp_factor"] == pytest.approx(5.0)


def test_warp_records_original_position_in_metadata():
    obs = _observer()
    point = _point(position=(1.0, 2.0e16, 3.0e15))
    out = warp_gaussian_point(point, obs)
    assert out.metadata["original_position"] == pytest.approx(
        [1.0, 2.0e16, 3.0e15]
    )


def test_warp_no_nans_at_extreme_warp_factor():
    obs = _observer(warp_factor=500.0, beta=0.99)
    out = warp_gaussian_point(_point(intensity=10.0), obs)
    assert math.isfinite(out.position[0])
    assert math.isfinite(out.position[1])
    assert math.isfinite(out.position[2])
    assert math.isfinite(out.intensity)
    assert math.isfinite(out.sigma)


# --- warp_gaussian_field ---


def test_warp_field_preserves_count():
    obs = _observer(warp_factor=3.0)
    points = [_point(position=(float(i) * 1.0e15, 1.0e16, 0.0)) for i in range(10)]
    out = warp_gaussian_field(points, obs)
    assert len(out) == len(points)


def test_warp_field_empty_list_safe():
    obs = _observer()
    assert warp_gaussian_field([], obs) == []


def test_warp_field_max_points_truncates_deterministically():
    obs = _observer(warp_factor=2.0)
    points = [_point(position=(float(i) * 1.0e14, 1.0e16, 0.0)) for i in range(50)]
    out = warp_gaussian_field(points, obs, max_points=10)
    assert len(out) == 10
    # deterministic: same inputs -> same ids in same order
    repeat = warp_gaussian_field(points, obs, max_points=10)
    assert [p.object_id for p in out] == [p.object_id for p in repeat]


def test_warp_field_zero_max_points_returns_empty():
    obs = _observer()
    points = [_point() for _ in range(5)]
    assert warp_gaussian_field(points, obs, max_points=0) == []


def test_warp_field_with_none_ai_model_matches_deterministic():
    obs = _observer(warp_factor=4.0)
    points = [
        _point(position=(0.5e16, 1.0e16, 0.2e16)),
        _point(position=(-0.3e16, 1.5e16, 0.0), color=(100.0, 200.0, 50.0)),
    ]
    a = warp_gaussian_field(points, obs, ai_model=None)
    b = warp_gaussian_field(points, obs, ai_model=None)
    for x, y in zip(a, b):
        np.testing.assert_array_equal(x.position, y.position)
        np.testing.assert_array_equal(x.color, y.color)
        assert x.intensity == y.intensity


# --- AIViewerConfig: enable_field_warp + field_warp_mode ---


def test_config_field_warp_defaults():
    cfg = AIViewerConfig()
    assert cfg.enable_field_warp is False
    assert cfg.field_warp_mode == "deterministic"
    cfg.validate()


@pytest.mark.parametrize("mode", ["none", "deterministic", "ai"])
def test_config_field_warp_modes_validate(mode):
    AIViewerConfig(enable_field_warp=True, field_warp_mode=mode).validate()


def test_config_invalid_field_warp_mode_rejected():
    with pytest.raises(ValueError):
        AIViewerConfig(field_warp_mode="vulkan").validate()


# --- splat renderer with normalize_exposure flag ---


def test_renderer_normalize_exposure_default_is_true():
    from ai_viewer import GaussianSplatRenderer
    from cosmic_engine.rendering.simple_camera import SimpleCamera

    cam = SimpleCamera(
        position_m=Vector3.zero(),
        forward=Vector3(0.0, 1.0, 0.0),
        up=Vector3(0.0, 0.0, 1.0),
        fov_degrees=90.0,
        image_width=32,
        image_height=32,
    )
    renderer = GaussianSplatRenderer(32, 32, cam)
    assert renderer.normalize_exposure is True


def test_renderer_clamps_extreme_intensity_without_overflow():
    from ai_viewer import GaussianSplatRenderer
    from cosmic_engine.rendering.simple_camera import SimpleCamera

    cam = SimpleCamera(
        position_m=Vector3.zero(),
        forward=Vector3(0.0, 1.0, 0.0),
        up=Vector3(0.0, 0.0, 1.0),
        fov_degrees=90.0,
        image_width=32,
        image_height=32,
    )
    points = [
        GaussianPoint(
            position=np.array([0.0, 1.0e16, 0.0]),
            color=np.array([255.0, 255.0, 255.0]),
            intensity=1.0e30,  # extreme
            sigma=2.0e14,
        )
    ]
    renderer = GaussianSplatRenderer(32, 32, cam, normalize_exposure=False)
    image = renderer.render(points)
    assert image.dtype == np.uint8
    assert image.max() <= 255
    assert np.isfinite(image).all()
