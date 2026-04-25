"""Tests for Phase 24 Gaussian splat renderer."""

from __future__ import annotations

import math

import numpy as np
import pytest

from ai_viewer import (
    AIViewerConfig,
    GaussianPoint,
    GaussianSplatRenderer,
    build_gaussian_field_from_density,
    build_gaussian_field_from_galaxy_batch,
)

from cosmic_engine.core.object_types import CosmicObjectType
from cosmic_engine.core.truth import TruthLevel
from cosmic_engine.core.universe_object import UniverseObject
from cosmic_engine.core.vector import Vector3
from cosmic_engine.cosmos.galaxy import GalaxyProperties, create_galaxy_object
from cosmic_engine.rendering import SimpleCamera, build_galaxy_field_batch


def _camera(width: int = 64, height: int = 64) -> SimpleCamera:
    return SimpleCamera(
        position_m=Vector3.zero(),
        forward=Vector3(0.0, 1.0, 0.0),
        up=Vector3(0.0, 0.0, 1.0),
        fov_degrees=120.0,
        image_width=width,
        image_height=height,
    )


def _galaxy(object_id: str, position: Vector3, *, z: float = 0.1) -> UniverseObject:
    return create_galaxy_object(
        id=object_id,
        name=object_id,
        position_m=position,
        properties=GalaxyProperties(redshift_z=z, apparent_magnitude=10.0),
        source="CSV",
    )


# --- GaussianPoint ---


def test_gaussian_point_normalizes_inputs():
    p = GaussianPoint(
        position=[1.0, 2.0, 3.0],
        color=(255, 128, 0),
        intensity=2,
        sigma=10,
    )
    assert isinstance(p.position, np.ndarray)
    assert p.position.shape == (3,)
    assert p.color.shape == (3,)
    assert isinstance(p.intensity, float)
    assert isinstance(p.sigma, float)


def test_gaussian_point_accepts_numpy_arrays():
    p = GaussianPoint(
        position=np.array([1.0, 2.0, 3.0]),
        color=np.array([255.0, 0.0, 0.0]),
        intensity=1.5,
        sigma=5.0,
    )
    np.testing.assert_array_equal(p.position, [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(p.color, [255.0, 0.0, 0.0])


# --- field builders ---


def test_build_gaussian_field_from_galaxy_batch_one_per_object():
    galaxies = [
        _galaxy("g1", Vector3(0.0, 1.0e22, 0.0), z=0.1),
        _galaxy("g2", Vector3(1.0e22, 1.0e22, 0.0), z=0.5),
    ]
    batch = build_galaxy_field_batch(galaxies, _camera())
    points = build_gaussian_field_from_galaxy_batch(batch)
    assert len(points) == 2
    for p in points:
        assert p.position.shape == (3,)
        assert p.color.shape == (3,)
        assert p.sigma > 0.0


def test_build_gaussian_field_empty_batch():
    batch = build_galaxy_field_batch([], _camera())
    assert build_gaussian_field_from_galaxy_batch(batch) == []


def test_build_gaussian_field_rejects_non_positive_sigma_scale():
    batch = build_galaxy_field_batch(
        [_galaxy("g", Vector3(0.0, 1.0e22, 0.0))], _camera()
    )
    with pytest.raises(ValueError):
        build_gaussian_field_from_galaxy_batch(batch, sigma_scale=0.0)


def test_build_gaussian_field_from_density_rejects_low_density():
    grid = np.zeros((4, 4, 4))
    grid[1, 2, 3] = 0.5  # above threshold
    grid[0, 0, 0] = 0.001  # below threshold
    points = build_gaussian_field_from_density(grid, threshold=0.01)
    assert len(points) == 1


def test_build_gaussian_field_from_density_rejects_non_3d():
    with pytest.raises(ValueError):
        build_gaussian_field_from_density(np.zeros((4, 4)))


def test_build_gaussian_field_from_density_rejects_non_positive_cell_size():
    with pytest.raises(ValueError):
        build_gaussian_field_from_density(
            np.zeros((4, 4, 4)), cell_size_m=0.0
        )


def test_build_gaussian_field_from_density_handles_all_zero():
    points = build_gaussian_field_from_density(np.zeros((4, 4, 4)))
    assert points == []


# --- GaussianSplatRenderer ---


def test_renderer_validates_dimensions():
    cam = _camera()
    with pytest.raises(ValueError):
        GaussianSplatRenderer(0, 64, cam)
    with pytest.raises(ValueError):
        GaussianSplatRenderer(64, -1, cam)


def test_renderer_returns_correct_shape_and_dtype():
    cam = _camera(64, 32)
    renderer = GaussianSplatRenderer(64, 32, cam)
    image = renderer.render([])
    assert image.shape == (32, 64, 3)
    assert image.dtype == np.uint8


def test_renderer_empty_input_returns_black_image():
    cam = _camera()
    renderer = GaussianSplatRenderer(64, 64, cam)
    image = renderer.render([])
    assert (image == 0).all()


def test_renderer_no_nans_in_output():
    cam = _camera(64, 64)
    points = [
        GaussianPoint(
            position=np.array([0.0, 1.0e16, 0.0]),
            color=np.array([255.0, 200.0, 100.0]),
            intensity=1.0,
            sigma=1.0e15,
        )
    ]
    renderer = GaussianSplatRenderer(64, 64, cam)
    image = renderer.render(points)
    assert np.isfinite(image).all()


def test_renderer_skips_points_behind_camera():
    cam = _camera(32, 32)
    behind = [
        GaussianPoint(
            position=np.array([0.0, -1.0e16, 0.0]),  # negative y == behind +y camera
            color=np.array([255.0, 255.0, 255.0]),
            intensity=10.0,
            sigma=1.0e15,
        )
    ]
    renderer = GaussianSplatRenderer(32, 32, cam)
    image = renderer.render(behind)
    assert (image == 0).all()


def test_renderer_lights_up_near_image_center_for_centered_point():
    cam = _camera(64, 64)
    # Point straight in front of the camera (along +y)
    points = [
        GaussianPoint(
            position=np.array([0.0, 1.0e16, 0.0]),
            color=np.array([255.0, 255.0, 255.0]),
            intensity=10.0,
            sigma=2.0e14,
        )
    ]
    renderer = GaussianSplatRenderer(64, 64, cam)
    image = renderer.render(points)
    # Brightest pixel should be near the image center
    flat = image.sum(axis=2)
    cy, cx = np.unravel_index(int(np.argmax(flat)), flat.shape)
    assert abs(cy - 32) <= 3
    assert abs(cx - 32) <= 3


def test_renderer_off_screen_point_does_not_light_up_image():
    cam = _camera(64, 64)
    # 90° off-axis from forward should not appear in a 120° FOV camera
    # at this distance and sigma.
    points = [
        GaussianPoint(
            position=np.array([1.0e16, 0.001, 0.0]),  # almost orthogonal to forward
            color=np.array([255.0, 255.0, 255.0]),
            intensity=10.0,
            sigma=1.0e10,  # very small footprint
        )
    ]
    renderer = GaussianSplatRenderer(64, 64, cam)
    image = renderer.render(points)
    assert (image == 0).all() or image.max() < 5


# --- AIViewerConfig render_mode ---


def test_config_default_render_mode_is_ppm():
    cfg = AIViewerConfig()
    assert cfg.render_mode == "ppm"
    cfg.validate()


def test_config_gaussian_render_mode_validates():
    AIViewerConfig(render_mode="gaussian").validate()


def test_config_invalid_render_mode_rejected():
    with pytest.raises(ValueError):
        AIViewerConfig(render_mode="vulkan").validate()


def test_config_gaussian_sigma_scale_must_be_positive():
    with pytest.raises(ValueError):
        AIViewerConfig(gaussian_sigma_scale=0.0).validate()
    with pytest.raises(ValueError):
        AIViewerConfig(gaussian_sigma_scale=-1.0).validate()


def test_config_max_gaussian_points_must_be_positive():
    with pytest.raises(ValueError):
        AIViewerConfig(max_gaussian_points=0).validate()


# --- AIViewer integration with render_mode='gaussian' ---


class _StubClient:
    def receive_message(self): return None
    def connect(self): pass
    def disconnect(self): pass


def test_ai_viewer_gaussian_mode_uses_passthrough_postprocessor(capsys):
    from ai_viewer import AIViewer
    from ai_viewer.postprocess import FramePostProcessor

    cfg = AIViewerConfig(
        enable_window=False,
        render_mode="gaussian",
        output_directory="outputs/viewer-test",
    )
    viewer = AIViewer(cfg, _StubClient())
    captured = capsys.readouterr().out
    assert "gaussian" in captured.lower()
    assert type(viewer.postprocessor) is FramePostProcessor


# --- end-to-end: small galaxy batch round trip ---


def test_renderer_smoke_test_with_real_galaxy_batch():
    galaxies = [
        _galaxy(f"g{i}", Vector3(float(i) * 1.0e21, 5.0e21, 0.0))
        for i in range(20)
    ]
    cam = _camera(48, 48)
    batch = build_galaxy_field_batch(galaxies, cam)
    points = build_gaussian_field_from_galaxy_batch(batch)
    renderer = GaussianSplatRenderer(48, 48, cam)
    image = renderer.render(points)
    assert image.shape == (48, 48, 3)
    assert np.isfinite(image).all()
    # at least some pixel should be lit
    assert image.max() > 0
