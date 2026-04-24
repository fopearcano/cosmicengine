"""Tests for Phase 8 galaxy field batch and density grid."""

from __future__ import annotations

import math

import numpy as np
import pytest

from cosmic_engine.core.object_types import CosmicObjectType
from cosmic_engine.core.truth import TruthLevel
from cosmic_engine.core.universe_object import UniverseObject
from cosmic_engine.core.vector import Vector3
from cosmic_engine.cosmos.galaxy import GalaxyProperties, create_galaxy_object
from cosmic_engine.data.galaxy_catalog import generate_synthetic_galaxy_catalog
from cosmic_engine.rendering import (
    SimpleCamera,
    build_galaxy_field_batch,
    galaxy_batch_to_density_grid,
)


def _camera() -> SimpleCamera:
    return SimpleCamera(
        position_m=Vector3.zero(),
        forward=Vector3(0.0, 1.0, 0.0),
        up=Vector3(0.0, 0.0, 1.0),
        fov_degrees=120.0,
        image_width=64,
        image_height=64,
    )


def _galaxy(object_id: str, position: Vector3, *, z: float | None = 0.1, mag: float | None = 6.5):
    return create_galaxy_object(
        id=object_id,
        name=object_id,
        position_m=position,
        properties=GalaxyProperties(redshift_z=z, apparent_magnitude=mag),
        source="CSV",
    )


def _star(position: Vector3) -> UniverseObject:
    return UniverseObject(
        id="not_a_galaxy",
        name="Star",
        object_type=CosmicObjectType.STAR,
        position_m=position,
        velocity_m_s=Vector3.zero(),
        truth_level=TruthLevel.CATALOG_IMPORTED,
    )


# --- GalaxyFieldBatch ---


def test_galaxy_batch_handles_empty_input():
    batch = build_galaxy_field_batch([], _camera())
    assert len(batch) == 0
    assert batch.positions_m.shape == (0, 3)
    assert batch.directions.shape == (0, 3)
    assert batch.distances_m.shape == (0,)
    assert batch.brightness.shape == (0,)
    assert batch.redshifts_z.shape == (0,)
    assert batch.colors_rgb.shape == (0, 3)


def test_galaxy_batch_shapes_for_three_galaxies():
    galaxies = [
        _galaxy("g1", Vector3(1e22, 2e22, 3e22), z=0.01),
        _galaxy("g2", Vector3(-2e22, 1e22, 5e22), z=0.5),
        _galaxy("g3", Vector3(4e22, -3e22, 1e22), z=1.2),
    ]
    batch = build_galaxy_field_batch(galaxies, _camera())
    assert len(batch) == 3
    assert batch.positions_m.shape == (3, 3)
    assert batch.directions.shape == (3, 3)
    assert batch.distances_m.shape == (3,)
    assert batch.brightness.shape == (3,)
    assert batch.redshifts_z.shape == (3,)
    assert batch.colors_rgb.shape == (3, 3)


def test_galaxy_batch_filters_non_galaxies():
    galaxies = [
        _galaxy("g1", Vector3(1e22, 2e22, 3e22)),
        _galaxy("g2", Vector3(-2e22, 1e22, 5e22)),
    ]
    objects = list(galaxies) + [_star(Vector3(1.0, 1.0, 1.0))]
    batch = build_galaxy_field_batch(objects, _camera())
    assert batch.object_ids == ["g1", "g2"]


def test_galaxy_batch_directions_are_unit_vectors():
    galaxies = [
        _galaxy("g1", Vector3(3e22, 4e22, 0.0)),
        _galaxy("g2", Vector3(-1e22, 2e22, 2e22)),
    ]
    batch = build_galaxy_field_batch(galaxies, _camera())
    mags = np.linalg.norm(batch.directions, axis=1)
    np.testing.assert_allclose(mags, 1.0, atol=1e-12)


def test_galaxy_batch_brightness_is_positive():
    galaxies = [
        _galaxy("g1", Vector3(1e22, 2e22, 3e22), mag=6.0),
        _galaxy("g2", Vector3(1e23, 2e23, 3e23), mag=None),  # inverse-square fallback
    ]
    batch = build_galaxy_field_batch(galaxies, _camera())
    assert (batch.brightness > 0).all()
    assert np.isfinite(batch.brightness).all()


@pytest.mark.parametrize("z", [-0.5, 0.0, 0.1, 0.5, 1.0, 2.5, float("nan")])
def test_galaxy_batch_color_bounds_across_redshifts(z):
    galaxies = [_galaxy("g", Vector3(1e22, 1e22, 1e22), z=z)]
    batch = build_galaxy_field_batch(galaxies, _camera())
    colors = batch.colors_rgb[0]
    assert ((colors >= 0) & (colors <= 255)).all()


def test_galaxy_batch_color_redshifts_shift_toward_red():
    low = _galaxy("low", Vector3(1e22, 1e22, 1e22), z=0.0)
    high = _galaxy("high", Vector3(1e22, 1e22, 1e22), z=1.0)
    batch = build_galaxy_field_batch([low, high], _camera())
    # green and blue channels drop as z rises; red stays saturated
    assert batch.colors_rgb[1, 1] < batch.colors_rgb[0, 1]
    assert batch.colors_rgb[1, 2] < batch.colors_rgb[0, 2]
    assert batch.colors_rgb[1, 0] == pytest.approx(batch.colors_rgb[0, 0])


# --- density grid ---


def test_density_grid_shape_and_normalization():
    galaxies = [
        _galaxy("g1", Vector3(0.0, 0.0, 0.0)),
        _galaxy("g2", Vector3(1e22, 1e22, 1e22)),
    ]
    # include a star to verify it is ignored by the batch builder
    objects = list(galaxies) + [_star(Vector3(5.0, 5.0, 5.0))]
    batch = build_galaxy_field_batch(objects, _camera())
    grid = galaxy_batch_to_density_grid(batch, grid_size=8, extent_m=5.0e22)
    assert grid.shape == (8, 8, 8)
    assert grid.min() >= 0.0
    assert grid.max() <= 1.0
    assert grid.max() == pytest.approx(1.0)


def test_density_grid_empty_batch_returns_zeros():
    batch = build_galaxy_field_batch([], _camera())
    grid = galaxy_batch_to_density_grid(batch, grid_size=4, extent_m=1.0e22)
    assert grid.shape == (4, 4, 4)
    assert grid.sum() == 0.0


def test_density_grid_ignores_out_of_extent_galaxies():
    galaxies = [
        _galaxy("in", Vector3(1e22, 1e22, 1e22)),
        _galaxy("out", Vector3(5e23, 5e23, 5e23)),  # far outside
    ]
    batch = build_galaxy_field_batch(galaxies, _camera())
    grid = galaxy_batch_to_density_grid(batch, grid_size=4, extent_m=2.0e22)
    assert (grid > 0).sum() == 1  # only the inside galaxy counted


@pytest.mark.parametrize("grid_size, extent_m", [(0, 1.0), (-3, 1.0), (8, 0.0), (8, -1.0)])
def test_density_grid_validates_inputs(grid_size, extent_m):
    batch = build_galaxy_field_batch([], _camera())
    with pytest.raises(ValueError):
        galaxy_batch_to_density_grid(batch, grid_size=grid_size, extent_m=extent_m)


def test_density_grid_on_synthetic_catalog():
    galaxies = generate_synthetic_galaxy_catalog(2000, 1.0e24, seed=7)
    batch = build_galaxy_field_batch(galaxies, _camera())
    grid = galaxy_batch_to_density_grid(batch, grid_size=16, extent_m=1.2e24)
    assert grid.shape == (16, 16, 16)
    assert grid.max() == pytest.approx(1.0)
    assert (grid > 0).any()
