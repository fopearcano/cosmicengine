"""Tests for Phase 7 NumPy-batched photon and perception pipeline."""

from __future__ import annotations

import math

import numpy as np
import pytest

from cosmic_engine.core.coordinates import ra_dec_distance_to_cartesian
from cosmic_engine.core.object_types import CosmicObjectType
from cosmic_engine.core.truth import TruthLevel
from cosmic_engine.core.units import (
    LIGHTYEAR_IN_METERS,
    SPEED_OF_LIGHT_M_S,
)
from cosmic_engine.core.universe_object import UniverseObject
from cosmic_engine.core.vector import Vector3
from cosmic_engine.core.vector_batch import (
    array_to_vectors,
    clamp_array,
    distances_from_origin,
    normalize_vectors,
    vectors_to_array,
)
from cosmic_engine.perception import (
    ObserverState,
    transform_photon_field,
    transform_photon_field_batch,
)
from cosmic_engine.rendering import (
    SimpleCamera,
    build_star_photon_field,
    build_star_photon_field_batch,
    photon_batch_to_samples,
)


FORWARD = Vector3(0.0, 1.0, 0.0)
UP = Vector3(0.0, 0.0, 1.0)


def _camera() -> SimpleCamera:
    return SimpleCamera(
        position_m=Vector3.zero(),
        forward=FORWARD,
        up=UP,
        fov_degrees=120.0,
        image_width=64,
        image_height=64,
    )


def _star(
    object_id: str,
    ra: float,
    dec: float,
    distance_ly: float,
    *,
    magnitude: float | None = 1.0,
    spectral_class: str = "G2V",
) -> UniverseObject:
    pos = ra_dec_distance_to_cartesian(
        ra, dec, distance_ly * LIGHTYEAR_IN_METERS
    )
    metadata = {"apparent_magnitude": magnitude} if magnitude is not None else {}
    return UniverseObject(
        id=object_id,
        name=object_id.title(),
        object_type=CosmicObjectType.STAR,
        position_m=pos,
        velocity_m_s=Vector3.zero(),
        truth_level=TruthLevel.CATALOG_IMPORTED,
        source="test",
        spectral_class=spectral_class,
        metadata=metadata,
    )


# --- vector_batch ---


def test_vectors_to_array_shape_and_content():
    vs = [Vector3(1.0, 2.0, 3.0), Vector3(-4.0, 0.0, 0.5)]
    arr = vectors_to_array(vs)
    assert arr.shape == (2, 3)
    assert arr.dtype == np.float64
    np.testing.assert_array_equal(arr, [[1.0, 2.0, 3.0], [-4.0, 0.0, 0.5]])


def test_vectors_to_array_empty():
    arr = vectors_to_array([])
    assert arr.shape == (0, 3)


def test_array_to_vectors_round_trip():
    vs = [Vector3(1.0, 2.0, 3.0), Vector3(4.0, 5.0, 6.0)]
    assert array_to_vectors(vectors_to_array(vs)) == vs


def test_normalize_vectors_unit_length_and_zero_passthrough():
    arr = np.array([[3.0, 0.0, 4.0], [0.0, 0.0, 0.0], [1.0, 2.0, 2.0]])
    out = normalize_vectors(arr)
    np.testing.assert_allclose(np.linalg.norm(out[[0, 2]], axis=1), [1.0, 1.0])
    np.testing.assert_array_equal(out[1], [0.0, 0.0, 0.0])


def test_distances_from_origin():
    arr = np.array([[3.0, 4.0, 0.0], [1.0, 2.0, 2.0]])
    np.testing.assert_allclose(distances_from_origin(arr), [5.0, 3.0])


def test_clamp_array():
    arr = np.array([-10.0, 0.5, 300.0])
    np.testing.assert_array_equal(clamp_array(arr, 0.0, 255.0), [0.0, 0.5, 255.0])


def test_vector_batch_rejects_wrong_shape():
    with pytest.raises(ValueError):
        normalize_vectors(np.zeros((5,)))
    with pytest.raises(ValueError):
        distances_from_origin(np.zeros((5, 4)))


# --- batch builder ---


def test_batch_builder_handles_empty_list():
    batch = build_star_photon_field_batch([], _camera())
    assert len(batch) == 0
    assert batch.directions.shape == (0, 3)
    assert batch.distances_m.shape == (0,)
    assert batch.brightness.shape == (0,)
    assert batch.colors_rgb.shape == (0, 3)


def test_batch_builder_filters_non_stars():
    star = _star("s1", 45.0, 10.0, 10.0)
    galaxy = UniverseObject(
        id="g",
        name="G",
        object_type=CosmicObjectType.GALAXY,
        position_m=Vector3(1.0, 1.0, 1.0),
        velocity_m_s=Vector3.zero(),
        truth_level=TruthLevel.CATALOG_IMPORTED,
    )
    batch = build_star_photon_field_batch([star, galaxy], _camera())
    assert batch.object_ids == ["s1"]


def test_batch_directions_are_unit_vectors():
    stars = [
        _star("s1", 45.0, 10.0, 10.0),
        _star("s2", 200.0, -30.0, 500.0),
        _star("s3", 170.0, 60.0, 50.0),
    ]
    batch = build_star_photon_field_batch(stars, _camera())
    mags = np.linalg.norm(batch.directions, axis=1)
    np.testing.assert_allclose(mags, 1.0, atol=1e-12)


def test_batch_brightness_positive_and_colors_bounded():
    stars = [
        _star("dim", 10.0, 10.0, 100.0, magnitude=10.0, spectral_class="M1Ia"),
        _star("bright", 200.0, -30.0, 8.0, magnitude=-1.5, spectral_class="A1V"),
    ]
    batch = build_star_photon_field_batch(stars, _camera())
    assert (batch.brightness > 0).all()
    assert ((batch.colors_rgb >= 0) & (batch.colors_rgb <= 255)).all()


def test_batch_missing_magnitude_defaults_to_one():
    star = _star("nomag", 10.0, 10.0, 100.0, magnitude=None)
    batch = build_star_photon_field_batch([star], _camera())
    assert batch.brightness.tolist() == [1.0]


# --- batch <-> samples round trip ---


def test_photon_batch_to_samples_length_matches():
    stars = [_star(f"s{i}", 30.0 * i, 10.0, 20.0) for i in range(5)]
    batch = build_star_photon_field_batch(stars, _camera())
    samples = photon_batch_to_samples(batch)
    assert len(samples) == len(batch) == 5


def test_batch_matches_scalar_builder():
    stars = [
        _star("a", 45.0, 10.0, 20.0, magnitude=2.0, spectral_class="G2V"),
        _star("b", 120.0, -20.0, 50.0, magnitude=-1.0, spectral_class="M1Ia"),
        _star("c", 300.0, 70.0, 5.0, magnitude=4.5, spectral_class="A1V"),
    ]
    cam = _camera()
    scalar = build_star_photon_field(stars, cam)
    batch = build_star_photon_field_batch(stars, cam)
    via_batch = photon_batch_to_samples(batch)
    # same ordering, same identity fields, same numeric fields within tol
    assert [s.object_id for s in via_batch] == [s.object_id for s in scalar]
    for a, b in zip(scalar, via_batch):
        assert a.object_type == b.object_type
        assert a.truth_level == b.truth_level
        assert a.color_rgb == b.color_rgb
        assert a.apparent_brightness == pytest.approx(
            b.apparent_brightness, rel=1e-12
        )
        for ca, cb in zip(
            (a.direction.x, a.direction.y, a.direction.z),
            (b.direction.x, b.direction.y, b.direction.z),
        ):
            assert ca == pytest.approx(cb, abs=1e-12)


# --- batch transform ---


def _observer(*, warp: float = 5.0) -> ObserverState:
    return ObserverState(
        position_m=Vector3.zero(),
        velocity_m_s=Vector3(0.0, 0.5 * SPEED_OF_LIGHT_M_S, 0.0),
        forward=FORWARD,
        up=UP,
        warp_factor=warp,
    )


def test_batch_transform_preserves_count():
    stars = [_star(f"s{i}", 30.0 * i, 5.0, 20.0) for i in range(10)]
    batch = build_star_photon_field_batch(stars, _camera())
    out = transform_photon_field_batch(batch, _observer())
    assert len(out) == len(batch)


def test_batch_transform_outputs_are_valid():
    stars = [
        _star("a", 45.0, 10.0, 20.0, magnitude=2.0),
        _star("b", 170.0, -30.0, 50.0, magnitude=-1.0),
    ]
    batch = build_star_photon_field_batch(stars, _camera())
    out = transform_photon_field_batch(batch, _observer(warp=50.0))
    mags = np.linalg.norm(out.directions, axis=1)
    np.testing.assert_allclose(mags, 1.0, atol=1e-12)
    assert (out.brightness >= 0).all()
    assert np.isfinite(out.brightness).all()
    assert ((out.colors_rgb >= 0) & (out.colors_rgb <= 255)).all()


def test_batch_transform_empty_batch_is_noop():
    empty = build_star_photon_field_batch([], _camera())
    out = transform_photon_field_batch(empty, _observer())
    assert len(out) == 0


def test_batch_transform_matches_scalar_transform():
    stars = [
        _star("a", 45.0, 10.0, 20.0, magnitude=2.0, spectral_class="G2V"),
        _star("b", 120.0, -20.0, 50.0, magnitude=-1.0, spectral_class="M1Ia"),
        _star("c", 80.0, 5.0, 100.0, magnitude=3.0, spectral_class="A1V"),
    ]
    cam = _camera()
    obs = _observer(warp=3.0)

    scalar_out = transform_photon_field(
        build_star_photon_field(stars, cam), obs
    )
    batch_out_samples = photon_batch_to_samples(
        transform_photon_field_batch(
            build_star_photon_field_batch(stars, cam), obs
        )
    )
    for a, b in zip(scalar_out, batch_out_samples):
        assert a.object_id == b.object_id
        assert a.apparent_brightness == pytest.approx(
            b.apparent_brightness, rel=1e-9
        )
        for ca, cb in zip(
            (a.direction.x, a.direction.y, a.direction.z),
            (b.direction.x, b.direction.y, b.direction.z),
        ):
            assert ca == pytest.approx(cb, abs=1e-10)
        # color integers must match within 1 (scalar rounds once; batch rounds once)
        for ca, cb in zip(a.color_rgb, b.color_rgb):
            assert abs(ca - cb) <= 1
