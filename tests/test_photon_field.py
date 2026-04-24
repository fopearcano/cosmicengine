"""Tests for Phase 3 photon-field and PPM export."""

from __future__ import annotations

from pathlib import Path

import pytest

from cosmic_engine.core.object_types import CosmicObjectType
from cosmic_engine.core.truth import TruthLevel
from cosmic_engine.core.universe_object import UniverseObject
from cosmic_engine.core.vector import Vector3
from cosmic_engine.rendering.photon_field import (
    PhotonSample,
    build_star_photon_field,
)
from cosmic_engine.rendering.image_export import render_photon_field_to_ppm
from cosmic_engine.rendering.simple_camera import SimpleCamera


def _star(
    object_id: str,
    position: Vector3,
    *,
    spectral_class: str = "G2V",
    apparent_magnitude: float | None = 1.0,
) -> UniverseObject:
    metadata = (
        {"apparent_magnitude": apparent_magnitude}
        if apparent_magnitude is not None
        else {}
    )
    return UniverseObject(
        id=object_id,
        name=object_id.title(),
        object_type=CosmicObjectType.STAR,
        position_m=position,
        velocity_m_s=Vector3.zero(),
        truth_level=TruthLevel.CATALOG_IMPORTED,
        source="test",
        spectral_class=spectral_class,
        metadata=metadata,
    )


def _galaxy(object_id: str, position: Vector3) -> UniverseObject:
    return UniverseObject(
        id=object_id,
        name=object_id,
        object_type=CosmicObjectType.GALAXY,
        position_m=position,
        velocity_m_s=Vector3.zero(),
        truth_level=TruthLevel.CATALOG_IMPORTED,
    )


def _camera() -> SimpleCamera:
    return SimpleCamera(
        position_m=Vector3.zero(),
        forward=Vector3(0.0, 1.0, 0.0),
        up=Vector3(0.0, 0.0, 1.0),
        fov_degrees=90.0,
        image_width=64,
        image_height=64,
    )


# --- camera validation ---


def test_camera_validate_accepts_sane_defaults():
    _camera().validate()


@pytest.mark.parametrize(
    "overrides",
    [
        {"fov_degrees": 0.0},
        {"fov_degrees": 180.0},
        {"fov_degrees": -10.0},
        {"image_width": 0},
        {"image_height": -1},
        {"forward": Vector3.zero()},
        {"up": Vector3.zero()},
    ],
)
def test_camera_validate_rejects_bad_fields(overrides):
    cam = _camera()
    for k, v in overrides.items():
        setattr(cam, k, v)
    with pytest.raises(ValueError):
        cam.validate()


# --- photon field ---


def test_photon_field_only_includes_stars():
    objs = [
        _star("a", Vector3(0.0, 1e16, 0.0)),
        _galaxy("g", Vector3(0.0, 2e16, 0.0)),
        _star("b", Vector3(1e15, 2e16, 0.0)),
    ]
    samples = build_star_photon_field(objs, _camera())
    assert [s.object_id for s in samples] == ["a", "b"]
    assert all(s.object_type == "star" for s in samples)


def test_photon_field_brightness_is_positive():
    samples = build_star_photon_field(
        [
            _star("bright", Vector3(0.0, 1e16, 0.0), apparent_magnitude=-1.0),
            _star("dim", Vector3(0.0, 2e16, 0.0), apparent_magnitude=10.0),
            _star("nomag", Vector3(0.0, 3e16, 0.0), apparent_magnitude=None),
        ],
        _camera(),
    )
    assert len(samples) == 3
    for s in samples:
        assert s.apparent_brightness > 0.0
    by_id = {s.object_id: s for s in samples}
    assert by_id["bright"].apparent_brightness > by_id["dim"].apparent_brightness


@pytest.mark.parametrize(
    "spectral_class, expected",
    [
        ("O5V", (150, 180, 255)),
        ("B0Ia", (170, 200, 255)),
        ("A1V", (240, 240, 255)),
        ("F5IV", (255, 250, 220)),
        ("G2V", (255, 240, 180)),
        ("K0III", (255, 200, 140)),
        ("M1Ia", (255, 140, 100)),
        ("XYZ", (255, 255, 255)),
        (None, (255, 255, 255)),
        ("", (255, 255, 255)),
    ],
)
def test_spectral_class_maps_to_rgb(spectral_class, expected):
    star = _star("s", Vector3(0.0, 1e16, 0.0), spectral_class=spectral_class or "")
    star.spectral_class = spectral_class
    samples = build_star_photon_field([star], _camera())
    assert samples[0].color_rgb == expected


def test_photon_field_skips_star_coincident_with_camera():
    # star exactly at camera position -> zero-length direction, skipped
    objs = [_star("here", Vector3.zero())]
    assert build_star_photon_field(objs, _camera()) == []


def test_photon_field_preserves_truth_level():
    objs = [_star("a", Vector3(0.0, 1e16, 0.0))]
    samples = build_star_photon_field(objs, _camera())
    assert samples[0].truth_level == TruthLevel.CATALOG_IMPORTED.value


# --- PPM export ---


def _read_ppm_header(path: Path) -> tuple[str, int, int, int]:
    text = path.read_text()
    lines = text.splitlines()
    magic = lines[0]
    w, h = (int(x) for x in lines[1].split())
    maxval = int(lines[2])
    return magic, w, h, maxval


def test_ppm_empty_samples_still_produces_black_image(tmp_path: Path):
    out = tmp_path / "empty.ppm"
    render_photon_field_to_ppm([], _camera(), str(out))
    assert out.exists()
    magic, w, h, _ = _read_ppm_header(out)
    assert magic == "P3"
    assert (w, h) == (64, 64)
    # body should contain no non-zero pixel values
    body = " ".join(out.read_text().splitlines()[3:]).split()
    assert all(v == "0" for v in body)


def test_ppm_renders_a_pixel_for_a_visible_star(tmp_path: Path):
    out = tmp_path / "one.ppm"
    samples = build_star_photon_field(
        [_star("a", Vector3(0.0, 1e16, 0.0), apparent_magnitude=0.0)],
        _camera(),
    )
    assert len(samples) == 1
    render_photon_field_to_ppm(samples, _camera(), str(out))
    body = " ".join(out.read_text().splitlines()[3:]).split()
    assert any(v != "0" for v in body)


def test_ppm_file_is_created_for_sample_set(tmp_path: Path):
    out = tmp_path / "scene.ppm"
    samples = [
        PhotonSample(
            object_id="a",
            name="A",
            object_type="star",
            direction=Vector3(0.0, 1.0, 0.0),
            distance_m=1e16,
            apparent_brightness=1.0,
            color_rgb=(255, 240, 180),
            truth_level="catalog_imported",
        )
    ]
    render_photon_field_to_ppm(samples, _camera(), str(out))
    assert out.is_file()
    magic, w, h, maxval = _read_ppm_header(out)
    assert magic == "P3"
    assert (w, h) == (64, 64)
    assert maxval == 255
