"""Tests for Phase 8 galaxy catalog loader and synthetic generator."""

from __future__ import annotations

from pathlib import Path

import pytest

from cosmic_engine.core.object_types import CosmicObjectType
from cosmic_engine.core.registry import UniverseRegistry
from cosmic_engine.core.truth import TruthLevel
from cosmic_engine.cosmos.galaxy import GalaxyProperties, create_galaxy_object
from cosmic_engine.core.vector import Vector3
from cosmic_engine.data.galaxy_catalog import (
    generate_synthetic_galaxy_catalog,
    load_galaxy_catalog,
    load_galaxy_catalog_into_registry,
)

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SAMPLE = _REPO_ROOT / "data" / "sample_galaxies.csv"


# --- create_galaxy_object ---


def test_create_galaxy_object_csv_source_yields_catalog_truth():
    obj = create_galaxy_object(
        "g1",
        "G1",
        Vector3(1.0, 2.0, 3.0),
        GalaxyProperties(redshift_z=0.01, apparent_magnitude=6.5),
        source="CSV",
    )
    assert obj.object_type is CosmicObjectType.GALAXY
    assert obj.truth_level is TruthLevel.CATALOG_IMPORTED
    assert obj.redshift_z == 0.01
    assert obj.velocity_m_s == Vector3.zero()
    assert obj.metadata["apparent_magnitude"] == 6.5


def test_create_galaxy_object_synthetic_source_yields_procedural_truth():
    obj = create_galaxy_object(
        "g2",
        "G2",
        Vector3.zero(),
        GalaxyProperties(stellar_mass_kg=1.0e40),
        source="synthetic",
    )
    assert obj.truth_level is TruthLevel.PROCEDURAL_APPROXIMATION
    assert obj.mass_kg == 1.0e40


def test_create_galaxy_object_synthetic_variant_still_procedural():
    obj = create_galaxy_object(
        "g3",
        "G3",
        Vector3.zero(),
        GalaxyProperties(),
        source="synthetic_desi_like",
    )
    assert obj.truth_level is TruthLevel.PROCEDURAL_APPROXIMATION


# --- CSV loader ---


def test_sample_galaxy_catalog_loads():
    objects = load_galaxy_catalog(str(_SAMPLE))
    assert len(objects) == 7
    by_id = {o.id: o for o in objects}
    andromeda = by_id["andromeda"]
    assert andromeda.object_type is CosmicObjectType.GALAXY
    assert andromeda.truth_level is TruthLevel.CATALOG_IMPORTED
    assert andromeda.redshift_z == pytest.approx(-0.001)
    assert andromeda.metadata["morphology"] == "spiral"
    assert andromeda.metadata["apparent_magnitude"] == pytest.approx(3.44)


def test_galaxy_catalog_skips_invalid_rows(tmp_path: Path):
    csv_path = tmp_path / "g.csv"
    csv_path.write_text(
        "id,name,x_m,y_m,z_m,redshift_z,apparent_magnitude,morphology\n"
        "g1,G1,1.0,2.0,3.0,0.01,6.5,spiral\n"
        "g2,G2,not_a_number,2.0,3.0,0.01,6.5,spiral\n"
        ",Nameless,1.0,2.0,3.0,0.01,6.5,spiral\n"
        "g3,,1.0,2.0,3.0,0.01,6.5,spiral\n"
        "g4,G4,1.0,2.0,3.0,,,\n"
    )
    objects = load_galaxy_catalog(str(csv_path))
    assert [o.id for o in objects] == ["g1", "g4"]
    # optional fields parse cleanly when missing
    g4 = objects[1]
    assert g4.redshift_z is None
    assert g4.metadata["apparent_magnitude"] is None
    assert g4.metadata["morphology"] is None


def test_galaxy_catalog_missing_required_column_raises(tmp_path: Path):
    csv_path = tmp_path / "bad.csv"
    # missing "morphology" column
    csv_path.write_text(
        "id,name,x_m,y_m,z_m,redshift_z,apparent_magnitude\n"
        "g1,G1,1.0,2.0,3.0,0.01,6.5\n"
    )
    with pytest.raises(ValueError):
        load_galaxy_catalog(str(csv_path))


def test_load_galaxy_catalog_into_registry_is_idempotent():
    registry = UniverseRegistry()
    load_galaxy_catalog_into_registry(str(_SAMPLE), registry)
    assert len(registry.list_objects()) == 7
    load_galaxy_catalog_into_registry(str(_SAMPLE), registry)  # duplicates
    assert len(registry.list_objects()) == 7


# --- synthetic generator ---


def test_synthetic_generator_is_deterministic_for_same_seed():
    a = generate_synthetic_galaxy_catalog(200, 1.0e24, seed=1234)
    b = generate_synthetic_galaxy_catalog(200, 1.0e24, seed=1234)
    assert len(a) == len(b) == 200
    for ga, gb in zip(a, b):
        assert ga.id == gb.id
        assert ga.position_m == gb.position_m
        assert ga.redshift_z == gb.redshift_z
        assert ga.metadata["apparent_magnitude"] == gb.metadata["apparent_magnitude"]


def test_synthetic_generator_different_seed_differs():
    a = generate_synthetic_galaxy_catalog(100, 1.0e24, seed=1)
    b = generate_synthetic_galaxy_catalog(100, 1.0e24, seed=2)
    positions_a = [g.position_m for g in a]
    positions_b = [g.position_m for g in b]
    assert positions_a != positions_b


def test_synthetic_generator_produces_galaxies_with_correct_truth():
    galaxies = generate_synthetic_galaxy_catalog(50, 1.0e24)
    assert all(g.object_type is CosmicObjectType.GALAXY for g in galaxies)
    assert all(g.truth_level is TruthLevel.PROCEDURAL_APPROXIMATION for g in galaxies)
    assert all(g.source == "synthetic_desi_like" for g in galaxies)


def test_synthetic_generator_respects_radius():
    radius = 1.0e24
    galaxies = generate_synthetic_galaxy_catalog(500, radius)
    for g in galaxies:
        d = (
            g.position_m.x ** 2 + g.position_m.y ** 2 + g.position_m.z ** 2
        ) ** 0.5
        assert d <= radius * (1.0 + 1e-9)  # relative float slack


def test_synthetic_generator_edge_cases():
    assert generate_synthetic_galaxy_catalog(0, 1.0e24) == []
    with pytest.raises(ValueError):
        generate_synthetic_galaxy_catalog(10, -1.0e24)
