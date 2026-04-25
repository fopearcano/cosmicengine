"""Tests for Phase 12 multi-source ingestion."""

from __future__ import annotations

from pathlib import Path

import pytest

from cosmic_engine.core.object_types import CosmicObjectType
from cosmic_engine.core.registry import UniverseRegistry
from cosmic_engine.core.truth import TruthLevel
from cosmic_engine.core.vector import Vector3
from cosmic_engine.data.desi_catalog import load_desi_catalog
from cosmic_engine.data.gaia_catalog import load_gaia_like_catalog
from cosmic_engine.data.jpl_ephemeris import (
    load_jpl_ephemeris_placeholder,
    load_jpl_into_registry,
)
from cosmic_engine.data.sdss_catalog import load_sdss_like_catalog
from cosmic_engine.data.sources import (
    DataSource,
    load_catalog,
    load_catalog_into_registry,
    tag_source,
)


_REPO_ROOT = Path(__file__).resolve().parent.parent
_GAIA = _REPO_ROOT / "data" / "sample_gaia_like.csv"
_SDSS = _REPO_ROOT / "data" / "sample_sdss_like.csv"
_DESI = _REPO_ROOT / "data" / "sample_desi.csv"


# --- DataSource enum + tag_source ---


def test_data_source_values():
    assert DataSource.GAIA.value == "gaia"
    assert DataSource.SDSS.value == "sdss"
    assert DataSource.DESI.value == "desi"
    assert DataSource.JPL.value == "jpl"
    assert DataSource.NASA.value == "nasa"
    assert DataSource.SYNTHETIC.value == "synthetic"


def test_tag_source_sets_field_and_metadata():
    obj = load_jpl_ephemeris_placeholder()[0]
    obj.source = "previous"
    obj.metadata.pop("data_source", None)
    tag_source(obj, DataSource.NASA)
    assert obj.source == "nasa"
    assert obj.metadata["data_source"] == "nasa"


# --- Gaia loader ---


def test_gaia_loader_creates_stars():
    objects = load_gaia_like_catalog(str(_GAIA))
    assert len(objects) == 10
    for obj in objects:
        assert obj.object_type is CosmicObjectType.STAR
        assert obj.truth_level is TruthLevel.CATALOG_IMPORTED
        assert obj.source == "gaia"
        assert obj.metadata["data_source"] == "gaia"
        assert obj.position_m != Vector3.zero()
        assert obj.metadata["apparent_magnitude"] is not None
        assert obj.metadata["parallax_mas"] > 0.0


def test_gaia_parallax_to_distance_is_positive():
    objects = load_gaia_like_catalog(str(_GAIA))
    for obj in objects:
        d = (
            obj.position_m.x ** 2
            + obj.position_m.y ** 2
            + obj.position_m.z ** 2
        ) ** 0.5
        assert d > 0.0


def test_gaia_skips_non_positive_parallax(tmp_path: Path):
    csv_path = tmp_path / "gaia.csv"
    csv_path.write_text(
        "source_id,ra,dec,parallax_mas,phot_g_mean_mag,bp_rp\n"
        "good,10.0,5.0,100.0,5.0,0.5\n"
        "negative,10.0,5.0,-1.5,5.0,0.5\n"
        "zero,10.0,5.0,0.0,5.0,0.5\n"
        "bad_num,10.0,5.0,xxx,5.0,0.5\n"
        "missing_id,,5.0,100.0,5.0,0.5\n"
        "good2,30.0,10.0,50.0,4.0,0.4\n"
    )
    objects = load_gaia_like_catalog(str(csv_path))
    assert [o.id for o in objects] == ["good", "good2"]


def test_gaia_missing_required_column_raises(tmp_path: Path):
    csv_path = tmp_path / "gaia.csv"
    csv_path.write_text(
        "source_id,ra,dec,phot_g_mean_mag\n"  # no parallax_mas
        "g1,10.0,5.0,5.0\n"
    )
    with pytest.raises(ValueError):
        load_gaia_like_catalog(str(csv_path))


# --- SDSS loader ---


def test_sdss_loader_creates_galaxies():
    objects = load_sdss_like_catalog(str(_SDSS))
    assert len(objects) == 10
    for obj in objects:
        assert obj.object_type is CosmicObjectType.GALAXY
        assert obj.truth_level is TruthLevel.CATALOG_IMPORTED
        assert obj.source == "sdss"
        assert obj.metadata["data_source"] == "sdss"
        assert obj.position_m != Vector3.zero()
        assert obj.redshift_z is not None and obj.redshift_z > 0


def test_sdss_loader_records_distance_model():
    objects = load_sdss_like_catalog(str(_SDSS))
    assert all("distance_model" in o.metadata for o in objects)


def test_sdss_skips_negative_z(tmp_path: Path):
    csv_path = tmp_path / "sdss.csv"
    csv_path.write_text(
        "objid,ra,dec,z,modelMag_r\n"
        "good,180.0,5.0,0.1,18.0\n"
        "neg_z,180.0,5.0,-0.05,18.0\n"
        "bad_num,180.0,5.0,zzz,18.0\n"
        ",180.0,5.0,0.1,18.0\n"
        "good2,200.0,5.0,0.2,18.5\n"
    )
    assert [o.id for o in load_sdss_like_catalog(str(csv_path))] == [
        "good",
        "good2",
    ]


# --- DESI loader (new schema) ---


def test_desi_loader_creates_galaxies():
    objects = load_desi_catalog(str(_DESI))
    assert len(objects) == 10
    for obj in objects:
        assert obj.object_type is CosmicObjectType.GALAXY
        assert obj.source == "desi"
        assert obj.metadata["data_source"] == "desi"
        assert obj.position_m != Vector3.zero()


def test_desi_optional_mag_can_be_missing(tmp_path: Path):
    csv_path = tmp_path / "desi.csv"
    csv_path.write_text(
        "targetid,ra,dec,z,mag\n"
        "t1,150.0,2.0,0.1,\n"
    )
    obj = load_desi_catalog(str(csv_path))[0]
    assert obj.metadata["apparent_magnitude"] is None
    assert obj.metadata["absolute_magnitude"] is None


# --- JPL placeholder ---


def test_jpl_placeholder_includes_sun_and_eight_planets():
    objects = load_jpl_ephemeris_placeholder()
    by_id = {o.id: o for o in objects}
    assert set(by_id) == {
        "sun",
        "mercury",
        "venus",
        "earth",
        "mars",
        "jupiter",
        "saturn",
        "uranus",
        "neptune",
    }
    assert by_id["sun"].object_type is CosmicObjectType.STAR
    for planet in ("mercury", "venus", "earth", "mars", "jupiter",
                   "saturn", "uranus", "neptune"):
        assert by_id[planet].object_type is CosmicObjectType.PLANET
    for obj in objects:
        assert obj.truth_level is TruthLevel.PHYSICS_SIMULATED
        assert obj.source == "jpl"
        assert obj.metadata["model"] == "simplified_keplerian"


def test_jpl_into_registry_idempotent():
    registry = UniverseRegistry()
    load_jpl_into_registry(registry)
    assert len(registry.list_objects()) == 9
    load_jpl_into_registry(registry)  # second call should not duplicate
    assert len(registry.list_objects()) == 9


# --- unified dispatch ---


def test_load_catalog_dispatches_correctly():
    gaia = load_catalog(str(_GAIA), DataSource.GAIA)
    sdss = load_catalog(str(_SDSS), DataSource.SDSS)
    desi = load_catalog(str(_DESI), DataSource.DESI)
    assert all(o.object_type is CosmicObjectType.STAR for o in gaia)
    assert all(o.object_type is CosmicObjectType.GALAXY for o in sdss)
    assert all(o.object_type is CosmicObjectType.GALAXY for o in desi)


def test_load_catalog_rejects_unsupported_source():
    with pytest.raises(ValueError):
        load_catalog(str(_GAIA), DataSource.JPL)
    with pytest.raises(ValueError):
        load_catalog(str(_GAIA), DataSource.SYNTHETIC)


# --- registry integration ---


def test_load_catalog_into_registry_handles_all_sources():
    registry = UniverseRegistry()
    load_catalog_into_registry(str(_GAIA), DataSource.GAIA, registry)
    load_catalog_into_registry(str(_SDSS), DataSource.SDSS, registry)
    load_catalog_into_registry(str(_DESI), DataSource.DESI, registry)
    load_jpl_into_registry(registry)
    objects = registry.list_objects()
    # 10 + 10 + 10 + 9 = 39 (Gaia + SDSS + DESI + JPL Sun+8 planets)
    assert len(objects) == 39


def test_mixed_catalogs_do_not_collide_ids():
    registry = UniverseRegistry()
    # Loading all four sources twice should never raise even though IDs
    # repeat across sources only by accident; idempotent on second run.
    for _ in range(2):
        load_catalog_into_registry(str(_GAIA), DataSource.GAIA, registry)
        load_catalog_into_registry(str(_SDSS), DataSource.SDSS, registry)
        load_catalog_into_registry(str(_DESI), DataSource.DESI, registry)
        load_jpl_into_registry(registry)
    assert len(registry.list_objects()) == 39
