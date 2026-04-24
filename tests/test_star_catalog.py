"""Tests for Phase 2 CSV star-catalog ingestion."""

from __future__ import annotations

from pathlib import Path

import pytest

from cosmic_engine.core.object_types import CosmicObjectType
from cosmic_engine.core.registry import UniverseRegistry
from cosmic_engine.core.truth import TruthLevel
from cosmic_engine.core.vector import Vector3
from cosmic_engine.data.catalog_loader import load_csv
from cosmic_engine.data.star_catalog import (
    load_star_catalog,
    load_star_catalog_into_registry,
)

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SAMPLE_CSV = _REPO_ROOT / "data" / "sample_stars.csv"


# --- catalog_loader ---


def test_load_csv_strips_and_skips_blank_rows(tmp_path: Path):
    csv_path = tmp_path / "t.csv"
    csv_path.write_text(
        "a,b ,c\n"
        " 1, 2 , 3 \n"
        "\n"
        "   ,   ,   \n"
        "4,5,6\n"
    )
    rows = load_csv(str(csv_path))
    assert rows == [
        {"a": "1", "b": "2", "c": "3"},
        {"a": "4", "b": "5", "c": "6"},
    ]


def test_load_csv_missing_file(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_csv(str(tmp_path / "nope.csv"))


def test_load_csv_empty_file(tmp_path: Path):
    csv_path = tmp_path / "empty.csv"
    csv_path.write_text("")
    with pytest.raises(ValueError):
        load_csv(str(csv_path))


# --- star_catalog on bundled sample ---


def test_sample_catalog_loads_expected_count():
    objects = load_star_catalog(str(_SAMPLE_CSV))
    # 8 rows in data/sample_stars.csv
    assert len(objects) == 8


def test_sample_catalog_sun_is_at_origin():
    by_id = {o.id: o for o in load_star_catalog(str(_SAMPLE_CSV))}
    sol = by_id["sol"]
    assert sol.position_m == Vector3.zero()
    assert sol.spectral_class == "G2V"
    assert sol.metadata["apparent_magnitude"] == pytest.approx(-26.74)


def test_sample_catalog_non_sun_stars_have_nonzero_position():
    for obj in load_star_catalog(str(_SAMPLE_CSV)):
        if obj.id == "sol":
            continue
        assert obj.position_m != Vector3.zero()


def test_sample_catalog_truth_level_and_type():
    for obj in load_star_catalog(str(_SAMPLE_CSV)):
        assert obj.object_type is CosmicObjectType.STAR
        assert obj.truth_level is TruthLevel.CATALOG_IMPORTED
        assert obj.source == "CSV"
        assert obj.velocity_m_s == Vector3.zero()
        assert obj.mass_kg is None
        assert obj.radius_m is None
        assert obj.luminosity_w is None
        assert obj.redshift_z is None


# --- invalid rows ---


def _write_csv(path: Path, rows: list[str]) -> None:
    header = "id,name,ra_deg,dec_deg,distance_ly,apparent_magnitude,spectral_class\n"
    path.write_text(header + "\n".join(rows) + "\n")


def test_invalid_rows_are_skipped(tmp_path: Path):
    csv_path = tmp_path / "stars.csv"
    _write_csv(
        csv_path,
        [
            "good,Good Star,10,20,30,5.0,G2V",
            "bad_num,Bad RA,not_a_number,20,30,5.0,G2V",
            "bad_dist,Bad Distance,10,20,-1,5.0,G2V",
            "no_id,,10,20,30,5.0,G2V",
            ",Nameless,10,20,30,5.0,G2V",
            "no_spec,Spec Missing,10,20,30,5.0,",
            "good2,Another Good,50,30,10,3.0,A0V",
        ],
    )
    objects = load_star_catalog(str(csv_path))
    assert [o.id for o in objects] == ["good", "good2"]


def test_missing_required_column_raises(tmp_path: Path):
    csv_path = tmp_path / "bad.csv"
    # no spectral_class column
    csv_path.write_text(
        "id,name,ra_deg,dec_deg,distance_ly,apparent_magnitude\n"
        "x,X,0,0,1,2.0\n"
    )
    with pytest.raises(ValueError):
        load_star_catalog(str(csv_path))


# --- registry integration ---


def test_load_star_catalog_into_registry_adds_all():
    registry = UniverseRegistry()
    load_star_catalog_into_registry(str(_SAMPLE_CSV), registry)
    assert len(registry.list_objects()) == 8
    assert registry.get_object("sirius") is not None


def test_load_star_catalog_into_registry_skips_duplicates():
    registry = UniverseRegistry()
    load_star_catalog_into_registry(str(_SAMPLE_CSV), registry)
    # second load should not raise and should not add anything new
    load_star_catalog_into_registry(str(_SAMPLE_CSV), registry)
    assert len(registry.list_objects()) == 8
