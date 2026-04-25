"""Tests for Phase 10 DESI-like catalog loader."""

from __future__ import annotations

from pathlib import Path

import pytest

from cosmic_engine.core.object_types import CosmicObjectType
from cosmic_engine.core.registry import UniverseRegistry
from cosmic_engine.core.truth import TruthLevel
from cosmic_engine.core.vector import Vector3
from cosmic_engine.data.desi_like_catalog import (
    load_desi_into_registry,
    load_desi_like_catalog,
)


_REPO_ROOT = Path(__file__).resolve().parent.parent
_SAMPLE = _REPO_ROOT / "data" / "sample_desi_like.csv"


def test_sample_desi_csv_loads():
    objects = load_desi_like_catalog(str(_SAMPLE))
    assert len(objects) == 10
    assert all(o.object_type is CosmicObjectType.GALAXY for o in objects)
    assert all(o.truth_level is TruthLevel.CATALOG_IMPORTED for o in objects)
    assert all(o.source == "desi_like_csv" for o in objects)


def test_redshift_stored_on_object_and_metadata():
    by_id = {o.id: o for o in load_desi_like_catalog(str(_SAMPLE))}
    g1 = by_id["desi_g1"]
    assert g1.redshift_z == pytest.approx(0.05)
    assert g1.metadata["apparent_magnitude"] == pytest.approx(18.2)


def test_positions_are_not_zero():
    for obj in load_desi_like_catalog(str(_SAMPLE)):
        assert obj.position_m != Vector3.zero()


def test_velocity_defaults_to_zero():
    for obj in load_desi_like_catalog(str(_SAMPLE)):
        assert obj.velocity_m_s == Vector3.zero()


def test_invalid_rows_are_skipped(tmp_path: Path):
    csv_path = tmp_path / "desi.csv"
    csv_path.write_text(
        "id,ra_deg,dec_deg,redshift_z,magnitude\n"
        "good,150.1,2.3,0.05,18.2\n"
        "bad_z,150.1,2.3,not_a_float,18.2\n"
        "neg_z,150.1,2.3,-0.01,18.2\n"
        ",10.0,2.3,0.1,18.2\n"
        "missing_ra,,2.3,0.05,18.2\n"
        "good2,80.5,-30.0,0.08,18.9\n"
    )
    objects = load_desi_like_catalog(str(csv_path))
    assert [o.id for o in objects] == ["good", "good2"]


def test_optional_magnitude_can_be_missing(tmp_path: Path):
    csv_path = tmp_path / "desi.csv"
    csv_path.write_text(
        "id,ra_deg,dec_deg,redshift_z,magnitude\n"
        "g_no_mag,10.0,5.0,0.1,\n"
    )
    obj = load_desi_like_catalog(str(csv_path))[0]
    assert obj.metadata["apparent_magnitude"] is None
    assert obj.redshift_z == pytest.approx(0.1)


def test_missing_required_column_raises(tmp_path: Path):
    csv_path = tmp_path / "desi.csv"
    csv_path.write_text(
        "id,ra_deg,dec_deg,magnitude\n"  # no redshift_z
        "g1,150.1,2.3,18.2\n"
    )
    with pytest.raises(ValueError):
        load_desi_like_catalog(str(csv_path))


def test_load_desi_into_registry_is_idempotent():
    registry = UniverseRegistry()
    load_desi_into_registry(str(_SAMPLE), registry)
    assert len(registry.list_objects()) == 10
    load_desi_into_registry(str(_SAMPLE), registry)
    assert len(registry.list_objects()) == 10
