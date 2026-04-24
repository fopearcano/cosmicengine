"""Tests for the Phase 1 core object model."""

from __future__ import annotations

import pytest

from cosmic_engine.core import (
    CosmicObjectType,
    TruthLevel,
    UniverseObject,
    UniverseRegistry,
    Vector3,
)


def _sample_object(object_id: str = "sun", **overrides) -> UniverseObject:
    defaults = dict(
        id=object_id,
        name="Sun",
        object_type=CosmicObjectType.STAR,
        position_m=Vector3.zero(),
        velocity_m_s=Vector3.zero(),
        truth_level=TruthLevel.EPHEMERIS_REAL,
        source="test",
        mass_kg=1.989e30,
        radius_m=6.957e8,
        luminosity_w=3.828e26,
        spectral_class="G2V",
    )
    defaults.update(overrides)
    return UniverseObject(**defaults)


def test_vector3_serialization_round_trip():
    v = Vector3(1.0, -2.5, 3.25)
    assert v.to_list() == [1.0, -2.5, 3.25]
    assert Vector3.from_list(v.to_list()) == v
    assert Vector3.zero() == Vector3(0.0, 0.0, 0.0)


def test_vector3_from_list_rejects_wrong_length():
    with pytest.raises(ValueError):
        Vector3.from_list([1.0, 2.0])


def test_universe_object_validate_accepts_sample():
    _sample_object().validate()


@pytest.mark.parametrize(
    "overrides",
    [
        {"id": ""},
        {"name": ""},
        {"mass_kg": -1.0},
        {"radius_m": -1.0},
        {"luminosity_w": -1.0},
        {"redshift_z": -0.1},
    ],
)
def test_universe_object_validate_rejects_bad_fields(overrides):
    obj = _sample_object(**overrides)
    with pytest.raises(ValueError):
        obj.validate()


def test_universe_object_dict_round_trip():
    original = _sample_object(
        redshift_z=0.0,
        position_m=Vector3(1.0, 2.0, 3.0),
        velocity_m_s=Vector3(-1.0, 0.0, 1.0),
    )
    original.metadata["note"] = "hello"
    data = original.to_dict()
    assert data["object_type"] == "star"
    assert data["truth_level"] == "ephemeris_real"
    assert data["position_m"] == [1.0, 2.0, 3.0]
    restored = UniverseObject.from_dict(data)
    assert restored == original


def test_registry_add_get_remove():
    registry = UniverseRegistry()
    obj = _sample_object()
    registry.add_object(obj)
    assert registry.get_object("sun") is obj
    assert registry.list_objects() == [obj]
    registry.remove_object("sun")
    assert registry.get_object("sun") is None
    assert registry.list_objects() == []


def test_registry_remove_missing_does_not_raise():
    registry = UniverseRegistry()
    registry.remove_object("nonexistent")


def test_registry_duplicate_id_raises():
    registry = UniverseRegistry()
    registry.add_object(_sample_object())
    with pytest.raises(ValueError):
        registry.add_object(_sample_object())


def test_registry_query_by_type():
    registry = UniverseRegistry()
    star = _sample_object("sun")
    galaxy = _sample_object(
        "milky_way",
        name="Milky Way",
        object_type=CosmicObjectType.GALAXY,
        truth_level=TruthLevel.CATALOG_IMPORTED,
    )
    registry.add_object(star)
    registry.add_object(galaxy)
    assert registry.query_by_type(CosmicObjectType.STAR) == [star]
    assert registry.query_by_type(CosmicObjectType.GALAXY) == [galaxy]
    assert registry.query_by_type(CosmicObjectType.PLANET) == []


def test_registry_dict_round_trip_preserves_enums_and_vectors():
    registry = UniverseRegistry()
    registry.add_object(_sample_object("sun"))
    registry.add_object(
        _sample_object(
            "milky_way",
            name="Milky Way",
            object_type=CosmicObjectType.GALAXY,
            position_m=Vector3(1e20, 0.0, 0.0),
            truth_level=TruthLevel.CATALOG_IMPORTED,
        )
    )
    restored = UniverseRegistry.from_dict(registry.to_dict())
    assert {o.id for o in restored.list_objects()} == {"sun", "milky_way"}
    milky = restored.get_object("milky_way")
    assert milky is not None
    assert milky.object_type is CosmicObjectType.GALAXY
    assert milky.truth_level is TruthLevel.CATALOG_IMPORTED
    assert milky.position_m == Vector3(1e20, 0.0, 0.0)
