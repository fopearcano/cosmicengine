"""Minimal star-catalog ingestion.

Reads a CSV with columns:
    id, name, ra_deg, dec_deg, distance_ly,
    apparent_magnitude, spectral_class
and produces :class:`UniverseObject` instances tagged with
``TruthLevel.CATALOG_IMPORTED``.
"""

from __future__ import annotations

import logging
from typing import Any

from cosmic_engine.core.coordinates import ra_dec_distance_to_cartesian
from cosmic_engine.core.object_types import CosmicObjectType
from cosmic_engine.core.registry import UniverseRegistry
from cosmic_engine.core.truth import TruthLevel
from cosmic_engine.core.units import lightyear_to_meters
from cosmic_engine.core.universe_object import UniverseObject
from cosmic_engine.core.vector import Vector3
from cosmic_engine.data.catalog_loader import load_csv

_LOG = logging.getLogger(__name__)

_REQUIRED_COLUMNS = (
    "id",
    "name",
    "ra_deg",
    "dec_deg",
    "distance_ly",
    "apparent_magnitude",
    "spectral_class",
)


def _parse_row(row: dict[str, str]) -> UniverseObject | None:
    """Return a :class:`UniverseObject` for ``row``, or ``None`` if invalid."""
    try:
        obj_id = row["id"]
        name = row["name"]
        if not obj_id or not name:
            return None

        ra_deg = float(row["ra_deg"])
        dec_deg = float(row["dec_deg"])
        distance_ly = float(row["distance_ly"])
        apparent_magnitude = float(row["apparent_magnitude"])
        spectral_class = row["spectral_class"]

        if distance_ly < 0:
            return None
        if not spectral_class:
            return None
    except (KeyError, ValueError):
        return None

    distance_m = lightyear_to_meters(distance_ly)
    position_m = ra_dec_distance_to_cartesian(ra_deg, dec_deg, distance_m)

    metadata: dict[str, Any] = {"apparent_magnitude": apparent_magnitude}

    return UniverseObject(
        id=obj_id,
        name=name,
        object_type=CosmicObjectType.STAR,
        position_m=position_m,
        velocity_m_s=Vector3.zero(),
        truth_level=TruthLevel.CATALOG_IMPORTED,
        source="CSV",
        spectral_class=spectral_class,
        metadata=metadata,
    )


def load_star_catalog(file_path: str) -> list[UniverseObject]:
    """Load a star catalog CSV into a list of :class:`UniverseObject`.

    Rows with missing or unparseable numeric fields are skipped; the
    count of skipped rows is emitted to the module logger.
    """
    rows = load_csv(file_path)
    if rows:
        missing = [c for c in _REQUIRED_COLUMNS if c not in rows[0]]
        if missing:
            raise ValueError(
                f"star catalog {file_path} missing columns: {missing}"
            )

    objects: list[UniverseObject] = []
    skipped = 0
    for row in rows:
        obj = _parse_row(row)
        if obj is None:
            skipped += 1
            continue
        objects.append(obj)

    if skipped:
        _LOG.warning(
            "load_star_catalog(%s): skipped %d invalid row(s)",
            file_path,
            skipped,
        )
    return objects


def load_star_catalog_into_registry(
    file_path: str,
    registry: UniverseRegistry,
) -> None:
    """Load a star catalog and add it to ``registry``.

    Objects whose IDs already exist in the registry are skipped with a
    warning rather than raising, so repeated loads are idempotent.
    """
    duplicates = 0
    for obj in load_star_catalog(file_path):
        if registry.get_object(obj.id) is not None:
            duplicates += 1
            continue
        registry.add_object(obj)
    if duplicates:
        _LOG.warning(
            "load_star_catalog_into_registry(%s): skipped %d duplicate id(s)",
            file_path,
            duplicates,
        )
