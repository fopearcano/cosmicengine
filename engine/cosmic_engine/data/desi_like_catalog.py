"""DESI-like CSV ingestion.

Reads a small RA / Dec / redshift catalog, computes Hubble-law
distances via :mod:`cosmic_engine.physics.cosmology`, and emits
:class:`UniverseObject` galaxies tagged ``truth_level=CATALOG_IMPORTED``
with ``source="desi_like_csv"``.

Not a real DESI client — no network calls, no FITS, no proper
cosmology. The format is a structural placeholder for a future ingest
step that will speak DESI's actual schema.
"""

from __future__ import annotations

import logging

from cosmic_engine.core.coordinates import ra_dec_distance_to_cartesian
from cosmic_engine.core.registry import UniverseRegistry
from cosmic_engine.core.universe_object import UniverseObject
from cosmic_engine.cosmos.galaxy import GalaxyProperties, create_galaxy_object
from cosmic_engine.data.catalog_loader import load_csv
from cosmic_engine.physics.cosmology import redshift_to_distance_m

_LOG = logging.getLogger(__name__)

_REQUIRED_COLUMNS = ("id", "ra_deg", "dec_deg", "redshift_z")


def _parse_optional_float(raw: str) -> float | None:
    raw = raw.strip()
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _parse_row(row: dict[str, str]) -> UniverseObject | None:
    try:
        obj_id = row["id"]
        if not obj_id:
            return None
        ra_deg = float(row["ra_deg"])
        dec_deg = float(row["dec_deg"])
        z = float(row["redshift_z"])
    except (KeyError, ValueError):
        return None

    if z < 0.0:
        return None

    magnitude = _parse_optional_float(row.get("magnitude", ""))
    distance_m = redshift_to_distance_m(z)
    position = ra_dec_distance_to_cartesian(ra_deg, dec_deg, distance_m)

    properties = GalaxyProperties(
        redshift_z=z,
        apparent_magnitude=magnitude,
    )
    return create_galaxy_object(
        id=obj_id,
        name=obj_id,
        position_m=position,
        properties=properties,
        source="desi_like_csv",
    )


def load_desi_like_catalog(file_path: str) -> list[UniverseObject]:
    """Load a DESI-like CSV into a list of :class:`UniverseObject` galaxies.

    Required columns: ``id``, ``ra_deg``, ``dec_deg``, ``redshift_z``.
    ``magnitude`` is optional. Rows with missing / unparseable values
    or negative redshift are skipped.
    """
    rows = load_csv(file_path)
    if rows:
        missing = [c for c in _REQUIRED_COLUMNS if c not in rows[0]]
        if missing:
            raise ValueError(
                f"DESI-like catalog {file_path} missing columns: {missing}"
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
            "load_desi_like_catalog(%s): skipped %d invalid row(s)",
            file_path,
            skipped,
        )
    return objects


def load_desi_into_registry(
    file_path: str,
    registry: UniverseRegistry,
) -> None:
    """Load a DESI-like CSV and add each galaxy to ``registry``.

    Galaxies whose IDs already exist in the registry are skipped.
    """
    duplicates = 0
    for obj in load_desi_like_catalog(file_path):
        if registry.get_object(obj.id) is not None:
            duplicates += 1
            continue
        registry.add_object(obj)
    if duplicates:
        _LOG.warning(
            "load_desi_into_registry(%s): skipped %d duplicate id(s)",
            file_path,
            duplicates,
        )
