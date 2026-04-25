"""DESI catalog ingestion using the official-style column names.

Schema:
    targetid, ra, dec, z, mag (optional)

This is the modern entry point; :mod:`cosmic_engine.data.desi_like_catalog`
remains for the older ``id, ra_deg, dec_deg, redshift_z, magnitude``
schema and keeps existing demos / tests working.

Distances use the active cosmology mode (default ΛCDM).
"""

from __future__ import annotations

import logging
import math

from cosmic_engine.core.coordinates import ra_dec_distance_to_cartesian
from cosmic_engine.core.registry import UniverseRegistry
from cosmic_engine.core.units import PARSEC_IN_METERS
from cosmic_engine.core.universe_object import UniverseObject
from cosmic_engine.cosmos.galaxy import GalaxyProperties, create_galaxy_object
from cosmic_engine.data.catalog_loader import load_csv
from cosmic_engine.data.sources import DataSource, tag_source
from cosmic_engine.physics.cosmology import (
    get_cosmology_mode,
    redshift_to_distance_m,
)

_LOG = logging.getLogger(__name__)

_REQUIRED_COLUMNS = ("targetid", "ra", "dec", "z")


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
        targetid = row["targetid"]
        if not targetid:
            return None
        ra = float(row["ra"])
        dec = float(row["dec"])
        z = float(row["z"])
    except (KeyError, ValueError):
        return None

    if z < 0.0:
        return None

    magnitude = _parse_optional_float(row.get("mag", ""))
    distance_m = redshift_to_distance_m(z)
    position = ra_dec_distance_to_cartesian(ra, dec, distance_m)

    obj = create_galaxy_object(
        id=targetid,
        name=targetid,
        position_m=position,
        properties=GalaxyProperties(
            redshift_z=z,
            apparent_magnitude=magnitude,
        ),
        source="desi",
    )
    obj.metadata["distance_model"] = get_cosmology_mode()
    if magnitude is not None and distance_m > 0.0:
        d_l_pc = (1.0 + z) * distance_m / PARSEC_IN_METERS
        mu = 5.0 * (math.log10(d_l_pc) - 1.0)
        obj.metadata["absolute_magnitude"] = magnitude - mu
    else:
        obj.metadata["absolute_magnitude"] = None
    tag_source(obj, DataSource.DESI)
    return obj


def load_desi_catalog(file_path: str) -> list[UniverseObject]:
    """Load a DESI CSV into a list of galaxy :class:`UniverseObject`."""
    rows = load_csv(file_path)
    if rows:
        missing = [c for c in _REQUIRED_COLUMNS if c not in rows[0]]
        if missing:
            raise ValueError(
                f"DESI catalog {file_path} missing columns: {missing}"
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
            "load_desi_catalog(%s): skipped %d invalid row(s)",
            file_path,
            skipped,
        )
    return objects


def load_desi_into_registry(
    file_path: str,
    registry: UniverseRegistry,
) -> None:
    """Load a DESI CSV and add unique galaxies to ``registry``."""
    for obj in load_desi_catalog(file_path):
        if registry.get_object(obj.id) is None:
            registry.add_object(obj)
