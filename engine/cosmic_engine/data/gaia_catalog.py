"""Gaia-like star catalog ingestion.

Distance is computed from parallax via the textbook ``d_pc = 1000 / ϖ_mas``;
negative or zero parallaxes are skipped (they correspond to noise-dominated
or formally negative distance estimates that need a Bayesian prior to
interpret correctly).
"""

from __future__ import annotations

import logging

from cosmic_engine.core.coordinates import ra_dec_distance_to_cartesian
from cosmic_engine.core.object_types import CosmicObjectType
from cosmic_engine.core.registry import UniverseRegistry
from cosmic_engine.core.truth import TruthLevel
from cosmic_engine.core.units import PARSEC_IN_METERS
from cosmic_engine.core.universe_object import UniverseObject
from cosmic_engine.core.vector import Vector3
from cosmic_engine.data.catalog_loader import load_csv
from cosmic_engine.data.sources import DataSource, tag_source

_LOG = logging.getLogger(__name__)

_REQUIRED_COLUMNS = ("source_id", "ra", "dec", "parallax_mas", "phot_g_mean_mag")


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
        source_id = row["source_id"]
        if not source_id:
            return None
        ra = float(row["ra"])
        dec = float(row["dec"])
        parallax_mas = float(row["parallax_mas"])
        magnitude = float(row["phot_g_mean_mag"])
    except (KeyError, ValueError):
        return None

    if parallax_mas <= 0.0:
        return None

    distance_pc = 1000.0 / parallax_mas
    distance_m = distance_pc * PARSEC_IN_METERS
    position = ra_dec_distance_to_cartesian(ra, dec, distance_m)

    bp_rp = _parse_optional_float(row.get("bp_rp", ""))

    obj = UniverseObject(
        id=source_id,
        name=source_id,
        object_type=CosmicObjectType.STAR,
        position_m=position,
        velocity_m_s=Vector3.zero(),
        truth_level=TruthLevel.CATALOG_IMPORTED,
        metadata={
            "apparent_magnitude": magnitude,
            "color_index": bp_rp,
            "parallax_mas": parallax_mas,
        },
    )
    tag_source(obj, DataSource.GAIA)
    return obj


def load_gaia_like_catalog(file_path: str) -> list[UniverseObject]:
    """Load a Gaia-style CSV into a list of star :class:`UniverseObject`."""
    rows = load_csv(file_path)
    if rows:
        missing = [c for c in _REQUIRED_COLUMNS if c not in rows[0]]
        if missing:
            raise ValueError(
                f"Gaia catalog {file_path} missing columns: {missing}"
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
            "load_gaia_like_catalog(%s): skipped %d invalid row(s)",
            file_path,
            skipped,
        )
    return objects


def load_gaia_into_registry(
    file_path: str,
    registry: UniverseRegistry,
) -> None:
    """Load a Gaia-style CSV and add unique stars to ``registry``."""
    for obj in load_gaia_like_catalog(file_path):
        if registry.get_object(obj.id) is None:
            registry.add_object(obj)
