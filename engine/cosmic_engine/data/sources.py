"""Unified ingestion entry point and source taxonomy.

Phase 12 contribution: a thin dispatch layer that lets callers say
"load this CSV as Gaia / SDSS / DESI" without coupling to the specific
loader module. JPL is intentionally not in :func:`load_catalog` —
that one is a no-file in-memory placeholder.
"""

from __future__ import annotations

from enum import Enum

from cosmic_engine.core.registry import UniverseRegistry
from cosmic_engine.core.universe_object import UniverseObject


class DataSource(str, Enum):
    """Provenance tags for a registered :class:`UniverseObject`."""

    DESI = "desi"
    SDSS = "sdss"
    GAIA = "gaia"
    NASA = "nasa"
    JPL = "jpl"
    SYNTHETIC = "synthetic"


def tag_source(obj: UniverseObject, source: DataSource) -> None:
    """Stamp ``obj`` with the given data source.

    Sets ``obj.source`` to the enum value and additionally records
    ``metadata["data_source"]`` so source filtering survives any later
    serialization that may strip the top-level ``source`` field.
    """
    obj.source = source.value
    obj.metadata["data_source"] = source.value


def load_catalog(
    file_path: str,
    source: DataSource,
) -> list[UniverseObject]:
    """Dispatch to the file-based loader for ``source``.

    Supports :attr:`DataSource.GAIA`, :attr:`DataSource.SDSS`, and
    :attr:`DataSource.DESI`. Other sources (JPL, NASA, SYNTHETIC) have
    no file format here yet and raise :class:`ValueError`.
    """
    # Local imports so importing :mod:`sources` doesn't pull in every
    # loader module at startup.
    if source is DataSource.GAIA:
        from cosmic_engine.data.gaia_catalog import load_gaia_like_catalog

        return load_gaia_like_catalog(file_path)
    if source is DataSource.SDSS:
        from cosmic_engine.data.sdss_catalog import load_sdss_like_catalog

        return load_sdss_like_catalog(file_path)
    if source is DataSource.DESI:
        from cosmic_engine.data.desi_catalog import load_desi_catalog

        return load_desi_catalog(file_path)
    raise ValueError(f"unsupported source for file ingestion: {source}")


def load_catalog_into_registry(
    file_path: str,
    source: DataSource,
    registry: UniverseRegistry,
) -> None:
    """Load a file-based catalog and add each object to ``registry``.

    Objects with IDs that already exist in the registry are skipped
    (idempotent over repeated calls).
    """
    for obj in load_catalog(file_path, source):
        if registry.get_object(obj.id) is None:
            registry.add_object(obj)
