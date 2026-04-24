"""Data ingestion for CosmicEngine.

Phase 2 contribution: read simple tabular catalogs (CSV) from local files
and turn them into :class:`UniverseObject` instances. No network, no API
clients, no scientific libraries — just the standard library.
"""

from cosmic_engine.data.catalog_loader import load_csv
from cosmic_engine.data.star_catalog import (
    load_star_catalog,
    load_star_catalog_into_registry,
)

__all__ = [
    "load_csv",
    "load_star_catalog",
    "load_star_catalog_into_registry",
]
