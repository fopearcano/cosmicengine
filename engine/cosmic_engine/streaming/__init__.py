"""Streaming and Level-of-Detail layer.

Phase 23 contribution: scale the engine beyond what fits in RAM by
chunking large datasets into spatial tiles and streaming only the
tiles relevant to the current observer. A simple Level-of-Detail
selector then trims the candidate set to the runtime's active-object
budget while preserving the spatial distribution.

No databases, no GPU, no multiprocessing — file-backed JSON tiles and
an in-process LRU cache.
"""

from cosmic_engine.streaming.lod import (
    compute_lod_weight,
    select_lod_objects,
)
from cosmic_engine.streaming.spatial_index import SpatialIndex
from cosmic_engine.streaming.tile_store import (
    TileStore,
    partition_objects_into_tiles,
)

__all__ = [
    "SpatialIndex",
    "TileStore",
    "compute_lod_weight",
    "partition_objects_into_tiles",
    "select_lod_objects",
]
