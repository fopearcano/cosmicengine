"""File-backed tile storage for streaming large catalogs.

Each tile is one JSON file under ``data_directory`` named
``<tile_id>.json``. Tile IDs are produced by
:func:`partition_objects_into_tiles` from a fixed cell size, so the
runtime can deterministically map an observer position to the set of
tiles that cover its active radius.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from cosmic_engine.core.universe_object import UniverseObject


def tile_id_for_cell(ix: int, iy: int, iz: int) -> str:
    """Return the canonical tile id for a cell index triple."""
    return f"tile_{ix}_{iy}_{iz}"


def cell_index_for_position(
    x: float, y: float, z: float, cell_size_m: float
) -> tuple[int, int, int]:
    """Floor-divide a Cartesian position into cell indices."""
    if cell_size_m <= 0.0:
        raise ValueError("cell_size_m must be positive")
    return (
        int(math.floor(x / cell_size_m)),
        int(math.floor(y / cell_size_m)),
        int(math.floor(z / cell_size_m)),
    )


def partition_objects_into_tiles(
    objects: list[UniverseObject],
    cell_size_m: float,
) -> dict[str, list[UniverseObject]]:
    """Bucket objects into tiles keyed by cell index.

    Object identity is preserved; the same object never appears in
    more than one tile.
    """
    if cell_size_m <= 0.0:
        raise ValueError("cell_size_m must be positive")
    tiles: dict[str, list[UniverseObject]] = {}
    for obj in objects:
        ix, iy, iz = cell_index_for_position(
            obj.position_m.x,
            obj.position_m.y,
            obj.position_m.z,
            cell_size_m,
        )
        tiles.setdefault(tile_id_for_cell(ix, iy, iz), []).append(obj)
    return tiles


class TileStore:
    """JSON-backed object tiles on disk."""

    def __init__(self, data_directory: str) -> None:
        if not data_directory:
            raise ValueError("data_directory must be a non-empty string")
        self.data_directory = Path(data_directory)
        self.data_directory.mkdir(parents=True, exist_ok=True)

    def _tile_path(self, tile_id: str) -> Path:
        return self.data_directory / f"{tile_id}.json"

    def save_tile(
        self,
        tile_id: str,
        objects: list[UniverseObject],
    ) -> None:
        """Write ``objects`` to ``<tile_id>.json``."""
        if not tile_id:
            raise ValueError("tile_id must be a non-empty string")
        path = self._tile_path(tile_id)
        payload = {
            "tile_id": tile_id,
            "objects": [o.to_dict() for o in objects],
        }
        path.write_text(json.dumps(payload))

    def load_tile(self, tile_id: str) -> list[UniverseObject]:
        """Read and reconstruct the objects in ``<tile_id>.json``."""
        path = self._tile_path(tile_id)
        if not path.is_file():
            raise FileNotFoundError(f"tile not found: {path}")
        data = json.loads(path.read_text())
        return [UniverseObject.from_dict(o) for o in data.get("objects", [])]

    def list_tiles(self) -> list[str]:
        """Return all tile ids currently on disk, sorted."""
        return sorted(p.stem for p in self.data_directory.glob("*.json"))

    def has_tile(self, tile_id: str) -> bool:
        return self._tile_path(tile_id).is_file()
