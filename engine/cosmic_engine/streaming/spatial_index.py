"""A pure-Python uniform-grid spatial index.

Cells are addressed by integer ``(ix, iy, iz)`` keys; a position
``(x, y, z)`` maps to ``floor(x / cell_size_m), …``. Radius queries
visit every cell whose origin is within the bounding cube of
``radius`` and then filter by exact distance.

No KD-tree / R-tree / external library — uniform grids are
trivially deterministic and good enough for moderate object counts.
"""

from __future__ import annotations

import math

from cosmic_engine.core.universe_object import UniverseObject
from cosmic_engine.core.vector import Vector3


class SpatialIndex:
    """Hashed uniform grid keyed by integer cell coordinates."""

    def __init__(self, cell_size_m: float) -> None:
        if cell_size_m <= 0.0:
            raise ValueError(
                f"cell_size_m must be positive; got {cell_size_m}"
            )
        self.cell_size_m = float(cell_size_m)
        self._cells: dict[tuple[int, int, int], list[UniverseObject]] = {}
        self._count = 0

    def __len__(self) -> int:
        return self._count

    # --- internals --------------------------------------------------------

    def _cell_key(self, x: float, y: float, z: float) -> tuple[int, int, int]:
        return (
            int(math.floor(x / self.cell_size_m)),
            int(math.floor(y / self.cell_size_m)),
            int(math.floor(z / self.cell_size_m)),
        )

    # --- mutators ---------------------------------------------------------

    def insert(self, obj: UniverseObject) -> None:
        """Add a single :class:`UniverseObject` to its cell."""
        key = self._cell_key(
            obj.position_m.x, obj.position_m.y, obj.position_m.z
        )
        self._cells.setdefault(key, []).append(obj)
        self._count += 1

    def build(self, objects: list[UniverseObject]) -> None:
        """Rebuild the index from scratch over ``objects``."""
        self._cells.clear()
        self._count = 0
        for obj in objects:
            self.insert(obj)

    # --- queries ----------------------------------------------------------

    def query_cell(
        self, cell_key: tuple[int, int, int]
    ) -> list[UniverseObject]:
        """Return a copy of the contents of one cell."""
        return list(self._cells.get(cell_key, []))

    def query_radius(
        self,
        center: Vector3,
        radius: float,
    ) -> list[UniverseObject]:
        """Return objects within ``radius`` of ``center`` (inclusive)."""
        if radius < 0.0:
            raise ValueError("radius must be non-negative")
        if radius == 0.0:
            return []
        center_key = self._cell_key(center.x, center.y, center.z)
        cell_radius = int(math.ceil(radius / self.cell_size_m))
        radius_sq = radius * radius
        results: list[UniverseObject] = []
        for dx in range(-cell_radius, cell_radius + 1):
            for dy in range(-cell_radius, cell_radius + 1):
                for dz in range(-cell_radius, cell_radius + 1):
                    key = (
                        center_key[0] + dx,
                        center_key[1] + dy,
                        center_key[2] + dz,
                    )
                    cell = self._cells.get(key)
                    if not cell:
                        continue
                    for obj in cell:
                        ox = obj.position_m.x - center.x
                        oy = obj.position_m.y - center.y
                        oz = obj.position_m.z - center.z
                        if ox * ox + oy * oy + oz * oz <= radius_sq:
                            results.append(obj)
        return results

    def cell_count(self) -> int:
        """Number of non-empty cells in the index."""
        return len(self._cells)
