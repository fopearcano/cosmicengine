"""Central orchestrator for the headless engine."""

from __future__ import annotations

import logging
import math
from pathlib import Path

from cosmic_engine.core.registry import UniverseRegistry
from cosmic_engine.core.time import SimulationClock
from cosmic_engine.core.universe_object import UniverseObject
from cosmic_engine.core.vector import Vector3
from cosmic_engine.runtime.config import RuntimeConfig
from cosmic_engine.runtime.scene_state import SceneState
from cosmic_engine.streaming.lod import select_lod_objects
from cosmic_engine.streaming.spatial_index import SpatialIndex
from cosmic_engine.streaming.tile_store import (
    TileStore,
    cell_index_for_position,
    partition_objects_into_tiles,
    tile_id_for_cell,
)

_LOG = logging.getLogger(__name__)

_J2000_JD = 2_451_545.0
_DEFAULT_TILE_CACHE_SIZE = 16


class _TileCache:
    """Tiny LRU cache for tile contents (FIFO-ish on equal access)."""

    def __init__(self, max_size: int = _DEFAULT_TILE_CACHE_SIZE) -> None:
        if max_size <= 0:
            raise ValueError("max_size must be positive")
        self.max_size = max_size
        self._store: dict[str, list[UniverseObject]] = {}
        self._order: list[str] = []

    def __len__(self) -> int:
        return len(self._store)

    def get(self, tile_id: str) -> list[UniverseObject] | None:
        if tile_id not in self._store:
            return None
        self._order.remove(tile_id)
        self._order.append(tile_id)
        return self._store[tile_id]

    def put(self, tile_id: str, objects: list[UniverseObject]) -> None:
        if tile_id in self._store:
            self._order.remove(tile_id)
        elif len(self._store) >= self.max_size:
            evicted = self._order.pop(0)
            self._store.pop(evicted, None)
        self._store[tile_id] = objects
        self._order.append(tile_id)

    def clear(self) -> None:
        self._store.clear()
        self._order.clear()


class CosmicRuntime:
    """Owns a :class:`UniverseRegistry`, a :class:`SimulationClock`, and a config.

    The runtime is deliberately thin: it knows how to schedule physics,
    counting, and active-object selection but delegates every actual
    transform to the existing modules.
    """

    def __init__(
        self,
        registry: UniverseRegistry | None = None,
        config: RuntimeConfig | None = None,
        clock: SimulationClock | None = None,
    ) -> None:
        self.registry = registry if registry is not None else UniverseRegistry()
        self.config = config if config is not None else RuntimeConfig()
        self.config.validate()
        self.clock = (
            clock
            if clock is not None
            else SimulationClock(current_julian_date=_J2000_JD)
        )
        self.last_scene_state: SceneState | None = None
        # Streaming / LOD state (Phase 23). All optional; ``None`` means
        # the runtime behaves as before.
        self.spatial_index: SpatialIndex | None = None
        self.tile_store: TileStore | None = None
        self._tile_cell_size_m: float | None = None
        self._tile_cache: _TileCache | None = None
        self.last_loaded_tiles: list[str] = []

    # --- ingestion --------------------------------------------------------

    def add_objects(self, objects: list[UniverseObject]) -> None:
        """Add objects to the registry, skipping IDs that already exist."""
        for obj in objects:
            if self.registry.get_object(obj.id) is None:
                self.registry.add_object(obj)

    def load_sample_data(self) -> None:
        """Best-effort load of the bundled sample CSVs and JPL placeholder."""
        # Resolve repo root from this file's path:
        #   .../engine/cosmic_engine/runtime/runtime.py
        repo_root = Path(__file__).resolve().parents[3]
        data_dir = repo_root / "data"

        # Lazy imports keep module-level surface area minimal.
        from cosmic_engine.data.desi_catalog import load_desi_catalog
        from cosmic_engine.data.gaia_catalog import load_gaia_like_catalog
        from cosmic_engine.data.jpl_ephemeris import (
            load_jpl_ephemeris_placeholder,
        )
        from cosmic_engine.data.sdss_catalog import load_sdss_like_catalog

        sources: list[tuple[str, Path, callable]] = [
            ("gaia", data_dir / "sample_gaia_like.csv", load_gaia_like_catalog),
            ("sdss", data_dir / "sample_sdss_like.csv", load_sdss_like_catalog),
            ("desi", data_dir / "sample_desi.csv", load_desi_catalog),
        ]

        for label, path, loader in sources:
            if not path.is_file():
                _LOG.info("load_sample_data: %s not found at %s", label, path)
                continue
            try:
                self.add_objects(loader(str(path)))
            except Exception as e:  # pragma: no cover - defensive
                _LOG.warning("load_sample_data: %s failed: %s", label, e)

        # JPL placeholder needs no file.
        try:
            self.add_objects(
                load_jpl_ephemeris_placeholder(self.clock.get_julian_date())
            )
        except Exception as e:  # pragma: no cover - defensive
            _LOG.warning("load_sample_data: jpl placeholder failed: %s", e)

    # --- selection --------------------------------------------------------

    def select_active_objects(
        self,
        observer_position: Vector3,
    ) -> list[UniverseObject]:
        """Return a deterministic, capped slice of registry objects.

        With streaming or a spatial index attached, candidates come from
        the relevant tiles / cells around ``observer_position`` instead
        of the entire registry. The :func:`select_lod_objects`
        probabilistic downsampler caps the result at
        ``config.max_active_objects`` while preserving the spatial
        distribution.
        """
        candidates = self._gather_candidates(observer_position)
        candidates = sorted(candidates, key=lambda o: o.id)
        return select_lod_objects(
            candidates,
            observer_position,
            self.config.max_active_objects,
        )

    def _gather_candidates(
        self,
        observer_position: Vector3,
    ) -> list[UniverseObject]:
        """Choose between in-memory / spatial-index / tile-store sources."""
        radius = self.config.active_radius_m

        if self.spatial_index is not None and radius is not None:
            return self.spatial_index.query_radius(observer_position, radius)

        if (
            self.tile_store is not None
            and self._tile_cell_size_m is not None
            and radius is not None
        ):
            return self._gather_tiles_around(observer_position, radius)

        # Fallback: filter the in-memory registry.
        objects = self.registry.list_objects()
        if radius is None:
            return objects
        r2 = radius * radius
        ox = observer_position.x
        oy = observer_position.y
        oz = observer_position.z
        return [
            o
            for o in objects
            if (o.position_m.x - ox) ** 2
            + (o.position_m.y - oy) ** 2
            + (o.position_m.z - oz) ** 2
            <= r2
        ]

    def _gather_tiles_around(
        self,
        observer_position: Vector3,
        radius: float,
    ) -> list[UniverseObject]:
        """Load tiles around the observer through the LRU cache and filter."""
        if self.tile_store is None or self._tile_cell_size_m is None:
            return []
        cache = self._tile_cache
        cell_size = self._tile_cell_size_m
        cx, cy, cz = cell_index_for_position(
            observer_position.x,
            observer_position.y,
            observer_position.z,
            cell_size,
        )
        cell_radius = int(math.ceil(radius / cell_size))
        loaded_tiles: list[str] = []
        candidates: list[UniverseObject] = []
        r2 = radius * radius
        ox, oy, oz = observer_position.x, observer_position.y, observer_position.z
        for dx in range(-cell_radius, cell_radius + 1):
            for dy in range(-cell_radius, cell_radius + 1):
                for dz in range(-cell_radius, cell_radius + 1):
                    tile_id = tile_id_for_cell(cx + dx, cy + dy, cz + dz)
                    objects = cache.get(tile_id) if cache is not None else None
                    if objects is None:
                        if not self.tile_store.has_tile(tile_id):
                            continue
                        objects = self.tile_store.load_tile(tile_id)
                        if cache is not None:
                            cache.put(tile_id, objects)
                    loaded_tiles.append(tile_id)
                    for obj in objects:
                        dxp = obj.position_m.x - ox
                        dyp = obj.position_m.y - oy
                        dzp = obj.position_m.z - oz
                        if dxp * dxp + dyp * dyp + dzp * dzp <= r2:
                            candidates.append(obj)
        self.last_loaded_tiles = loaded_tiles
        return candidates

    # --- streaming bring-up ----------------------------------------------

    def build_spatial_index(self, cell_size_m: float) -> SpatialIndex:
        """Build a :class:`SpatialIndex` from the current registry contents."""
        index = SpatialIndex(cell_size_m)
        index.build(self.registry.list_objects())
        self.spatial_index = index
        return index

    def enable_streaming(
        self,
        cell_size_m: float,
        data_directory: str,
        clear_registry: bool = True,
        cache_size: int = _DEFAULT_TILE_CACHE_SIZE,
    ) -> TileStore:
        """Partition the current registry into tiles on disk.

        With ``clear_registry=True`` (the default) the in-memory
        registry is emptied so memory tracks the cache rather than
        the full dataset.
        """
        if cell_size_m <= 0.0:
            raise ValueError("cell_size_m must be positive")
        store = TileStore(data_directory)
        partitions = partition_objects_into_tiles(
            self.registry.list_objects(), cell_size_m
        )
        for tile_id, objects in partitions.items():
            store.save_tile(tile_id, objects)
        self.tile_store = store
        self._tile_cell_size_m = cell_size_m
        self._tile_cache = _TileCache(max_size=cache_size)
        if clear_registry:
            self.registry = UniverseRegistry()
        return store

    # --- stepping ---------------------------------------------------------

    def step(self, delta_seconds: float) -> SceneState:
        """Advance the clock and (if configured) one physics step."""
        self.clock.tick(delta_seconds)
        notes: list[str] = [
            f"clock advanced by {delta_seconds} s",
        ]

        if self.config.enable_physics and self.config.physics_backend != "none":
            notes.extend(self._run_physics_step(delta_seconds))
        else:
            notes.append("physics: disabled")

        state = self.build_scene_state()
        state.notes = notes
        self.last_scene_state = state
        return state

    def _run_physics_step(self, delta_seconds: float) -> list[str]:
        """Run one physics step on all massive registry objects."""
        from cosmic_engine.physics.nbody import (
            NBodySimulator,
            apply_nbody_state_to_objects,
            objects_to_nbody_state,
        )

        massive = [
            o for o in self.registry.list_objects() if o.mass_kg is not None
        ]
        if len(massive) < 2:
            return ["physics: skipped (need >= 2 massive bodies)"]

        try:
            state = objects_to_nbody_state(massive)
        except ValueError as e:
            return [f"physics: skipped ({e})"]

        backend = self.config.physics_backend
        integrator = "leapfrog" if backend == "exact_nbody" else "barnes_hut"
        try:
            sim = NBodySimulator(state, integrator=integrator)
            sim.step(delta_seconds)
            apply_nbody_state_to_objects(massive, sim.get_state())
        except Exception as e:  # pragma: no cover - defensive
            return [f"physics: failed ({e})"]
        return [f"physics: {backend} stepped {len(massive)} bodies"]

    # --- introspection ----------------------------------------------------

    def build_scene_state(self) -> SceneState:
        """Build a :class:`SceneState` from the current registry."""
        objects = self.registry.list_objects()
        type_counts: dict[str, int] = {}
        truth_counts: dict[str, int] = {}
        source_counts: dict[str, int] = {}
        for obj in objects:
            t = obj.object_type.value
            type_counts[t] = type_counts.get(t, 0) + 1
            tl = obj.truth_level.value
            truth_counts[tl] = truth_counts.get(tl, 0) + 1
            src = obj.source if obj.source is not None else "unknown"
            source_counts[src] = source_counts.get(src, 0) + 1

        return SceneState(
            julian_date=self.clock.get_julian_date(),
            total_objects=len(objects),
            active_objects=min(len(objects), self.config.max_active_objects),
            object_type_counts=type_counts,
            truth_level_counts=truth_counts,
            source_counts=source_counts,
            physics_backend=self.config.physics_backend,
            perception_enabled=self.config.enable_perception,
            ai_warp_enabled=self.config.enable_ai_warp,
            notes=[],
        )
