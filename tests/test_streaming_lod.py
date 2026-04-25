"""Tests for Phase 23 streaming + LOD."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from cosmic_engine.core.object_types import CosmicObjectType
from cosmic_engine.core.truth import TruthLevel
from cosmic_engine.core.universe_object import UniverseObject
from cosmic_engine.core.vector import Vector3
from cosmic_engine.runtime import CosmicRuntime, RuntimeConfig
from cosmic_engine.streaming import (
    SpatialIndex,
    TileStore,
    compute_lod_weight,
    partition_objects_into_tiles,
    select_lod_objects,
)
from cosmic_engine.streaming.tile_store import (
    cell_index_for_position,
    tile_id_for_cell,
)


def _star(object_id: str, position: Vector3) -> UniverseObject:
    return UniverseObject(
        id=object_id,
        name=object_id,
        object_type=CosmicObjectType.STAR,
        position_m=position,
        velocity_m_s=Vector3.zero(),
        truth_level=TruthLevel.CATALOG_IMPORTED,
        source="test",
    )


# --- SpatialIndex ---


def test_spatial_index_rejects_non_positive_cell_size():
    with pytest.raises(ValueError):
        SpatialIndex(0.0)
    with pytest.raises(ValueError):
        SpatialIndex(-1.0)


def test_spatial_index_query_radius_returns_only_within():
    index = SpatialIndex(cell_size_m=10.0)
    objects = [
        _star("near", Vector3(1.0, 0.0, 0.0)),
        _star("mid", Vector3(15.0, 0.0, 0.0)),
        _star("far", Vector3(50.0, 0.0, 0.0)),
        _star("offaxis", Vector3(0.0, 8.0, 0.0)),
    ]
    index.build(objects)
    assert len(index) == 4
    nearby = index.query_radius(Vector3.zero(), radius=12.0)
    ids = sorted(o.id for o in nearby)
    assert ids == ["near", "offaxis"]


def test_spatial_index_query_radius_zero_returns_empty():
    index = SpatialIndex(cell_size_m=5.0)
    index.insert(_star("a", Vector3(1.0, 0.0, 0.0)))
    assert index.query_radius(Vector3.zero(), 0.0) == []


def test_spatial_index_query_radius_rejects_negative():
    index = SpatialIndex(cell_size_m=5.0)
    with pytest.raises(ValueError):
        index.query_radius(Vector3.zero(), -1.0)


def test_spatial_index_query_cell():
    index = SpatialIndex(cell_size_m=10.0)
    obj = _star("a", Vector3(5.0, 5.0, 5.0))
    index.insert(obj)
    # cell index for (5,5,5) at cell_size 10 is (0,0,0)
    in_cell = index.query_cell((0, 0, 0))
    assert len(in_cell) == 1
    assert in_cell[0].id == "a"
    assert index.query_cell((1, 0, 0)) == []


def test_spatial_index_negative_coordinates():
    index = SpatialIndex(cell_size_m=10.0)
    obj = _star("neg", Vector3(-15.0, -5.0, 0.0))
    index.insert(obj)
    found = index.query_radius(Vector3(-15.0, -5.0, 0.0), radius=2.0)
    assert [o.id for o in found] == ["neg"]


# --- partition_objects_into_tiles ---


def test_partition_preserves_object_count_and_identity():
    objects = [
        _star(f"o{i}", Vector3(float(i), float(i * 2), 0.0)) for i in range(50)
    ]
    tiles = partition_objects_into_tiles(objects, cell_size_m=10.0)
    counted = sum(len(v) for v in tiles.values())
    assert counted == 50
    flat_ids = sorted(o.id for tile in tiles.values() for o in tile)
    assert flat_ids == sorted(o.id for o in objects)


def test_partition_tile_ids_match_cell_indices():
    obj = _star("a", Vector3(15.0, -5.0, 25.0))
    tiles = partition_objects_into_tiles([obj], cell_size_m=10.0)
    assert list(tiles) == [tile_id_for_cell(1, -1, 2)]


def test_partition_rejects_non_positive_cell_size():
    with pytest.raises(ValueError):
        partition_objects_into_tiles([], cell_size_m=0.0)
    with pytest.raises(ValueError):
        partition_objects_into_tiles([], cell_size_m=-3.0)


def test_cell_index_for_position_floors_correctly():
    assert cell_index_for_position(0.0, 0.0, 0.0, 10.0) == (0, 0, 0)
    assert cell_index_for_position(9.99, 0.0, 0.0, 10.0) == (0, 0, 0)
    assert cell_index_for_position(10.0, 0.0, 0.0, 10.0) == (1, 0, 0)
    assert cell_index_for_position(-0.1, 0.0, 0.0, 10.0) == (-1, 0, 0)


# --- TileStore ---


def test_tile_store_save_and_load_round_trip(tmp_path: Path):
    store = TileStore(str(tmp_path))
    objects = [
        _star("a", Vector3(1.0, 2.0, 3.0)),
        _star("b", Vector3(4.0, 5.0, 6.0)),
    ]
    store.save_tile("tile_0_0_0", objects)
    loaded = store.load_tile("tile_0_0_0")
    assert [o.id for o in loaded] == ["a", "b"]
    assert loaded[0].position_m == Vector3(1.0, 2.0, 3.0)
    # JSON file is human-readable
    raw = json.loads((tmp_path / "tile_0_0_0.json").read_text())
    assert raw["tile_id"] == "tile_0_0_0"
    assert len(raw["objects"]) == 2


def test_tile_store_load_missing_raises(tmp_path: Path):
    store = TileStore(str(tmp_path))
    with pytest.raises(FileNotFoundError):
        store.load_tile("tile_99_99_99")


def test_tile_store_list_tiles_sorted(tmp_path: Path):
    store = TileStore(str(tmp_path))
    store.save_tile("tile_0_0_0", [])
    store.save_tile("tile_1_0_0", [])
    store.save_tile("tile_0_2_0", [])
    assert store.list_tiles() == ["tile_0_0_0", "tile_0_2_0", "tile_1_0_0"]


def test_tile_store_rejects_empty_directory():
    with pytest.raises(ValueError):
        TileStore("")


# --- LOD ---


def test_compute_lod_weight_decreases_with_distance():
    near = compute_lod_weight(1.0)
    far = compute_lod_weight(100.0)
    assert near > far > 0.0


def test_compute_lod_weight_finite_at_zero():
    assert math.isfinite(compute_lod_weight(0.0))


def test_select_lod_returns_at_most_max_objects():
    objects = [
        _star(f"o{i}", Vector3(float(i), 0.0, 0.0)) for i in range(50)
    ]
    chosen = select_lod_objects(objects, Vector3.zero(), max_objects=10)
    assert len(chosen) == 10


def test_select_lod_returns_input_when_under_budget():
    objects = [_star(f"o{i}", Vector3(float(i), 0.0, 0.0)) for i in range(5)]
    chosen = select_lod_objects(objects, Vector3.zero(), max_objects=10)
    assert chosen == objects


def test_select_lod_zero_max_returns_empty():
    objects = [_star("a", Vector3(1.0, 0.0, 0.0))]
    assert select_lod_objects(objects, Vector3.zero(), 0) == []


def test_select_lod_empty_input_returns_empty():
    assert select_lod_objects([], Vector3.zero(), 5) == []


def test_select_lod_biased_toward_closer_objects():
    """Closer bins should be over-represented in the sample."""
    # 1000 objects spread over 10 distance bins; lower bin = closer.
    objects: list[UniverseObject] = []
    for bin_idx in range(10):
        for i in range(100):
            distance = (bin_idx + 1) * 1_000_000.0
            objects.append(
                _star(
                    f"o_{bin_idx}_{i}",
                    Vector3(distance, 0.0, 0.0),
                )
            )
    chosen = select_lod_objects(objects, Vector3.zero(), max_objects=200)
    by_bin: dict[int, int] = {}
    for obj in chosen:
        bin_idx = int(obj.position_m.x / 1_000_000.0) - 1
        by_bin[bin_idx] = by_bin.get(bin_idx, 0) + 1
    # Closest bin contributes more than the farthest one.
    assert by_bin.get(0, 0) > by_bin.get(9, 0)


def test_select_lod_is_deterministic_for_fixed_seed():
    objects = [
        _star(f"o{i}", Vector3(float(i), 0.0, 0.0)) for i in range(30)
    ]
    a = select_lod_objects(objects, Vector3.zero(), 10, seed=7)
    b = select_lod_objects(objects, Vector3.zero(), 10, seed=7)
    assert [o.id for o in a] == [o.id for o in b]


# --- Runtime streaming integration ---


def _star_grid(n: int, spacing: float = 5.0) -> list[UniverseObject]:
    out: list[UniverseObject] = []
    side = int(round(n ** (1 / 3)))
    for i in range(side):
        for j in range(side):
            for k in range(side):
                obj_id = f"g_{i}_{j}_{k}"
                out.append(
                    _star(obj_id, Vector3(i * spacing, j * spacing, k * spacing))
                )
    return out


def test_runtime_select_uses_spatial_index_when_built():
    runtime = CosmicRuntime(
        config=RuntimeConfig(active_radius_m=12.0, max_active_objects=200)
    )
    runtime.add_objects(_star_grid(125, spacing=5.0))
    runtime.build_spatial_index(cell_size_m=10.0)
    selected = runtime.select_active_objects(Vector3(10.0, 10.0, 10.0))
    assert len(selected) > 0
    for obj in selected:
        d = math.sqrt(
            (obj.position_m.x - 10.0) ** 2
            + (obj.position_m.y - 10.0) ** 2
            + (obj.position_m.z - 10.0) ** 2
        )
        assert d <= 12.0 + 1e-6


def test_runtime_enable_streaming_writes_tiles_and_clears_registry(tmp_path: Path):
    runtime = CosmicRuntime(
        config=RuntimeConfig(active_radius_m=15.0, max_active_objects=200)
    )
    objects = _star_grid(125, spacing=5.0)
    runtime.add_objects(objects)
    assert len(runtime.registry.list_objects()) == len(objects)
    store = runtime.enable_streaming(
        cell_size_m=20.0, data_directory=str(tmp_path / "tiles")
    )
    # Registry was cleared by default.
    assert len(runtime.registry.list_objects()) == 0
    tiles = store.list_tiles()
    assert tiles  # at least one tile written


def test_runtime_streaming_loads_tiles_on_demand(tmp_path: Path):
    runtime = CosmicRuntime(
        config=RuntimeConfig(active_radius_m=25.0, max_active_objects=200)
    )
    objects = _star_grid(125, spacing=5.0)
    runtime.add_objects(objects)
    runtime.enable_streaming(
        cell_size_m=20.0,
        data_directory=str(tmp_path / "tiles"),
        cache_size=4,
    )
    selected = runtime.select_active_objects(Vector3(10.0, 10.0, 10.0))
    assert len(selected) > 0
    assert runtime.last_loaded_tiles  # records which tiles got touched


def test_runtime_streaming_does_not_crash_with_no_cached_tile(tmp_path: Path):
    runtime = CosmicRuntime(
        config=RuntimeConfig(active_radius_m=10.0, max_active_objects=100)
    )
    runtime.add_objects(_star_grid(27, spacing=5.0))
    runtime.enable_streaming(
        cell_size_m=20.0, data_directory=str(tmp_path / "tiles")
    )
    # Move observer somewhere with no tiles — should return empty, not raise.
    selected = runtime.select_active_objects(Vector3(1.0e9, 1.0e9, 1.0e9))
    assert selected == []
    assert runtime.last_loaded_tiles == []


def test_runtime_streaming_keep_registry_flag(tmp_path: Path):
    runtime = CosmicRuntime(
        config=RuntimeConfig(active_radius_m=15.0, max_active_objects=200)
    )
    runtime.add_objects(_star_grid(64, spacing=5.0))
    runtime.enable_streaming(
        cell_size_m=20.0,
        data_directory=str(tmp_path / "tiles"),
        clear_registry=False,
    )
    assert len(runtime.registry.list_objects()) == 64


# --- _TileCache (LRU) ---


def test_tile_cache_evicts_oldest():
    from cosmic_engine.runtime.runtime import _TileCache

    cache = _TileCache(max_size=3)
    cache.put("a", [])
    cache.put("b", [])
    cache.put("c", [])
    assert cache.get("a") is not None
    cache.put("d", [])  # should evict "b" (oldest non-touched)
    assert cache.get("b") is None
    assert cache.get("a") is not None
    assert cache.get("c") is not None
    assert cache.get("d") is not None


def test_tile_cache_get_promotes_to_most_recent():
    from cosmic_engine.runtime.runtime import _TileCache

    cache = _TileCache(max_size=2)
    cache.put("a", [])
    cache.put("b", [])
    assert cache.get("a") is not None  # touch a
    cache.put("c", [])  # should evict b, not a
    assert cache.get("a") is not None
    assert cache.get("b") is None
    assert cache.get("c") is not None


def test_tile_cache_rejects_bad_size():
    from cosmic_engine.runtime.runtime import _TileCache

    with pytest.raises(ValueError):
        _TileCache(max_size=0)
