"""Demo: stream a 200,000-galaxy synthetic catalog through tiles + LOD."""

from __future__ import annotations

import shutil
import time
from pathlib import Path

from cosmic_engine.core.units import LIGHTYEAR_IN_METERS
from cosmic_engine.core.vector import Vector3
from cosmic_engine.data.galaxy_catalog import generate_synthetic_galaxy_catalog
from cosmic_engine.runtime import CosmicRuntime, RuntimeConfig


_REPO_ROOT = Path(__file__).resolve().parent.parent
_TILE_DIR = _REPO_ROOT / "outputs" / "tiles"
_GALAXY_COUNT = 200_000
_RADIUS_M = 1.0e25
_TILE_CELL_SIZE_M = 5.0e23  # ~16 Mpc per tile
_ACTIVE_RADIUS_M = 1.5e24   # ~50 Mpc
_MAX_ACTIVE = 5_000


def _observer_path(steps: int) -> list[Vector3]:
    """A small straight-line walk through the synthetic field."""
    span = _RADIUS_M * 0.6
    return [
        Vector3(
            (i / max(steps - 1, 1) - 0.5) * span,
            (i / max(steps - 1, 1) - 0.5) * span,
            0.0,
        )
        for i in range(steps)
    ]


def main() -> None:
    if _TILE_DIR.exists():
        shutil.rmtree(_TILE_DIR)
    _TILE_DIR.mkdir(parents=True, exist_ok=True)

    print(f"generating {_GALAXY_COUNT} synthetic galaxies...")
    t_gen = time.perf_counter()
    galaxies = generate_synthetic_galaxy_catalog(_GALAXY_COUNT, _RADIUS_M, seed=42)
    print(f"generation time      : {(time.perf_counter() - t_gen) * 1000:.1f} ms")

    runtime = CosmicRuntime(
        config=RuntimeConfig(
            enable_physics=False,
            enable_perception=False,
            active_radius_m=_ACTIVE_RADIUS_M,
            max_active_objects=_MAX_ACTIVE,
        )
    )
    runtime.add_objects(galaxies)
    print(f"in-memory registry   : {len(runtime.registry.list_objects())} objects")

    print(f"partitioning into tiles (cell={_TILE_CELL_SIZE_M:.2e} m)...")
    t_part = time.perf_counter()
    store = runtime.enable_streaming(
        cell_size_m=_TILE_CELL_SIZE_M,
        data_directory=str(_TILE_DIR),
        cache_size=24,
    )
    print(f"partition+persist    : {(time.perf_counter() - t_part) * 1000:.1f} ms")
    tiles = store.list_tiles()
    total_tile_bytes = sum(
        (Path(_TILE_DIR) / f"{t}.json").stat().st_size for t in tiles
    )
    print(f"tiles on disk        : {len(tiles)}")
    print(f"total tile bytes     : {total_tile_bytes / (1024 * 1024):.2f} MiB")
    print(f"in-memory registry   : {len(runtime.registry.list_objects())}  (cleared)")
    print()

    print(
        f"{'step':<5} {'observer (Mly)':<26} "
        f"{'tiles':>8} {'cache':>8} {'active':>8} {'sel ms':>8}"
    )
    print("-" * 70)

    observers = _observer_path(steps=5)
    for i, obs_pos in enumerate(observers):
        t0 = time.perf_counter()
        active = runtime.select_active_objects(obs_pos)
        elapsed = (time.perf_counter() - t0) * 1000.0
        cache_size = len(runtime._tile_cache) if runtime._tile_cache else 0
        pos_mly = (
            obs_pos.x / LIGHTYEAR_IN_METERS / 1_000_000.0,
            obs_pos.y / LIGHTYEAR_IN_METERS / 1_000_000.0,
            obs_pos.z / LIGHTYEAR_IN_METERS / 1_000_000.0,
        )
        print(
            f"{i:<5} "
            f"({pos_mly[0]:>+5.0f},{pos_mly[1]:>+5.0f},{pos_mly[2]:>+5.0f})  "
            f"{len(runtime.last_loaded_tiles):>8} "
            f"{cache_size:>8} "
            f"{len(active):>8} "
            f"{elapsed:>8.1f}"
        )

    print()
    print(f"total dataset        : {_GALAXY_COUNT} galaxies")
    print(f"tile count           : {len(tiles)}")
    print(f"max active per frame : {_MAX_ACTIVE} (LOD-bounded)")
    print(f"tile cache cap       : 24 tiles")


if __name__ == "__main__":
    main()
