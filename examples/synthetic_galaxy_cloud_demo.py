"""Demo: generate a synthetic galaxy point cloud, render, and grid it."""

from __future__ import annotations

import time
from pathlib import Path

from cosmic_engine.core.vector import Vector3
from cosmic_engine.data.galaxy_catalog import generate_synthetic_galaxy_catalog
from cosmic_engine.rendering import (
    SimpleCamera,
    build_galaxy_field_batch,
    galaxy_batch_to_density_grid,
    render_galaxy_batch_to_ppm,
)

_OUTPUT = Path(__file__).resolve().parent / "output_synthetic_galaxies.ppm"


def main() -> None:
    count = 50_000
    radius_m = 1.0e25  # ~0.3 Gpc; cosmological only in scale, not in physics

    t0 = time.perf_counter()
    galaxies = generate_synthetic_galaxy_catalog(count, radius_m, seed=42)
    t1 = time.perf_counter()

    camera = SimpleCamera(
        position_m=Vector3(0.0, -radius_m * 1.5, 0.0),
        forward=Vector3(0.0, 1.0, 0.0),
        up=Vector3(0.0, 0.0, 1.0),
        fov_degrees=90.0,
        image_width=512,
        image_height=512,
    )
    batch = build_galaxy_field_batch(galaxies, camera)
    t2 = time.perf_counter()

    render_galaxy_batch_to_ppm(batch, camera, str(_OUTPUT))
    t3 = time.perf_counter()

    grid = galaxy_batch_to_density_grid(batch, grid_size=64, extent_m=radius_m * 1.2)
    t4 = time.perf_counter()

    z_min = float(batch.redshifts_z.min())
    z_max = float(batch.redshifts_z.max())
    d_min = float(batch.distances_m.min())
    d_max = float(batch.distances_m.max())

    print(f"generated galaxies : {len(galaxies)}")
    print(f"visible batch size : {len(batch)}")
    print(f"distance range m   : [{d_min:.3e}, {d_max:.3e}]")
    print(f"redshift range     : [{z_min:.3f}, {z_max:.3f}]")
    print(f"density grid shape : {grid.shape}")
    print(f"density grid max   : {grid.max():.3f}")
    print(f"density occupied   : {int((grid > 0).sum())} of {grid.size} cells")
    print()
    print(f"generate           : {(t1 - t0) * 1000:8.2f} ms")
    print(f"build batch        : {(t2 - t1) * 1000:8.2f} ms")
    print(f"render PPM         : {(t3 - t2) * 1000:8.2f} ms")
    print(f"density grid       : {(t4 - t3) * 1000:8.2f} ms")
    print()
    print(f"output path        : {_OUTPUT}")


if __name__ == "__main__":
    main()
