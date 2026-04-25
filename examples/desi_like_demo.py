"""Demo: load a tiny DESI-like CSV, project, and render a starfield-style PPM."""

from __future__ import annotations

from pathlib import Path

from cosmic_engine.core.registry import UniverseRegistry
from cosmic_engine.core.vector import Vector3
from cosmic_engine.data.desi_like_catalog import load_desi_into_registry
from cosmic_engine.physics.cosmology import redshift_to_distance_lightyears
from cosmic_engine.rendering import (
    SimpleCamera,
    build_galaxy_field_batch,
    render_galaxy_batch_to_ppm,
)


_REPO_ROOT = Path(__file__).resolve().parent.parent
_CATALOG = _REPO_ROOT / "data" / "sample_desi_like.csv"
_OUTPUT = Path(__file__).resolve().parent / "output_desi_like.ppm"


def main() -> None:
    registry = UniverseRegistry()
    load_desi_into_registry(str(_CATALOG), registry)
    galaxies = registry.list_objects()

    camera = SimpleCamera(
        position_m=Vector3.zero(),
        forward=Vector3(1.0, 0.0, 0.0),
        up=Vector3(0.0, 0.0, 1.0),
        fov_degrees=170.0,
        image_width=512,
        image_height=512,
    )
    batch = build_galaxy_field_batch(galaxies, camera)
    render_galaxy_batch_to_ppm(batch, camera, str(_OUTPUT))

    redshifts = [g.redshift_z for g in galaxies]
    distances_m = [
        (
            g.position_m.x ** 2 + g.position_m.y ** 2 + g.position_m.z ** 2
        )
        ** 0.5
        for g in galaxies
    ]
    z_min, z_max = min(redshifts), max(redshifts)
    d_min, d_max = min(distances_m), max(distances_m)

    print(f"galaxies loaded     : {len(galaxies)}")
    print(f"redshift range      : [{z_min:.3f}, {z_max:.3f}]")
    print(f"distance range (m)  : [{d_min:.3e}, {d_max:.3e}]")
    print(
        f"distance range (ly) : "
        f"[{redshift_to_distance_lightyears(z_min):.3e}, "
        f"{redshift_to_distance_lightyears(z_max):.3e}]"
    )
    print(f"output path         : {_OUTPUT}")


if __name__ == "__main__":
    main()
