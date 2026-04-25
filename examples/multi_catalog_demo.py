"""Demo: ingest Gaia + SDSS + DESI + JPL into one registry and render."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from cosmic_engine.core.object_types import CosmicObjectType
from cosmic_engine.core.registry import UniverseRegistry
from cosmic_engine.core.vector import Vector3
from cosmic_engine.data.jpl_ephemeris import load_jpl_into_registry
from cosmic_engine.data.sources import DataSource, load_catalog_into_registry
from cosmic_engine.rendering import (
    SimpleCamera,
    build_galaxy_field_batch,
    build_star_photon_field,
    render_photon_field_to_ppm,
)
from cosmic_engine.rendering.photon_field import PhotonSample


_REPO_ROOT = Path(__file__).resolve().parent.parent
_GAIA = _REPO_ROOT / "data" / "sample_gaia_like.csv"
_SDSS = _REPO_ROOT / "data" / "sample_sdss_like.csv"
_DESI = _REPO_ROOT / "data" / "sample_desi.csv"
_OUTPUT = Path(__file__).resolve().parent / "output_multi_catalog.ppm"


def _galaxy_batch_to_samples(batch) -> list[PhotonSample]:
    samples: list[PhotonSample] = []
    for i in range(len(batch)):
        r, g, b = batch.colors_rgb[i]
        samples.append(
            PhotonSample(
                object_id=batch.object_ids[i],
                name=batch.names[i],
                object_type="galaxy",
                direction=Vector3(
                    float(batch.directions[i, 0]),
                    float(batch.directions[i, 1]),
                    float(batch.directions[i, 2]),
                ),
                distance_m=float(batch.distances_m[i]),
                apparent_brightness=float(batch.brightness[i]),
                color_rgb=(
                    max(0, min(255, int(round(float(r))))),
                    max(0, min(255, int(round(float(g))))),
                    max(0, min(255, int(round(float(b))))),
                ),
                truth_level=batch.truth_levels[i],
            )
        )
    return samples


def main() -> None:
    registry = UniverseRegistry()
    load_catalog_into_registry(str(_GAIA), DataSource.GAIA, registry)
    load_catalog_into_registry(str(_SDSS), DataSource.SDSS, registry)
    load_catalog_into_registry(str(_DESI), DataSource.DESI, registry)
    load_jpl_into_registry(registry)

    objects = registry.list_objects()
    by_type = Counter(o.object_type.value for o in objects)
    by_source = Counter(o.source for o in objects)

    print(f"total objects        : {len(objects)}")
    print()
    print("counts by type:")
    for k, v in sorted(by_type.items()):
        print(f"  {k:<12} {v}")
    print()
    print("counts by source:")
    for k, v in sorted(by_source.items(), key=lambda kv: (kv[0] or "")):
        print(f"  {k:<12} {v}")

    stars = [o for o in objects if o.object_type is CosmicObjectType.STAR]
    galaxies = [o for o in objects if o.object_type is CosmicObjectType.GALAXY]

    camera = SimpleCamera(
        position_m=Vector3.zero(),
        forward=Vector3(0.0, 1.0, 0.0),
        up=Vector3(0.0, 0.0, 1.0),
        fov_degrees=170.0,
        image_width=512,
        image_height=512,
    )
    star_samples = build_star_photon_field(stars, camera)
    galaxy_batch = build_galaxy_field_batch(galaxies, camera)
    galaxy_samples = _galaxy_batch_to_samples(galaxy_batch)
    all_samples = star_samples + galaxy_samples

    render_photon_field_to_ppm(all_samples, camera, str(_OUTPUT))

    print()
    print(f"star photon samples  : {len(star_samples)}")
    print(f"galaxy field batch   : {len(galaxy_batch)}")
    print(f"combined samples     : {len(all_samples)}")
    print(f"output path          : {_OUTPUT}")


if __name__ == "__main__":
    main()
