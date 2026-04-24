"""Demo: render the bundled sample star catalog to a PPM starfield."""

from __future__ import annotations

from pathlib import Path

from cosmic_engine.core.registry import UniverseRegistry
from cosmic_engine.core.vector import Vector3
from cosmic_engine.data.star_catalog import load_star_catalog_into_registry
from cosmic_engine.rendering import (
    SimpleCamera,
    build_star_photon_field,
    render_photon_field_to_ppm,
)

_REPO_ROOT = Path(__file__).resolve().parent.parent
_CATALOG = _REPO_ROOT / "data" / "sample_stars.csv"
_OUTPUT = Path(__file__).resolve().parent / "output_starfield.ppm"


def main() -> None:
    registry = UniverseRegistry()
    load_star_catalog_into_registry(str(_CATALOG), registry)
    objects = registry.list_objects()

    camera = SimpleCamera(
        position_m=Vector3.zero(),
        forward=Vector3(0.0, 1.0, 0.0),
        up=Vector3(0.0, 0.0, 1.0),
        fov_degrees=140.0,
        image_width=512,
        image_height=512,
    )

    samples = build_star_photon_field(objects, camera)
    render_photon_field_to_ppm(samples, camera, str(_OUTPUT))

    print(f"loaded objects: {len(objects)}")
    print(f"photon samples: {len(samples)}")
    print(f"output path:    {_OUTPUT}")


if __name__ == "__main__":
    main()
