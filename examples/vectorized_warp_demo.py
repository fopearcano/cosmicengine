"""Demo: compare scalar vs vectorized photon-field and perception paths."""

from __future__ import annotations

import time
from pathlib import Path

from cosmic_engine.core.coordinates import ra_dec_distance_to_cartesian
from cosmic_engine.core.object_types import CosmicObjectType
from cosmic_engine.core.registry import UniverseRegistry
from cosmic_engine.core.truth import TruthLevel
from cosmic_engine.core.units import LIGHTYEAR_IN_METERS, SPEED_OF_LIGHT_M_S
from cosmic_engine.core.universe_object import UniverseObject
from cosmic_engine.core.vector import Vector3
from cosmic_engine.data.star_catalog import load_star_catalog_into_registry
from cosmic_engine.perception import ObserverState, transform_photon_field
from cosmic_engine.perception.vectorized_transform import (
    transform_photon_field_batch,
)
from cosmic_engine.rendering import (
    SimpleCamera,
    build_star_photon_field,
    build_star_photon_field_batch,
    render_photon_batch_to_ppm,
)

_REPO_ROOT = Path(__file__).resolve().parent.parent
_CATALOG = _REPO_ROOT / "data" / "sample_stars.csv"
_OUTPUT = Path(__file__).resolve().parent / "output_vectorized_warp.ppm"
_SPECTRAL_CYCLE = ["O5V", "B1V", "A1V", "F5IV", "G2V", "K0III", "M1Ia"]


def _synthetic_star(i: int) -> UniverseObject:
    """Deterministic procedural star for benchmarking; no randomness."""
    ra = (i * 37.9) % 360.0
    dec = ((i * 73.1) % 180.0) - 90.0
    distance_ly = 10.0 + float(i % 500)
    position = ra_dec_distance_to_cartesian(
        ra, dec, distance_ly * LIGHTYEAR_IN_METERS
    )
    magnitude = float((i % 20) - 5)  # range [-5, 14]
    return UniverseObject(
        id=f"syn_{i}",
        name=f"Syn {i}",
        object_type=CosmicObjectType.STAR,
        position_m=position,
        velocity_m_s=Vector3.zero(),
        truth_level=TruthLevel.PROCEDURAL_APPROXIMATION,
        source="synthetic",
        spectral_class=_SPECTRAL_CYCLE[i % len(_SPECTRAL_CYCLE)],
        metadata={"apparent_magnitude": magnitude},
    )


def main() -> None:
    registry = UniverseRegistry()
    load_star_catalog_into_registry(str(_CATALOG), registry)

    synthetic_count = 10_000
    objects = registry.list_objects() + [
        _synthetic_star(i) for i in range(synthetic_count)
    ]
    print(f"total catalog: {len(objects)} stars (incl. {synthetic_count} synthetic)")

    camera = SimpleCamera(
        position_m=Vector3.zero(),
        forward=Vector3(0.0, 1.0, 0.0),
        up=Vector3(0.0, 0.0, 1.0),
        fov_degrees=140.0,
        image_width=512,
        image_height=512,
    )
    observer = ObserverState(
        position_m=Vector3.zero(),
        velocity_m_s=Vector3(0.0, 0.5 * SPEED_OF_LIGHT_M_S, 0.0),
        forward=Vector3(0.0, 1.0, 0.0),
        up=Vector3(0.0, 0.0, 1.0),
        warp_factor=5.0,
    )

    t0 = time.perf_counter()
    scalar_samples = build_star_photon_field(objects, camera)
    t1 = time.perf_counter()
    scalar_transformed = transform_photon_field(scalar_samples, observer)
    t2 = time.perf_counter()

    t3 = time.perf_counter()
    batch = build_star_photon_field_batch(objects, camera)
    t4 = time.perf_counter()
    warped_batch = transform_photon_field_batch(batch, observer)
    t5 = time.perf_counter()

    render_photon_batch_to_ppm(warped_batch, camera, str(_OUTPUT))

    print()
    print(f"scalar build     : {(t1 - t0) * 1000:8.2f} ms  ({len(scalar_samples)} samples)")
    print(f"scalar transform : {(t2 - t1) * 1000:8.2f} ms")
    print(f"batch  build     : {(t4 - t3) * 1000:8.2f} ms  ({len(batch)} samples)")
    print(f"batch  transform : {(t5 - t4) * 1000:8.2f} ms")
    print()
    print(f"output path      : {_OUTPUT}")


if __name__ == "__main__":
    main()
