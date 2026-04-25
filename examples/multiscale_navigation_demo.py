"""Demo: walk an observer across galaxy / star / solar-system scales."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from ai_viewer.frame_buffer import FrameBuffer
from ai_viewer.neural_field import (
    GaussianSplatRenderer,
    build_gaussian_field_from_galaxy_batch,
)

from cosmic_engine.core.object_types import CosmicObjectType
from cosmic_engine.core.units import (
    AU_IN_METERS,
    LIGHTYEAR_IN_METERS,
    PARSEC_IN_METERS,
)
from cosmic_engine.core.vector import Vector3
from cosmic_engine.data.galaxy_catalog import generate_synthetic_galaxy_catalog
from cosmic_engine.multiscale import DEFAULT_ZONES, ScaleManager
from cosmic_engine.physics.solar_system import create_solar_system_objects
from cosmic_engine.rendering import (
    SimpleCamera,
    build_galaxy_field_batch,
    build_star_photon_field,
    render_photon_field_to_ppm,
)
from cosmic_engine.runtime import CosmicRuntime, RuntimeConfig


_REPO_ROOT = Path(__file__).resolve().parent.parent
_OUT_DIR = _REPO_ROOT / "outputs" / "viewer"


def _save_ppm(image: np.ndarray, path: Path) -> None:
    fb = FrameBuffer(image.shape[1], image.shape[0])
    fb.pixels = image
    path.parent.mkdir(parents=True, exist_ok=True)
    fb.save_ppm(str(path))


def _load_mixed_dataset(runtime: CosmicRuntime) -> None:
    runtime.load_sample_data()  # bundled Gaia + SDSS + DESI + JPL
    # Add a synthetic galaxy cloud so intergalactic scales aren't bare.
    galaxies = generate_synthetic_galaxy_catalog(
        2_000, radius_m=1.0e25, seed=42
    )
    runtime.add_objects(galaxies)
    # Add a fresh solar-system snapshot in case load_sample_data was
    # already called (the JPL placeholder is idempotent on duplicates).
    runtime.add_objects(create_solar_system_objects(2_451_545.0))


def _render_zone(
    rep: dict,
    runtime: CosmicRuntime,
    camera: SimpleCamera,
    out_path: Path,
) -> int:
    """Pick a renderer based on the representation type and write a PPM.

    Returns the number of objects actually drawn.
    """
    rep_type = rep["type"]
    objects = rep["objects"]
    if rep_type in ("galaxy_field", "blended"):
        # Both galaxies and stars render via the photon path; if the
        # representation contains either, we just render whatever we
        # have.
        galaxies = [
            o for o in objects if o.object_type is CosmicObjectType.GALAXY
        ]
        stars = [
            o for o in objects if o.object_type is CosmicObjectType.STAR
        ]
        all_samples = []
        if galaxies:
            batch = build_galaxy_field_batch(galaxies, camera)
            points = build_gaussian_field_from_galaxy_batch(batch)
            renderer = GaussianSplatRenderer(
                camera.image_width, camera.image_height, camera
            )
            image = renderer.render(points)
            _save_ppm(image, out_path)
            return len(galaxies)
        if stars:
            samples = build_star_photon_field(stars, camera)
            render_photon_field_to_ppm(samples, camera, str(out_path))
            return len(samples)
    if rep_type == "star_field":
        stars = [
            o for o in objects if o.object_type is CosmicObjectType.STAR
        ]
        samples = build_star_photon_field(stars, camera)
        render_photon_field_to_ppm(samples, camera, str(out_path))
        return len(samples)
    if rep_type == "nbody":
        # No native renderer for an N-body system at this scale yet —
        # write an empty frame and rely on the printout for state.
        empty = np.zeros((camera.image_height, camera.image_width, 3),
                         dtype=np.uint8)
        _save_ppm(empty, out_path)
        return 0
    # density_field / neural_field placeholders
    empty = np.zeros((camera.image_height, camera.image_width, 3),
                     dtype=np.uint8)
    _save_ppm(empty, out_path)
    return 0


def main() -> None:
    width, height = 256, 256

    runtime = CosmicRuntime(
        config=RuntimeConfig(
            enable_physics=False,
            enable_perception=False,
            active_radius_m=1.0e30,
            max_active_objects=20_000,
        )
    )
    _load_mixed_dataset(runtime)
    print(f"loaded objects     : {len(runtime.registry.list_objects())}")

    runtime.enable_multiscale(
        ScaleManager(list(DEFAULT_ZONES)), blend_width=0.1
    )
    print(f"multiscale zones   : {len(runtime.scale_manager.zones)}")

    # Walk from far intergalactic distances down to the solar system.
    observer_distances_m = [
        ("intergalactic", 5.0e25),
        ("intergalactic_inner", 5.0e21),
        ("interstellar", 5.0e17),
        ("solar_system", 5.0e10),
        ("microscale", 1.0e3),
    ]

    print()
    print(
        f"{'step':<22} {'dist (m)':>10}  {'zone':<14} "
        f"{'rep_type':<14} {'objects':>7}"
    )
    print("-" * 80)

    camera = SimpleCamera(
        position_m=Vector3.zero(),
        forward=Vector3(0.0, 1.0, 0.0),
        up=Vector3(0.0, 0.0, 1.0),
        fov_degrees=160.0,
        image_width=width,
        image_height=height,
    )

    transitions = 0
    last_rep_type = None
    for label, distance in observer_distances_m:
        # We don't actually move the observer; instead we vary the
        # observer position so the median active-object distance
        # lands in different zones. Camera stays at origin for
        # consistent rendering.
        observer = Vector3(0.0, -distance, 0.0)
        rep = runtime.get_multiscale_scene(observer)
        zone_label = (
            f"{rep.get('primary_zone')}->{rep.get('secondary_zone')}"
            if rep.get("blended")
            else rep.get("zone_name", "?")
        )
        rep_type = rep["type"]
        if last_rep_type is not None and rep_type != last_rep_type:
            transitions += 1
        last_rep_type = rep_type

        out = _OUT_DIR / f"output_multiscale_{label}.ppm"
        drawn = _render_zone(rep, runtime, camera, out)
        print(
            f"{label:<22} {distance:>10.1e}  {zone_label:<14} "
            f"{rep_type:<14} {drawn:>7}"
        )

    print()
    print(f"transitions observed : {transitions}")
    print(f"frames in            : {_OUT_DIR}")


if __name__ == "__main__":
    main()
