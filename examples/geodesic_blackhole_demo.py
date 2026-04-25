"""Demo: render a galaxy field with no GR, lensing, and geodesic ray marching."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from ai_viewer.frame_buffer import FrameBuffer
from ai_viewer.neural_field import (
    BlackHole,
    GaussianSplatRenderer,
    GeodesicRayMarcher,
    apply_lensing_to_points,
    build_gaussian_field_from_galaxy_batch,
    trace_points_through_geodesic,
)

from cosmic_engine.core.vector import Vector3
from cosmic_engine.data.galaxy_catalog import generate_synthetic_galaxy_catalog
from cosmic_engine.rendering import (
    SimpleCamera,
    build_galaxy_field_batch,
)


_REPO_ROOT = Path(__file__).resolve().parent.parent
_OUT_DIR = _REPO_ROOT / "outputs" / "viewer"

_BH_POSITION = np.array([0.0, 0.0, 0.0], dtype=np.float64)
_BH_MASS_KG = 4.0e50      # event horizon ~ 5.9e23 m
_LENS_MASS_KG = 4.0e50    # same mass for fair comparison
_GEODESIC_STEPS = 12
_GEODESIC_STEP_SIZE_M = 2.0e24


def _save(image: np.ndarray, path: Path) -> None:
    fb = FrameBuffer(image.shape[1], image.shape[0])
    fb.pixels = image
    path.parent.mkdir(parents=True, exist_ok=True)
    fb.save_ppm(str(path))


def main() -> None:
    galaxy_count = 20_000
    radius_m = 1.0e25
    width, height = 256, 256

    print(f"generating {galaxy_count} synthetic galaxies...")
    galaxies = generate_synthetic_galaxy_catalog(galaxy_count, radius_m, seed=42)

    camera = SimpleCamera(
        position_m=Vector3(0.0, -radius_m * 1.5, 0.0),
        forward=Vector3(0.0, 1.0, 0.0),
        up=Vector3(0.0, 0.0, 1.0),
        fov_degrees=90.0,
        image_width=width,
        image_height=height,
    )
    observer_pos = np.array(
        [camera.position_m.x, camera.position_m.y, camera.position_m.z]
    )

    t_build = time.perf_counter()
    batch = build_galaxy_field_batch(galaxies, camera)
    base_points = build_gaussian_field_from_galaxy_batch(batch, sigma_scale=1.0)
    print(
        f"gaussian points    : {len(base_points)}  "
        f"(built in {(time.perf_counter() - t_build) * 1000:.1f} ms)"
    )

    bh = BlackHole(_BH_POSITION, _BH_MASS_KG)
    print(
        f"black hole         : mass={bh.mass_kg:.2e} kg  "
        f"R_s={bh.schwarzschild_radius():.3e} m"
    )
    marcher = GeodesicRayMarcher(
        bh, step_size=_GEODESIC_STEP_SIZE_M, max_steps=_GEODESIC_STEPS
    )

    renderer = GaussianSplatRenderer(width, height, camera)

    # Mode 1: no GR
    t0 = time.perf_counter()
    img_none = renderer.render(base_points)
    none_seconds = time.perf_counter() - t0
    out_none = _OUT_DIR / "output_geodesic_none.ppm"
    _save(img_none, out_none)

    # Mode 2: simple lensing approximation (Phase 28)
    t0 = time.perf_counter()
    lensed = apply_lensing_to_points(
        base_points, _BH_POSITION, _LENS_MASS_KG, observer_pos
    )
    img_lensing = renderer.render(lensed)
    lensing_seconds = time.perf_counter() - t0
    out_lensing = _OUT_DIR / "output_geodesic_lensing.ppm"
    _save(img_lensing, out_lensing)

    # Mode 3: full geodesic ray marching (Phase 29)
    t0 = time.perf_counter()
    geodesic_points = trace_points_through_geodesic(
        base_points, marcher, observer_pos
    )
    img_geodesic = renderer.render(geodesic_points)
    geodesic_seconds = time.perf_counter() - t0
    out_geodesic = _OUT_DIR / "output_geodesic_full.ppm"
    _save(img_geodesic, out_geodesic)

    print()
    print(f"{'mode':<18}  {'points':>8}  {'time ms':>8}  output")
    print("-" * 75)
    print(
        f"{'no_gr':<18}  {len(base_points):>8}  "
        f"{none_seconds * 1000:>8.1f}  {out_none}"
    )
    print(
        f"{'lensing':<18}  {len(lensed):>8}  "
        f"{lensing_seconds * 1000:>8.1f}  {out_lensing}"
    )
    print(
        f"{'geodesic':<18}  {len(geodesic_points):>8}  "
        f"{geodesic_seconds * 1000:>8.1f}  {out_geodesic}"
    )

    absorbed_geodesic = len(base_points) - len(geodesic_points)
    print()
    print(f"geodesic steps used      : {_GEODESIC_STEPS}")
    print(f"geodesic step size (m)   : {_GEODESIC_STEP_SIZE_M:.2e}")
    print(f"absorbed by event horizon: {absorbed_geodesic}")


if __name__ == "__main__":
    main()
