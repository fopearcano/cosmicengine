"""Demo: render a galaxy field around a black hole, off / weak / strong."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from ai_viewer.frame_buffer import FrameBuffer
from ai_viewer.neural_field import (
    BlackHole,
    GaussianSplatRenderer,
    apply_black_hole_to_points,
    apply_lensing_to_points,
    build_gaussian_field_from_galaxy_batch,
)

from cosmic_engine.core.vector import Vector3
from cosmic_engine.data.galaxy_catalog import generate_synthetic_galaxy_catalog
from cosmic_engine.rendering import (
    SimpleCamera,
    build_galaxy_field_batch,
)


_REPO_ROOT = Path(__file__).resolve().parent.parent
_OUT_DIR = _REPO_ROOT / "outputs" / "viewer"

# For the demo we use deliberately oversized masses so the visual
# deflection is observable at this catalog scale (Gpc impact parameters
# vs a 256x256 image — physical supermassive masses produce sub-pixel
# bends). The math is the same; only the parameters are exaggerated.
_BH_POSITION = np.array([0.0, 0.0, 0.0], dtype=np.float64)
_BH_MASS_KG = 8.0e50   # event horizon ~ 1.2e24 m: absorbs the inner cluster
_LENS_MASS_KG = 1.0e50  # weaker lens so direction bends a few degrees


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

    t_build = time.perf_counter()
    batch = build_galaxy_field_batch(galaxies, camera)
    base_points = build_gaussian_field_from_galaxy_batch(batch, sigma_scale=1.0)
    print(
        f"gaussian points    : {len(base_points)} "
        f"(built in {(time.perf_counter() - t_build) * 1000:.1f} ms)"
    )

    observer_pos = np.array(
        [camera.position_m.x, camera.position_m.y, camera.position_m.z]
    )

    bh = BlackHole(_BH_POSITION, _BH_MASS_KG)
    print(
        f"black hole         : mass={bh.mass_kg:.2e} kg  "
        f"R_s={bh.schwarzschild_radius():.3e} m"
    )

    renderer = GaussianSplatRenderer(width, height, camera)

    # Mode 1: no GR
    t0 = time.perf_counter()
    img_no_gr = renderer.render(base_points)
    no_gr_seconds = time.perf_counter() - t0
    out_no_gr = _OUT_DIR / "output_no_gr.ppm"
    _save(img_no_gr, out_no_gr)

    # Mode 2: weak lensing (smaller mass, no horizon culling)
    t0 = time.perf_counter()
    lensed = apply_lensing_to_points(
        base_points, _BH_POSITION, _LENS_MASS_KG, observer_pos
    )
    img_lensing = renderer.render(lensed)
    lensing_seconds = time.perf_counter() - t0
    out_lensing = _OUT_DIR / "output_lensing.ppm"
    _save(img_lensing, out_lensing)

    # Mode 3: strong-field black hole (horizon culling + maximum lensing)
    t0 = time.perf_counter()
    bh_points = apply_black_hole_to_points(base_points, bh, observer_pos)
    img_bh = renderer.render(bh_points)
    bh_seconds = time.perf_counter() - t0
    out_bh = _OUT_DIR / "output_blackhole.ppm"
    _save(img_bh, out_bh)

    print()
    print(f"{'mode':<18}  {'points':>8}  {'time ms':>8}  output")
    print("-" * 70)
    print(
        f"{'no_gr':<18}  {len(base_points):>8}  "
        f"{no_gr_seconds * 1000:>8.1f}  {out_no_gr}"
    )
    print(
        f"{'weak_lensing':<18}  {len(lensed):>8}  "
        f"{lensing_seconds * 1000:>8.1f}  {out_lensing}"
    )
    print(
        f"{'strong_blackhole':<18}  {len(bh_points):>8}  "
        f"{bh_seconds * 1000:>8.1f}  {out_bh}"
    )

    # Quick stability check: how many points sit inside the event horizon?
    inside = sum(1 for p in base_points if bh.is_inside_event_horizon(p.position))
    print()
    print(f"points absorbed by event horizon : {inside}")
    print(f"points surviving black-hole pass : {len(bh_points)}")


if __name__ == "__main__":
    main()
