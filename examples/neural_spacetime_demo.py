"""Demo: trace ray geodesics with analytical vs neural curvature."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from ai_viewer.frame_buffer import FrameBuffer
from ai_viewer.neural_field import (
    BlackHole,
    GaussianSplatRenderer,
    GeodesicRayMarcher,
    build_gaussian_field_from_galaxy_batch,
    trace_points_through_geodesic,
)

from cosmic_engine.ai import ONNXSpacetimeField
from cosmic_engine.core.vector import Vector3
from cosmic_engine.data.galaxy_catalog import generate_synthetic_galaxy_catalog
from cosmic_engine.rendering import (
    SimpleCamera,
    build_galaxy_field_batch,
)


_REPO_ROOT = Path(__file__).resolve().parent.parent
_OUT_DIR = _REPO_ROOT / "outputs" / "viewer"
_DEFAULT_MODEL = _REPO_ROOT / "data" / "spacetime_field_identity.onnx"


def _save(image: np.ndarray, path: Path) -> None:
    fb = FrameBuffer(image.shape[1], image.shape[0])
    fb.pixels = image
    path.parent.mkdir(parents=True, exist_ok=True)
    fb.save_ppm(str(path))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default=str(_DEFAULT_MODEL),
        help="Path to a SpacetimeFieldModel ONNX. Pass a missing path "
        "to demo the analytical-only fallback.",
    )
    args = parser.parse_args()

    galaxy_count = 20_000
    radius_m = 1.0e25
    width, height = 256, 256

    print(f"generating {galaxy_count} synthetic galaxies...")
    galaxies = generate_synthetic_galaxy_catalog(
        galaxy_count, radius_m, seed=42
    )
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

    batch = build_galaxy_field_batch(galaxies, camera)
    base_points = build_gaussian_field_from_galaxy_batch(batch, sigma_scale=1.0)
    print(f"gaussian points    : {len(base_points)}")

    bh = BlackHole((0.0, 0.0, 0.0), 4.0e50)
    print(
        f"black hole         : mass={bh.mass_kg:.2e} kg  "
        f"R_s={bh.schwarzschild_radius():.3e} m"
    )

    spacetime_model = None
    fallback_used = False
    if args.model:
        try:
            spacetime_model = ONNXSpacetimeField(args.model)
            print(
                f"spacetime model    : {args.model} "
                f"(confidence={spacetime_model.confidence():.2f})"
            )
        except Exception as exc:
            spacetime_model = None
            fallback_used = True
            print(f"spacetime model    : unavailable ({exc})")
            print("                     (analytical path is the fallback)")
    else:
        print("spacetime model    : --model omitted; analytical only")

    renderer = GaussianSplatRenderer(width, height, camera)
    marcher_analytic = GeodesicRayMarcher(
        bh, step_size=2.0e24, max_steps=12, spacetime_model=None
    )
    marcher_neural = GeodesicRayMarcher(
        bh,
        step_size=2.0e24,
        max_steps=12,
        spacetime_model=spacetime_model,
    )

    # Mode 1: analytical only
    t0 = time.perf_counter()
    analytic_points = trace_points_through_geodesic(
        base_points, marcher_analytic, observer_pos
    )
    analytic_image = renderer.render(analytic_points)
    analytic_seconds = time.perf_counter() - t0
    out_analytic = _OUT_DIR / "output_spacetime_analytic.ppm"
    _save(analytic_image, out_analytic)

    # Mode 2: neural (or fallback to analytical if model is None)
    t0 = time.perf_counter()
    neural_points = trace_points_through_geodesic(
        base_points, marcher_neural, observer_pos
    )
    neural_image = renderer.render(neural_points)
    neural_seconds = time.perf_counter() - t0
    out_neural = _OUT_DIR / "output_spacetime_neural.ppm"
    _save(neural_image, out_neural)

    print()
    print(f"{'mode':<14}  {'points':>8}  {'time ms':>8}  output")
    print("-" * 70)
    print(
        f"{'analytical':<14}  {len(analytic_points):>8}  "
        f"{analytic_seconds * 1000:>8.1f}  {out_analytic}"
    )
    print(
        f"{'neural':<14}  {len(neural_points):>8}  "
        f"{neural_seconds * 1000:>8.1f}  {out_neural}"
    )
    print()
    print(f"fallback triggered : {fallback_used or spacetime_model is None}")
    if spacetime_model is not None:
        print(
            f"model confidence   : {spacetime_model.confidence():.2f}  "
            f"(last_error={spacetime_model.last_error})"
        )


if __name__ == "__main__":
    main()
