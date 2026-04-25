"""Demo: Gaussian splat-render a 50k synthetic galaxy field on CPU."""

from __future__ import annotations

import time
from pathlib import Path

from ai_viewer.frame_buffer import FrameBuffer
from ai_viewer.neural_field import (
    GaussianSplatRenderer,
    build_gaussian_field_from_galaxy_batch,
)

from cosmic_engine.core.vector import Vector3
from cosmic_engine.data.galaxy_catalog import generate_synthetic_galaxy_catalog
from cosmic_engine.rendering import (
    SimpleCamera,
    build_galaxy_field_batch,
)


_REPO_ROOT = Path(__file__).resolve().parent.parent
_OUT = _REPO_ROOT / "outputs" / "viewer" / "output_gaussian.ppm"
_GALAXY_COUNT = 50_000
_RADIUS_M = 1.0e25
_RENDER_W = 256
_RENDER_H = 256


def main() -> None:
    print(f"generating {_GALAXY_COUNT} synthetic galaxies...")
    t_gen = time.perf_counter()
    galaxies = generate_synthetic_galaxy_catalog(_GALAXY_COUNT, _RADIUS_M, seed=42)
    gen_seconds = time.perf_counter() - t_gen

    camera = SimpleCamera(
        position_m=Vector3(0.0, -_RADIUS_M * 1.5, 0.0),
        forward=Vector3(0.0, 1.0, 0.0),
        up=Vector3(0.0, 0.0, 1.0),
        fov_degrees=90.0,
        image_width=_RENDER_W,
        image_height=_RENDER_H,
    )

    t_batch = time.perf_counter()
    batch = build_galaxy_field_batch(galaxies, camera)
    batch_seconds = time.perf_counter() - t_batch

    t_field = time.perf_counter()
    points = build_gaussian_field_from_galaxy_batch(batch, sigma_scale=1.0)
    field_seconds = time.perf_counter() - t_field

    renderer = GaussianSplatRenderer(_RENDER_W, _RENDER_H, camera)
    t_render = time.perf_counter()
    image = renderer.render(points)
    render_seconds = time.perf_counter() - t_render

    fb = FrameBuffer(_RENDER_W, _RENDER_H)
    fb.pixels = image
    _OUT.parent.mkdir(parents=True, exist_ok=True)
    fb.save_ppm(str(_OUT))

    print()
    print(f"galaxy count       : {_GALAXY_COUNT}")
    print(f"galaxy batch size  : {len(batch)}")
    print(f"gaussian points    : {len(points)}")
    print(f"image size         : {_RENDER_W} x {_RENDER_H}")
    print()
    print(f"generate galaxies  : {gen_seconds * 1000:8.2f} ms")
    print(f"build galaxy batch : {batch_seconds * 1000:8.2f} ms")
    print(f"build field        : {field_seconds * 1000:8.2f} ms")
    print(f"render splats      : {render_seconds * 1000:8.2f} ms")
    print()
    print(f"output saved to    : {_OUT}")


if __name__ == "__main__":
    main()
