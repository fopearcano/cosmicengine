"""Demo: warp a Gaussian galaxy field at three warp factors and render."""

from __future__ import annotations

import time
from pathlib import Path

from ai_viewer.frame_buffer import FrameBuffer
from ai_viewer.neural_field import (
    GaussianSplatRenderer,
    build_gaussian_field_from_galaxy_batch,
    warp_gaussian_field,
)

from cosmic_engine.core.units import SPEED_OF_LIGHT_M_S
from cosmic_engine.core.vector import Vector3
from cosmic_engine.data.galaxy_catalog import generate_synthetic_galaxy_catalog
from cosmic_engine.perception.observer import ObserverState
from cosmic_engine.rendering import (
    SimpleCamera,
    build_galaxy_field_batch,
)


_REPO_ROOT = Path(__file__).resolve().parent.parent
_OUT_DIR = _REPO_ROOT / "outputs" / "viewer"
_GALAXY_COUNT = 20_000
_RADIUS_M = 1.0e25
_RENDER_W = 256
_RENDER_H = 256


def _camera() -> SimpleCamera:
    return SimpleCamera(
        position_m=Vector3(0.0, -_RADIUS_M * 1.5, 0.0),
        forward=Vector3(0.0, 1.0, 0.0),
        up=Vector3(0.0, 0.0, 1.0),
        fov_degrees=90.0,
        image_width=_RENDER_W,
        image_height=_RENDER_H,
    )


def _observer(*, warp_factor: float, camera: SimpleCamera) -> ObserverState:
    return ObserverState(
        position_m=camera.position_m,
        velocity_m_s=Vector3(0.0, 0.5 * SPEED_OF_LIGHT_M_S, 0.0),
        forward=camera.forward,
        up=camera.up,
        warp_factor=warp_factor,
    )


def _save(image, path: Path) -> None:
    fb = FrameBuffer(image.shape[1], image.shape[0])
    fb.pixels = image
    path.parent.mkdir(parents=True, exist_ok=True)
    fb.save_ppm(str(path))


def main() -> None:
    print(f"generating {_GALAXY_COUNT} synthetic galaxies...")
    galaxies = generate_synthetic_galaxy_catalog(_GALAXY_COUNT, _RADIUS_M, seed=42)

    camera = _camera()
    t0 = time.perf_counter()
    batch = build_galaxy_field_batch(galaxies, camera)
    points = build_gaussian_field_from_galaxy_batch(batch, sigma_scale=1.0)
    build_seconds = time.perf_counter() - t0
    print(f"gaussian points    : {len(points)} (built in {build_seconds * 1000:.1f} ms)")
    print()
    print(f"{'warp':>5}  {'warp ms':>9}  {'render ms':>10}  output")
    print("-" * 60)

    renderer = GaussianSplatRenderer(_RENDER_W, _RENDER_H, camera)
    for warp_factor in (1.0, 5.0, 50.0):
        observer = _observer(warp_factor=warp_factor, camera=camera)
        t_warp = time.perf_counter()
        warped = warp_gaussian_field(points, observer)
        warp_ms = (time.perf_counter() - t_warp) * 1000.0

        t_render = time.perf_counter()
        image = renderer.render(warped)
        render_ms = (time.perf_counter() - t_render) * 1000.0

        out = _OUT_DIR / f"output_gaussian_warp_{int(warp_factor)}.ppm"
        _save(image, out)
        print(
            f"{warp_factor:>5.0f}  {warp_ms:>9.2f}  {render_ms:>10.2f}  {out}"
        )

    print()
    print("done.")


if __name__ == "__main__":
    main()
