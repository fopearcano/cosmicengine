"""Demo: render a 200k-galaxy Gaussian field via WebGPU when available.

Falls back to the CPU splat renderer if no GPU adapter is reachable.
"""

from __future__ import annotations

import time
from pathlib import Path

from ai_viewer.frame_buffer import FrameBuffer
from ai_viewer.neural_field import (
    GaussianSplatRenderer,
    build_gaussian_field_from_galaxy_batch,
)
from ai_viewer.neural_field.gpu import (
    GaussianSplatPipeline,
    WebGPUDevice,
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
_OUT_PATH = _REPO_ROOT / "outputs" / "viewer" / "output_webgpu.ppm"


def _save(image, path: Path) -> None:
    fb = FrameBuffer(image.shape[1], image.shape[0])
    fb.pixels = image
    path.parent.mkdir(parents=True, exist_ok=True)
    fb.save_ppm(str(path))


def main() -> None:
    galaxy_count = 200_000
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
    observer = ObserverState(
        position_m=camera.position_m,
        velocity_m_s=Vector3(0.0, 0.5 * SPEED_OF_LIGHT_M_S, 0.0),
        forward=camera.forward,
        up=camera.up,
        warp_factor=2.0,
    )

    t_build = time.perf_counter()
    batch = build_galaxy_field_batch(galaxies, camera)
    points = build_gaussian_field_from_galaxy_batch(batch, sigma_scale=1.0)
    build_seconds = time.perf_counter() - t_build
    print(
        f"gaussian points    : {len(points)} (built in {build_seconds * 1000:.1f} ms)"
    )

    # Probe WebGPU.
    device = WebGPUDevice()
    print(f"requested backend  : webgpu")
    print(f"resolved backend   : {device.get_backend()}")
    if device.last_error:
        print(f"webgpu error       : {device.last_error}")

    # CPU baseline always available.
    cpu_renderer = GaussianSplatRenderer(width, height, camera)
    t_cpu = time.perf_counter()
    cpu_image = cpu_renderer.render(points)
    cpu_seconds = time.perf_counter() - t_cpu

    # GPU pipeline with WebGPU device. On unavailable hardware the
    # pipeline silently routes back through the CPU fallback.
    pipeline = GaussianSplatPipeline(device)
    t_gpu = time.perf_counter()
    gpu_image = pipeline.render(points, camera, observer)
    gpu_seconds = time.perf_counter() - t_gpu

    _save(gpu_image, _OUT_PATH)

    print()
    print(f"{'backend':<12}  {'time ms':>10}")
    print("-" * 30)
    print(f"{'cpu':<12}  {cpu_seconds * 1000:>10.2f}")
    print(
        f"{device.get_backend():<12}  {gpu_seconds * 1000:>10.2f}"
    )
    print()
    print(f"point count        : {len(points)}")
    print(f"output saved to    : {_OUT_PATH}")
    print(f"warp applied in GPU: yes (in-shader, when backend is webgpu)")


if __name__ == "__main__":
    main()
