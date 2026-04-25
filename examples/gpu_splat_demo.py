"""Demo: compare the CPU splat renderer with the GPU-mock pipeline."""

from __future__ import annotations

import time
from pathlib import Path

from ai_viewer.frame_buffer import FrameBuffer
from ai_viewer.neural_field import (
    GaussianSplatRenderer,
    build_gaussian_field_from_galaxy_batch,
)
from ai_viewer.neural_field.gpu import (
    GPUDevice,
    GaussianSplatPipeline,
)

from cosmic_engine.core.vector import Vector3
from cosmic_engine.data.galaxy_catalog import generate_synthetic_galaxy_catalog
from cosmic_engine.rendering import (
    SimpleCamera,
    build_galaxy_field_batch,
)


_REPO_ROOT = Path(__file__).resolve().parent.parent
_OUT_DIR = _REPO_ROOT / "outputs" / "viewer"
_GALAXY_COUNT = 100_000
_RADIUS_M = 1.0e25
_RENDER_W = 256
_RENDER_H = 256


def _save(image, path: Path) -> None:
    fb = FrameBuffer(image.shape[1], image.shape[0])
    fb.pixels = image
    path.parent.mkdir(parents=True, exist_ok=True)
    fb.save_ppm(str(path))


def main() -> None:
    print(f"generating {_GALAXY_COUNT} synthetic galaxies...")
    galaxies = generate_synthetic_galaxy_catalog(_GALAXY_COUNT, _RADIUS_M, seed=42)

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
    points = build_gaussian_field_from_galaxy_batch(batch, sigma_scale=1.0)
    build_seconds = time.perf_counter() - t_batch
    print(f"gaussian points    : {len(points)} (built in {build_seconds * 1000:.1f} ms)")
    print()

    # CPU path
    cpu_renderer = GaussianSplatRenderer(_RENDER_W, _RENDER_H, camera)
    t_cpu = time.perf_counter()
    cpu_image = cpu_renderer.render(points)
    cpu_seconds = time.perf_counter() - t_cpu
    cpu_path = _OUT_DIR / "output_gpu_cpu.ppm"
    _save(cpu_image, cpu_path)

    # GPU mock path
    device = GPUDevice(backend="mock_gpu")
    pipeline = GaussianSplatPipeline(device)
    t_mock = time.perf_counter()
    mock_image = pipeline.render(points, camera)
    mock_seconds = time.perf_counter() - t_mock
    mock_path = _OUT_DIR / "output_gpu_mock.ppm"
    _save(mock_image, mock_path)

    print(f"{'backend':<12}  {'time ms':>10}  output")
    print("-" * 60)
    print(f"{'cpu':<12}  {cpu_seconds * 1000:>10.2f}  {cpu_path}")
    print(
        f"{'mock_gpu':<12}  {mock_seconds * 1000:>10.2f}  {mock_path}"
    )
    speedup = cpu_seconds / mock_seconds if mock_seconds > 0 else float("nan")
    print()
    print(f"point count     : {len(points)}")
    print(f"speedup (mock)  : {speedup:.2f}x")
    print(f"backend (mock)  : {device.get_backend()}  available={device.is_available()}")


if __name__ == "__main__":
    main()
