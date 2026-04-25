"""Demo: enhance a synthetic galaxy density grid via simple + ONNX paths."""

from __future__ import annotations

from pathlib import Path

from cosmic_engine.ai import (
    DensityFieldModel,
    ONNXDensityModel,
    SimpleDensityEnhancer,
    enhance_density_field,
)
from cosmic_engine.ai.density_utils import (
    project_density_to_2d,
    render_density_image_to_ppm,
)
from cosmic_engine.core.vector import Vector3
from cosmic_engine.data.galaxy_catalog import generate_synthetic_galaxy_catalog
from cosmic_engine.rendering import (
    SimpleCamera,
    build_galaxy_field_batch,
    galaxy_batch_to_density_grid,
)


_REPO_ROOT = Path(__file__).resolve().parent.parent
_MODEL = _REPO_ROOT / "data" / "density_upscaler.onnx"
_OUTDIR = Path(__file__).resolve().parent

_GALAXY_COUNT = 20_000
_RADIUS_M = 1.0e25
_GRID_SIZE = 64


def _try_onnx() -> tuple[DensityFieldModel | None, str]:
    try:
        return ONNXDensityModel(str(_MODEL)), "ok"
    except Exception as e:  # pragma: no cover - demo convenience
        return None, f"unavailable: {e}"


def main() -> None:
    galaxies = generate_synthetic_galaxy_catalog(_GALAXY_COUNT, _RADIUS_M, seed=42)
    camera = SimpleCamera(
        position_m=Vector3.zero(),
        forward=Vector3(0.0, 1.0, 0.0),
        up=Vector3(0.0, 0.0, 1.0),
        fov_degrees=90.0,
        image_width=512,
        image_height=512,
    )
    batch = build_galaxy_field_batch(galaxies, camera)
    grid = galaxy_batch_to_density_grid(
        batch, grid_size=_GRID_SIZE, extent_m=_RADIUS_M * 1.2
    )

    simple_model = SimpleDensityEnhancer()
    onnx_model, onnx_status = _try_onnx()

    enhanced_simple = enhance_density_field(grid, simple_model)
    enhanced_ai = enhance_density_field(grid, onnx_model)

    render_density_image_to_ppm(
        project_density_to_2d(grid),
        str(_OUTDIR / "output_density_original.ppm"),
    )
    render_density_image_to_ppm(
        project_density_to_2d(enhanced_simple),
        str(_OUTDIR / "output_density_simple.ppm"),
    )
    render_density_image_to_ppm(
        project_density_to_2d(enhanced_ai),
        str(_OUTDIR / "output_density_ai.ppm"),
    )

    print(f"galaxy count          : {len(galaxies)}")
    print(f"original grid shape   : {grid.shape}  occupied: {(grid > 0).sum()}")
    print(
        f"simple-enhanced shape : {enhanced_simple.shape}  "
        f"model: SimpleDensityEnhancer  conf={simple_model.confidence():.2f}"
    )
    if onnx_model is None:
        print(
            f"ai-enhanced shape     : {enhanced_ai.shape}  "
            f"model: SimpleDensityEnhancer (fallback) [{onnx_status}]"
        )
    else:
        print(
            f"ai-enhanced shape     : {enhanced_ai.shape}  "
            f"model: ONNXDensityModel [{onnx_status}]  "
            f"conf={onnx_model.confidence():.2f}"
        )
    print()
    print(f"output_density_original.ppm  ({_OUTDIR / 'output_density_original.ppm'})")
    print(f"output_density_simple.ppm    ({_OUTDIR / 'output_density_simple.ppm'})")
    print(f"output_density_ai.ppm        ({_OUTDIR / 'output_density_ai.ppm'})")


if __name__ == "__main__":
    main()
