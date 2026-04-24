"""Demo: compare deterministic / placeholder / ONNX AI warp paths."""

from __future__ import annotations

from pathlib import Path

from cosmic_engine.ai import AIWarpModel, ONNXWarpModel, SimpleNeuralWarp
from cosmic_engine.core.registry import UniverseRegistry
from cosmic_engine.core.units import SPEED_OF_LIGHT_M_S
from cosmic_engine.core.vector import Vector3
from cosmic_engine.data.star_catalog import load_star_catalog_into_registry
from cosmic_engine.perception import ObserverState, transform_photon_field
from cosmic_engine.rendering import (
    SimpleCamera,
    build_star_photon_field,
    render_photon_field_to_ppm,
)

_REPO_ROOT = Path(__file__).resolve().parent.parent
_CATALOG = _REPO_ROOT / "data" / "sample_stars.csv"
_MODEL = _REPO_ROOT / "data" / "mock_warp_model.onnx"
_OUTDIR = Path(__file__).resolve().parent


def _try_onnx() -> tuple[AIWarpModel | None, str]:
    try:
        return ONNXWarpModel(str(_MODEL)), "ok"
    except Exception as e:  # pragma: no cover - demo convenience
        return None, f"failed: {e}"


def main() -> None:
    registry = UniverseRegistry()
    load_star_catalog_into_registry(str(_CATALOG), registry)
    objects = registry.list_objects()

    camera = SimpleCamera(
        position_m=Vector3.zero(),
        forward=Vector3(0.0, 1.0, 0.0),
        up=Vector3(0.0, 0.0, 1.0),
        fov_degrees=140.0,
        image_width=512,
        image_height=512,
    )
    samples = build_star_photon_field(objects, camera)

    onnx_model, onnx_status = _try_onnx()
    scenarios: list[tuple[str, AIWarpModel | None, str]] = [
        ("onnx_off", None, "deterministic"),
        ("onnx_simple", SimpleNeuralWarp(), "SimpleNeuralWarp"),
        ("onnx_real", onnx_model, f"ONNXWarpModel[{onnx_status}]"),
    ]

    speed = 0.5 * SPEED_OF_LIGHT_M_S
    for label, model, description in scenarios:
        observer = ObserverState(
            position_m=Vector3.zero(),
            velocity_m_s=Vector3(0.0, speed, 0.0),
            forward=Vector3(0.0, 1.0, 0.0),
            up=Vector3(0.0, 0.0, 1.0),
            warp_factor=5.0,
        )
        observer.validate()
        warped = transform_photon_field(samples, observer, model)
        out_path = _OUTDIR / f"output_{label}.ppm"
        render_photon_field_to_ppm(warped, camera, str(out_path))

        conf = f"{model.confidence():.2f}" if model is not None else "n/a"
        print(
            f"scenario={label:<12}  model={description:<28}  "
            f"confidence={conf:<4}  samples={len(warped)}  "
            f"-> {out_path.name}"
        )


if __name__ == "__main__":
    main()
