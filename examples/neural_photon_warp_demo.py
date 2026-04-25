"""Demo: deterministic vs. neural photon warp at two warp factors."""

from __future__ import annotations

from pathlib import Path

from ai_viewer import NeuralWarpViewer

from cosmic_engine.ai import ONNXPhotonWarpModel
from cosmic_engine.core.units import SPEED_OF_LIGHT_M_S
from cosmic_engine.core.vector import Vector3
from cosmic_engine.perception.observer import ObserverState
from cosmic_engine.rendering.simple_camera import SimpleCamera
from cosmic_engine.runtime import CosmicRuntime, RuntimeConfig


_REPO_ROOT = Path(__file__).resolve().parent.parent
_MODEL = _REPO_ROOT / "data" / "photon_warp_model.onnx"
_OUT_DIR = _REPO_ROOT / "outputs" / "viewer"


def _camera() -> SimpleCamera:
    return SimpleCamera(
        position_m=Vector3.zero(),
        forward=Vector3(0.0, 1.0, 0.0),
        up=Vector3(0.0, 0.0, 1.0),
        fov_degrees=170.0,
        image_width=256,
        image_height=256,
    )


def _observer(*, warp_factor: float) -> ObserverState:
    return ObserverState(
        position_m=Vector3.zero(),
        velocity_m_s=Vector3(0.0, 0.5 * SPEED_OF_LIGHT_M_S, 0.0),
        forward=Vector3(0.0, 1.0, 0.0),
        up=Vector3(0.0, 0.0, 1.0),
        warp_factor=warp_factor,
    )


def main() -> None:
    runtime = CosmicRuntime(
        config=RuntimeConfig(
            enable_physics=False,
            physics_backend="none",
            enable_perception=True,
            output_directory=str(_OUT_DIR),
        )
    )
    runtime.load_sample_data()
    print(f"runtime objects   : {len(runtime.registry.list_objects())}")

    if _MODEL.is_file():
        warp_model = ONNXPhotonWarpModel(str(_MODEL))
        load_status = warp_model.last_error or "ok"
        print(f"photon warp model : {_MODEL} ({load_status})")
    else:
        warp_model = None
        print(f"photon warp model : {_MODEL} (not found, neural mode disabled)")

    print()
    print("=== mode 1: deterministic perception (warp_factor=2.0) ===")
    NeuralWarpViewer(
        runtime, _camera(), _observer(warp_factor=2.0), warp_model=None
    ).run_once(output_path=str(_OUT_DIR / "output_photon_det.ppm"))

    if warp_model is not None:
        print()
        print("=== mode 2: neural photon warp (warp_factor=2.0) ===")
        NeuralWarpViewer(
            runtime, _camera(), _observer(warp_factor=2.0), warp_model=warp_model
        ).run_once(output_path=str(_OUT_DIR / "output_photon_ai.ppm"))

        print()
        print("=== mode 3: neural photon warp (warp_factor=50.0) ===")
        NeuralWarpViewer(
            runtime,
            _camera(),
            _observer(warp_factor=50.0),
            warp_model=warp_model,
        ).run_once(output_path=str(_OUT_DIR / "output_photon_extreme.ppm"))
    else:
        print()
        print("(skipped neural modes — no model available)")

    print()
    print(f"outputs in        : {_OUT_DIR}")


if __name__ == "__main__":
    main()
