"""Demo: drive the unified CosmicRuntime through three headless frames."""

from __future__ import annotations

from pathlib import Path

from cosmic_engine.core.units import SPEED_OF_LIGHT_M_S
from cosmic_engine.core.vector import Vector3
from cosmic_engine.perception.observer import ObserverState
from cosmic_engine.rendering.simple_camera import SimpleCamera
from cosmic_engine.runtime import CosmicRuntime, RuntimeConfig, run_headless_frame


_OUTPUT = Path(__file__).resolve().parent / "output_runtime_demo.ppm"
_DAY_SECONDS = 86_400.0


def main() -> None:
    config = RuntimeConfig(
        enable_physics=False,
        physics_backend="none",
        enable_perception=False,
        render_width=512,
        render_height=512,
    )
    runtime = CosmicRuntime(config=config)
    runtime.load_sample_data()
    print(f"loaded objects: {len(runtime.registry.list_objects())}")

    camera = SimpleCamera(
        position_m=Vector3.zero(),
        forward=Vector3(0.0, 1.0, 0.0),
        up=Vector3(0.0, 0.0, 1.0),
        fov_degrees=170.0,
        image_width=config.render_width,
        image_height=config.render_height,
    )
    observer = ObserverState(
        position_m=Vector3.zero(),
        velocity_m_s=Vector3(0.0, 0.5 * SPEED_OF_LIGHT_M_S, 0.0),
        forward=Vector3(0.0, 1.0, 0.0),
        up=Vector3(0.0, 0.0, 1.0),
        warp_factor=2.0,
    )

    print("\n=== frame 1: baseline (no physics, no perception) ===")
    state1 = run_headless_frame(runtime, camera, observer, output_path=None)
    print(state1.to_json())

    print("\n=== frame 2: physics step (exact_nbody) ===")
    config.enable_physics = True
    config.physics_backend = "exact_nbody"
    runtime.step(_DAY_SECONDS)
    state2 = run_headless_frame(runtime, camera, observer, output_path=None)
    print(state2.to_json())

    print("\n=== frame 3: perception enabled, with PPM output ===")
    config.enable_perception = True
    state3 = run_headless_frame(runtime, camera, observer, output_path=str(_OUTPUT))
    print(state3.to_json())

    print(f"\noutput image: {_OUTPUT}")


if __name__ == "__main__":
    main()
