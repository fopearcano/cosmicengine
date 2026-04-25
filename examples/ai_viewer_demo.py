"""Demo: spin up a runtime server in-process and consume it from the AI Viewer."""

from __future__ import annotations

import time
from pathlib import Path

from ai_viewer import AIViewer, AIViewerConfig, RuntimeClient

from cosmic_engine.core.units import SPEED_OF_LIGHT_M_S
from cosmic_engine.core.vector import Vector3
from cosmic_engine.perception.observer import ObserverState
from cosmic_engine.rendering.simple_camera import SimpleCamera
from cosmic_engine.runtime import (
    CosmicRuntime,
    RuntimeConfig,
    RuntimeServer,
    run_streaming_frame,
)


_OUT_DIR = (
    Path(__file__).resolve().parent.parent / "outputs" / "viewer"
)


def main() -> None:
    runtime_config = RuntimeConfig(
        enable_physics=False,
        physics_backend="none",
        enable_perception=True,
        output_directory=str(_OUT_DIR),
    )
    runtime = CosmicRuntime(config=runtime_config)
    runtime.load_sample_data()
    print(f"runtime objects   : {len(runtime.registry.list_objects())}")

    server = RuntimeServer(
        runtime, host="127.0.0.1", port=0, tick_rate_hz=10.0
    )
    server.start()
    print(f"server listening  : 127.0.0.1:{server.port}")

    viewer_config = AIViewerConfig(
        server_host="127.0.0.1",
        server_port=server.port,
        enable_ai_postprocess=True,
        output_directory=str(_OUT_DIR),
    )
    viewer_config.validate()
    client = RuntimeClient(viewer_config)
    viewer = AIViewer(viewer_config, client)

    try:
        client.connect()
        # First, drain a few SceneState messages.
        for _ in range(3):
            message = viewer.run_once()
            if message is None:
                break
            if viewer.last_scene_state is not None:
                print()
                print(viewer.render_scene_state_text(viewer.last_scene_state))

        # Then have the runtime produce one PPM and broadcast it as a frame.
        camera = SimpleCamera(
            position_m=Vector3.zero(),
            forward=Vector3(0.0, 1.0, 0.0),
            up=Vector3(0.0, 0.0, 1.0),
            fov_degrees=170.0,
            image_width=128,
            image_height=128,
        )
        observer = ObserverState(
            position_m=Vector3.zero(),
            velocity_m_s=Vector3(0.0, 0.5 * SPEED_OF_LIGHT_M_S, 0.0),
            forward=Vector3(0.0, 1.0, 0.0),
            up=Vector3(0.0, 0.0, 1.0),
            warp_factor=2.0,
        )
        _, frame_path = run_streaming_frame(runtime, camera, observer)
        server.broadcast_frame(frame_path)
        # Give the broadcaster a moment to push the frame.
        time.sleep(0.2)

        # Drain up to a few more messages, looking for the frame.
        for _ in range(20):
            message = viewer.run_once()
            if message is None:
                break
            if message.get("type") == "frame":
                break

        print()
        print(f"scene_states received : {viewer.scene_states_received}")
        print(f"frames received       : {viewer.frames_received}")
        if viewer.last_frame_path:
            print(f"last frame saved      : {viewer.last_frame_path}")
            print()
            print("ASCII preview (40 cols):")
            print(viewer.frame_buffer.to_ascii_preview(max_width=40))
    finally:
        client.disconnect()
        server.stop()


if __name__ == "__main__":
    main()
