"""Demo: real-time viewer window with a chained postprocessor.

Spins up a runtime server in-process at high tick rate, has it push a
PPM frame every cycle, and runs the AI Viewer with a contrast +
color-shift postprocessing chain. Falls back to headless mode (no
window, ASCII preview only) if a display is not available.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

from ai_viewer import (
    AIViewer,
    AIViewerConfig,
    BrightnessProcessor,
    ColorShiftProcessor,
    CompositeProcessor,
    ContrastBoostProcessor,
    RuntimeClient,
    ViewerWindow,
)

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


_OUT_DIR = Path(__file__).resolve().parent.parent / "outputs" / "viewer"


def _frame_pusher(
    server: RuntimeServer,
    runtime: CosmicRuntime,
    camera: SimpleCamera,
    observer: ObserverState,
    stop_event: threading.Event,
    period_seconds: float,
) -> None:
    """Background thread that broadcasts a fresh PPM frame each period."""
    while not stop_event.is_set():
        try:
            _, frame_path = run_streaming_frame(runtime, camera, observer)
            server.broadcast_frame(frame_path)
        except Exception:
            pass
        stop_event.wait(period_seconds)


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
        runtime, host="127.0.0.1", port=0, tick_rate_hz=20.0
    )
    server.start()
    print(f"server listening  : 127.0.0.1:{server.port}")

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

    stop_event = threading.Event()
    pusher = threading.Thread(
        target=_frame_pusher,
        args=(server, runtime, camera, observer, stop_event, 0.1),
        daemon=True,
        name="frame-pusher",
    )
    pusher.start()

    viewer_config = AIViewerConfig(
        server_host="127.0.0.1",
        server_port=server.port,
        width=128,
        height=128,
        output_directory=str(_OUT_DIR),
        enable_window=True,
        enable_postprocess=True,
        max_fps=30.0,
    )
    viewer_config.validate()
    window = ViewerWindow(width=128, height=128)
    postprocessor = CompositeProcessor(
        [
            ContrastBoostProcessor(factor=1.4),
            BrightnessProcessor(factor=1.1),
            ColorShiftProcessor(r_shift=0, g_shift=10, b_shift=30),
        ]
    )
    client = RuntimeClient(viewer_config)
    viewer = AIViewer(viewer_config, client, window=window, postprocessor=postprocessor)

    print(f"window available  : {window.available}")
    print("running for ~3 seconds (drain up to 30 messages)...")
    try:
        client.connect()
        deadline = time.time() + 3.0
        processed = 0
        while time.time() < deadline and processed < 60:
            message = viewer.run_once()
            if message is None:
                break
            processed += 1
    finally:
        stop_event.set()
        pusher.join(timeout=1.0)
        client.disconnect()
        window.close()
        server.stop()

    print()
    print(f"scene_states received : {viewer.scene_states_received}")
    print(f"frames received       : {viewer.frames_received}")
    print(f"final FPS             : {viewer.fps():.1f}")
    if viewer.last_frame_path:
        print(f"last frame saved      : {viewer.last_frame_path}")
        print()
        print("ASCII preview (40 cols, postprocessed):")
        print(viewer.frame_buffer.to_ascii_preview(max_width=40))


if __name__ == "__main__":
    main()
