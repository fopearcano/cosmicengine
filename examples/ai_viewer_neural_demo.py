"""Demo: AI Viewer with an optional ONNX neural postprocessor.

Usage:
    python examples/ai_viewer_neural_demo.py
        -> deterministic Composite postprocessor (no model)
    python examples/ai_viewer_neural_demo.py --model data/neural_postprocess_identity.onnx
        -> ONNXFrameProcessor wrapped in SafeProcessor
"""

from __future__ import annotations

import argparse
import threading
import time
from pathlib import Path

from ai_viewer import (
    AIViewer,
    AIViewerConfig,
    RuntimeClient,
    SafeProcessor,
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
    while not stop_event.is_set():
        try:
            _, frame_path = run_streaming_frame(runtime, camera, observer)
            server.broadcast_frame(frame_path)
        except Exception:
            pass
        stop_event.wait(period_seconds)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "Path to an ONNX model. If omitted, the deterministic "
            "Composite postprocessor is used."
        ),
    )
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument(
        "--neural-input-width",
        type=int,
        default=None,
        help="Optional neural model input width.",
    )
    parser.add_argument(
        "--neural-input-height",
        type=int,
        default=None,
        help="Optional neural model input height.",
    )
    parser.add_argument("--seconds", type=float, default=2.0)
    args = parser.parse_args()

    if args.model is None:
        print("no --model given: viewer will use the deterministic fallback")
    else:
        print(f"--model = {args.model}")

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
        image_width=args.width,
        image_height=args.height,
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
        width=args.width,
        height=args.height,
        output_directory=str(_OUT_DIR),
        enable_window=False,
        enable_postprocess=True,
        max_fps=30.0,
        neural_model_path=args.model,
        neural_input_width=args.neural_input_width,
        neural_input_height=args.neural_input_height,
    )
    viewer_config.validate()

    client = RuntimeClient(viewer_config)
    viewer = AIViewer(viewer_config, client)

    try:
        client.connect()
        deadline = time.time() + args.seconds
        processed = 0
        while time.time() < deadline:
            message = viewer.run_once()
            if message is None:
                break
            processed += 1
    finally:
        stop_event.set()
        pusher.join(timeout=1.0)
        client.disconnect()
        server.stop()

    print()
    print(f"messages processed   : {processed}")
    print(f"scene_states received: {viewer.scene_states_received}")
    print(f"frames received      : {viewer.frames_received}")
    print(f"final FPS            : {viewer.fps():.1f}")
    if isinstance(viewer.postprocessor, SafeProcessor):
        inner_error = getattr(viewer.postprocessor.wrapped, "last_error", None)
        if inner_error:
            print(f"neural model error   : {inner_error}")
            print("(viewer fell back to passthrough — frames sent unchanged)")
    if viewer.last_frame_path:
        print(f"last frame saved     : {viewer.last_frame_path}")


if __name__ == "__main__":
    main()
