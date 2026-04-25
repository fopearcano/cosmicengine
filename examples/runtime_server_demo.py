"""Demo: start the CosmicEngine runtime server for a few seconds."""

from __future__ import annotations

import time

from cosmic_engine.runtime import CosmicRuntime, RuntimeConfig, RuntimeServer


def main() -> None:
    runtime = CosmicRuntime(
        config=RuntimeConfig(
            enable_physics=False,
            physics_backend="none",
            enable_perception=False,
        )
    )
    runtime.load_sample_data()
    print(f"loaded objects     : {len(runtime.registry.list_objects())}")

    server = RuntimeServer(
        runtime,
        host="127.0.0.1",
        port=8765,
        tick_rate_hz=5.0,
    )
    server.start()
    print(f"server listening   : {server.host}:{server.port}")
    print(f"tick rate (Hz)     : {server.tick_rate_hz}")
    print("running for 3 seconds...")

    try:
        time.sleep(3.0)
    finally:
        server.stop()
        last = runtime.last_scene_state
        if last is not None:
            print()
            print("final scene state:")
            print(last.to_json())
        print("server stopped cleanly")


if __name__ == "__main__":
    main()
