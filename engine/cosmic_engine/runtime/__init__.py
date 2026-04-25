"""Headless orchestration layer.

Phase 16 contribution: a thin coordinator that ties registry, time,
physics, perception, and rendering preparation into a single object
graph. The runtime owns no new domain logic — it only schedules calls
into the modules built in the previous phases.

No GUI, no real-time loop, no GPU. Designed for offline frame
generation, batch jobs, and tests.
"""

from cosmic_engine.runtime.config import RuntimeConfig
from cosmic_engine.runtime.pipeline import (
    run_headless_frame,
    run_streaming_frame,
)
from cosmic_engine.runtime.runtime import CosmicRuntime
from cosmic_engine.runtime.scene_state import SceneState
from cosmic_engine.runtime.server import RuntimeServer
from cosmic_engine.runtime.stream import (
    encode_frame_to_base64,
    scene_state_to_json,
)

__all__ = [
    "CosmicRuntime",
    "RuntimeConfig",
    "RuntimeServer",
    "SceneState",
    "encode_frame_to_base64",
    "run_headless_frame",
    "run_streaming_frame",
    "scene_state_to_json",
]
