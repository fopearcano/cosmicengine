"""Headless orchestration layer.

Phase 16 contribution: a thin coordinator that ties registry, time,
physics, perception, and rendering preparation into a single object
graph. The runtime owns no new domain logic — it only schedules calls
into the modules built in the previous phases.

No GUI, no real-time loop, no GPU. Designed for offline frame
generation, batch jobs, and tests.
"""

from cosmic_engine.runtime.config import RuntimeConfig
from cosmic_engine.runtime.pipeline import run_headless_frame
from cosmic_engine.runtime.runtime import CosmicRuntime
from cosmic_engine.runtime.scene_state import SceneState

__all__ = [
    "CosmicRuntime",
    "RuntimeConfig",
    "SceneState",
    "run_headless_frame",
]
