"""Serialization helpers for the runtime server.

Stays plain JSON over UTF-8 so any client (Python, Web, Unreal) can
consume the stream with zero dependencies. Optional base64 encoding
of frame files is provided for embedding small PPMs in the stream.
"""

from __future__ import annotations

import base64

from cosmic_engine.runtime.scene_state import SceneState


def scene_state_to_json(
    scene_state: SceneState,
    *,
    indent: int | None = None,
) -> str:
    """Return a JSON string for ``scene_state``.

    Defaults to a compact (single-line) representation so the runtime
    server can use it as a newline-delimited stream message.
    """
    return scene_state.to_json(indent=indent)


def encode_frame_to_base64(image_path: str) -> str:
    """Read ``image_path`` and return its base64-encoded contents."""
    with open(image_path, "rb") as fh:
        return base64.b64encode(fh.read()).decode("ascii")
