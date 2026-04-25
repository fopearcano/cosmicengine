"""Tests for Phase 17 runtime server / streaming."""

from __future__ import annotations

import base64
import json
import socket
import time
from pathlib import Path

import pytest

from cosmic_engine.core.object_types import CosmicObjectType
from cosmic_engine.core.truth import TruthLevel
from cosmic_engine.core.universe_object import UniverseObject
from cosmic_engine.core.vector import Vector3
from cosmic_engine.perception.observer import ObserverState
from cosmic_engine.rendering.simple_camera import SimpleCamera
from cosmic_engine.runtime import (
    CosmicRuntime,
    RuntimeConfig,
    RuntimeServer,
    SceneState,
    encode_frame_to_base64,
    run_streaming_frame,
    scene_state_to_json,
)


def _sample_state() -> SceneState:
    return SceneState(
        julian_date=2_451_545.0,
        total_objects=2,
        active_objects=2,
        object_type_counts={"star": 2},
        truth_level_counts={"catalog_imported": 2},
        source_counts={"gaia": 2},
        physics_backend="none",
        perception_enabled=False,
        ai_warp_enabled=False,
        notes=["frame ok"],
    )


def _star(object_id: str, position: Vector3) -> UniverseObject:
    return UniverseObject(
        id=object_id,
        name=object_id,
        object_type=CosmicObjectType.STAR,
        position_m=position,
        velocity_m_s=Vector3.zero(),
        truth_level=TruthLevel.CATALOG_IMPORTED,
        source="gaia",
        spectral_class="G2V",
        metadata={"apparent_magnitude": 1.0},
    )


# --- stream helpers ---


def test_scene_state_to_json_contains_expected_keys():
    payload = json.loads(scene_state_to_json(_sample_state()))
    for key in (
        "julian_date",
        "total_objects",
        "active_objects",
        "object_type_counts",
        "truth_level_counts",
        "source_counts",
        "physics_backend",
        "perception_enabled",
        "ai_warp_enabled",
        "notes",
    ):
        assert key in payload


def test_scene_state_to_json_compact_default():
    text = scene_state_to_json(_sample_state())
    assert "\n" not in text


def test_encode_frame_to_base64_round_trip(tmp_path: Path):
    p = tmp_path / "f.ppm"
    p.write_bytes(b"hello-frame")
    encoded = encode_frame_to_base64(str(p))
    assert base64.b64decode(encoded) == b"hello-frame"


# --- run_streaming_frame ---


def test_run_streaming_frame_writes_a_ppm(tmp_path: Path):
    runtime = CosmicRuntime(
        config=RuntimeConfig(
            enable_physics=False,
            enable_perception=False,
            output_directory=str(tmp_path),
        )
    )
    runtime.add_objects([_star("s1", Vector3(0.0, 1.0e15, 0.0))])
    camera = SimpleCamera(
        position_m=Vector3.zero(),
        forward=Vector3(0.0, 1.0, 0.0),
        up=Vector3(0.0, 0.0, 1.0),
        fov_degrees=120.0,
        image_width=64,
        image_height=64,
    )
    observer = ObserverState(
        position_m=Vector3.zero(),
        velocity_m_s=Vector3.zero(),
        forward=Vector3(0.0, 1.0, 0.0),
        up=Vector3(0.0, 0.0, 1.0),
    )
    state, frame_path = run_streaming_frame(runtime, camera, observer)
    assert isinstance(state, SceneState)
    assert Path(frame_path).is_file()


# --- RuntimeServer ---


def test_server_rejects_non_positive_tick_rate():
    runtime = CosmicRuntime()
    with pytest.raises(ValueError):
        RuntimeServer(runtime, port=0, tick_rate_hz=0.0)
    with pytest.raises(ValueError):
        RuntimeServer(runtime, port=0, tick_rate_hz=-1.0)


def test_server_starts_and_stops_cleanly():
    runtime = CosmicRuntime()
    server = RuntimeServer(runtime, port=0, tick_rate_hz=20.0)
    server.start()
    try:
        assert server.running is True
        assert server.port > 0
    finally:
        server.stop()
    assert server.running is False
    assert server.connection_count() == 0


def test_server_loop_runs_at_least_one_iteration():
    runtime = CosmicRuntime()
    server = RuntimeServer(runtime, port=0, tick_rate_hz=50.0)
    server.start()
    try:
        deadline = time.time() + 1.0
        while time.time() < deadline and server.last_frame_time is None:
            time.sleep(0.02)
        assert server.last_frame_time is not None
        assert runtime.last_scene_state is not None
    finally:
        server.stop()


def test_server_broadcasts_state_to_a_subscriber():
    runtime = CosmicRuntime()
    runtime.add_objects([_star("s1", Vector3(0.0, 1.0e15, 0.0))])
    server = RuntimeServer(runtime, port=0, tick_rate_hz=20.0)
    server.start()
    try:
        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client.settimeout(2.0)
        client.connect(("127.0.0.1", server.port))
        try:
            buffer = bytearray()
            deadline = time.time() + 2.0
            while b"\n" not in buffer and time.time() < deadline:
                chunk = client.recv(4096)
                if not chunk:
                    break
                buffer.extend(chunk)
            line, _, _ = bytes(buffer).partition(b"\n")
            assert line, "no message received from server"
            message = json.loads(line.decode("utf-8"))
            assert message["type"] == "scene_state"
            assert "julian_date" in message["data"]
            assert message["data"]["total_objects"] == 1
        finally:
            client.close()
    finally:
        server.stop()


def test_server_dead_subscribers_do_not_break_broadcast():
    runtime = CosmicRuntime()
    server = RuntimeServer(runtime, port=0, tick_rate_hz=50.0)
    server.start()
    try:
        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client.settimeout(1.0)
        client.connect(("127.0.0.1", server.port))
        client.close()  # immediate disconnect
        # give the server a few ticks to detect the dead socket
        time.sleep(0.2)
        # next broadcast must not raise; the dead connection should be
        # cleaned up to zero
        assert server.running is True
        # one or more ticks must still have run
        assert server.last_frame_time is not None
    finally:
        server.stop()
