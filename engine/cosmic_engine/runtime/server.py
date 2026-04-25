"""TCP-based runtime server that streams :class:`SceneState` updates.

Protocol (intentionally tiny):
- Plain TCP, newline-delimited UTF-8 JSON messages.
- Each tick the server runs one ``runtime.step(period)`` and broadcasts
  the resulting :class:`SceneState` as one JSON object terminated by
  ``\\n``.
- Frames are not pushed by default. ``broadcast_frame(path)`` will
  base64-encode an image file and send a single
  ``{"type": "frame", "data": "..."}`` message on demand.

Standard library only: :mod:`socket`, :mod:`threading`, :mod:`time`.
No authentication or framing beyond the newline; this is for trusted
local consumers (AI viewer, Unreal bridge, web client) — not the open
internet.
"""

from __future__ import annotations

import json
import socket
import threading
import time

from cosmic_engine.runtime.runtime import CosmicRuntime
from cosmic_engine.runtime.scene_state import SceneState
from cosmic_engine.runtime.stream import encode_frame_to_base64, scene_state_to_json


class RuntimeServer:
    """Background TCP server broadcasting SceneState JSON to subscribers."""

    def __init__(
        self,
        runtime: CosmicRuntime,
        host: str = "127.0.0.1",
        port: int = 8765,
        tick_rate_hz: float = 10.0,
    ) -> None:
        if tick_rate_hz <= 0.0:
            raise ValueError(
                f"tick_rate_hz must be positive; got {tick_rate_hz}"
            )
        self.runtime = runtime
        self.host = host
        self.port = port
        self.tick_rate_hz = tick_rate_hz
        self.running: bool = False
        self.last_frame_time: float | None = None
        self.connections: list[socket.socket] = []
        # Per-connection observer_id filter (None = receives everything).
        self._connection_observer_ids: dict[int, str | None] = {}
        self._lock = threading.Lock()
        self._sock: socket.socket | None = None
        self._accept_thread: threading.Thread | None = None
        self._tick_thread: threading.Thread | None = None

    # --- lifecycle --------------------------------------------------------

    def start(self) -> None:
        """Open the listening socket and start the accept + tick threads."""
        if self.running:
            return
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((self.host, self.port))
        sock.listen()
        sock.settimeout(0.2)
        self._sock = sock
        # If port=0 was requested, OS picked a free port; surface it.
        self.port = sock.getsockname()[1]
        self.running = True
        self._accept_thread = threading.Thread(
            target=self._accept_loop, name="cosmic-runtime-accept", daemon=True
        )
        self._tick_thread = threading.Thread(
            target=self.run_loop, name="cosmic-runtime-tick", daemon=True
        )
        self._accept_thread.start()
        self._tick_thread.start()

    def stop(self) -> None:
        """Stop the loop and close all sockets cleanly."""
        self.running = False
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None
        with self._lock:
            for conn in self.connections:
                try:
                    conn.close()
                except OSError:
                    pass
            self.connections = []
            self._connection_observer_ids.clear()
        for t in (self._accept_thread, self._tick_thread):
            if t is not None and t.is_alive():
                t.join(timeout=2.0)
        self._accept_thread = None
        self._tick_thread = None

    # --- worker loops -----------------------------------------------------

    def _accept_loop(self) -> None:
        while self.running:
            sock = self._sock
            if sock is None:
                return
            try:
                conn, _ = sock.accept()
            except socket.timeout:
                continue
            except OSError:
                return
            conn.settimeout(0.5)
            with self._lock:
                self.connections.append(conn)

    def run_loop(self) -> None:
        """Main tick loop: step the runtime and broadcast each state.

        With observers registered on the runtime, each tick broadcasts
        one ``reality_view`` message per observer (clients that
        ``set_observer_filter`` an id only see their own). Otherwise
        the tick broadcasts a single ``scene_state`` as before.
        """
        period = 1.0 / self.tick_rate_hz
        while self.running:
            start = time.perf_counter()
            if len(self.runtime.observer_manager) > 0:
                try:
                    views = self.runtime.step_all_observers(period)
                except Exception:  # pragma: no cover - defensive
                    views = []
                for view in views:
                    self.broadcast_observer_view(view)
            else:
                try:
                    state = self.runtime.step(period)
                except Exception:  # pragma: no cover - defensive
                    state = self.runtime.last_scene_state
                if state is not None:
                    self.broadcast_state(state)
            self.last_frame_time = time.time()
            elapsed = time.perf_counter() - start
            sleep_for = period - elapsed
            if sleep_for > 0.0:
                time.sleep(sleep_for)

    # --- message senders --------------------------------------------------

    def broadcast_state(self, scene_state: SceneState) -> None:
        """Send the serialized state to every connected subscriber."""
        message = {
            "type": "scene_state",
            "data": scene_state.to_dict(),
        }
        self._send_message(message)

    def broadcast_frame(self, image_path: str) -> None:
        """Send a base64-encoded frame on demand. Optional, off by default."""
        message = {
            "type": "frame",
            "path": image_path,
            "data": encode_frame_to_base64(image_path),
        }
        self._send_message(message)

    def broadcast_observer_view(self, view) -> None:
        """Send a per-observer ``reality_view`` to filtered subscribers.

        ``view`` is a :class:`cosmic_engine.observer.RealityView`. The
        message body is the dict produced by ``view.to_dict()`` plus
        the message ``type`` discriminator. Subscribers can pre-filter
        with :meth:`set_observer_filter`; ``None`` means receive all.
        """
        payload_dict = view.to_dict()
        message = {
            "type": "reality_view",
            "observer_id": payload_dict["observer_id"],
            "scene_state": payload_dict["scene_state"],
            "representation_type": payload_dict["representation_type"],
            "metadata": payload_dict["metadata"],
            "frame": payload_dict["frame"],
        }
        self._send_message(message, observer_id=view.observer_id)

    def set_observer_filter(
        self,
        conn: socket.socket,
        observer_id: str | None,
    ) -> None:
        """Restrict which reality_view messages this connection receives.

        ``None`` means no filter (receive every observer's view).
        """
        with self._lock:
            self._connection_observer_ids[id(conn)] = observer_id

    def _send_message(
        self,
        message: dict,
        *,
        observer_id: str | None = None,
    ) -> None:
        payload = (json.dumps(message) + "\n").encode("utf-8")
        with self._lock:
            dead: list[socket.socket] = []
            for conn in self.connections:
                # If this is an observer-bound message and the
                # subscriber has filtered to a different observer, skip.
                if observer_id is not None:
                    wanted = self._connection_observer_ids.get(id(conn))
                    if wanted is not None and wanted != observer_id:
                        continue
                try:
                    conn.sendall(payload)
                except OSError:
                    dead.append(conn)
            for d in dead:
                self.connections.remove(d)
                self._connection_observer_ids.pop(id(d), None)
                try:
                    d.close()
                except OSError:
                    pass

    # --- introspection ----------------------------------------------------

    def connection_count(self) -> int:
        with self._lock:
            return len(self.connections)
