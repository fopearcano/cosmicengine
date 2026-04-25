"""TCP client for the CosmicEngine runtime server.

Speaks the same newline-delimited JSON protocol as
:class:`cosmic_engine.runtime.server.RuntimeServer`. No external
networking dependencies — :mod:`socket` and :mod:`json` only.
"""

from __future__ import annotations

import base64
import json
import socket

from ai_viewer.config import AIViewerConfig


class RuntimeClient:
    """Stateful subscriber that reads SceneState / frame messages."""

    def __init__(self, config: AIViewerConfig) -> None:
        config.validate()
        self.config = config
        self._sock: socket.socket | None = None
        self._buffer = bytearray()

    # --- lifecycle --------------------------------------------------------

    def connect(self, timeout: float = 5.0) -> None:
        """Open a TCP connection to ``config.server_host:server_port``."""
        if self._sock is not None:
            return
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        sock.connect((self.config.server_host, self.config.server_port))
        self._sock = sock
        self._buffer = bytearray()

    def disconnect(self) -> None:
        """Close the connection if open."""
        if self._sock is None:
            return
        try:
            self._sock.close()
        except OSError:
            pass
        self._sock = None
        self._buffer = bytearray()

    # --- low-level message reader ----------------------------------------

    def receive_message(self) -> dict | None:
        """Pull one newline-terminated JSON message from the stream.

        Returns ``None`` if the server closed the connection. Raises
        :class:`ConnectionError` if not connected; underlying socket
        exceptions propagate to the caller.
        """
        if self._sock is None:
            raise ConnectionError("RuntimeClient not connected")
        while b"\n" not in self._buffer:
            try:
                chunk = self._sock.recv(4096)
            except socket.timeout:
                return None
            if not chunk:
                return None
            self._buffer.extend(chunk)
        line, _, rest = bytes(self._buffer).partition(b"\n")
        self._buffer = bytearray(rest)
        if not line:
            return None
        return json.loads(line.decode("utf-8"))

    # --- typed accessors --------------------------------------------------

    def receive_scene_state(self) -> dict | None:
        """Return the next ``scene_state`` payload, or ``None``."""
        message = self.receive_message()
        if message is None:
            return None
        if message.get("type") != "scene_state":
            return None
        return message.get("data")

    def receive_frame(self) -> bytes | None:
        """Return the next decoded ``frame`` payload bytes, or ``None``."""
        message = self.receive_message()
        if message is None:
            return None
        if message.get("type") != "frame":
            return None
        encoded = message.get("data", "")
        if not encoded:
            return None
        return base64.b64decode(encoded)
