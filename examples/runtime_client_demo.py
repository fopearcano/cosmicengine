"""Demo: connect to a running CosmicEngine RuntimeServer and read a few messages."""

from __future__ import annotations

import argparse
import json
import socket
import sys


def _read_message(sock: socket.socket, buffer: bytearray) -> dict | None:
    """Pull one newline-delimited JSON message from ``sock`` (blocking)."""
    while b"\n" not in buffer:
        chunk = sock.recv(4096)
        if not chunk:
            return None
        buffer.extend(chunk)
    line, _, rest = buffer.partition(b"\n")
    buffer[:] = rest
    if not line:
        return None
    return json.loads(line.decode("utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--messages", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=5.0)
    args = parser.parse_args()

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(args.timeout)
    try:
        sock.connect((args.host, args.port))
    except OSError as e:
        print(f"connection failed: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"connected to {args.host}:{args.port}")
    buffer = bytearray()
    received = 0
    try:
        while received < args.messages:
            try:
                message = _read_message(sock, buffer)
            except socket.timeout:
                print("timed out waiting for message", file=sys.stderr)
                break
            if message is None:
                print("server closed the connection")
                break
            received += 1
            if message.get("type") == "scene_state":
                data = message["data"]
                print(
                    f"[{received}] julian_date={data['julian_date']:.6f}  "
                    f"active_objects={data['active_objects']}  "
                    f"physics={data['physics_backend']}  "
                    f"types={data['object_type_counts']}"
                )
            elif message.get("type") == "frame":
                print(
                    f"[{received}] frame at {message.get('path')} "
                    f"({len(message.get('data', '')) // 1024} KiB base64)"
                )
            else:
                print(f"[{received}] unknown message type: {message}")
    finally:
        sock.close()


if __name__ == "__main__":
    main()
