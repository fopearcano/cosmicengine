"""In-memory frame buffer with PPM I/O and an ASCII preview.

Plain-PPM only (P3, the format the existing renderer writes). No
external image dependencies; NumPy is used for the pixel array since
it is already a project dependency.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


_ASCII_RAMP = " .:-=+*#%@"


class FrameBuffer:
    """A ``(height, width, 3)`` uint8 image with PPM round-trip support."""

    def __init__(self, width: int = 0, height: int = 0) -> None:
        self.width = width
        self.height = height
        self.pixels = np.zeros((max(height, 0), max(width, 0), 3), dtype=np.uint8)

    # --- mutation ---------------------------------------------------------

    def clear(self) -> None:
        """Reset every pixel to black without resizing."""
        self.pixels[:] = 0

    # --- I/O --------------------------------------------------------------

    def load_ppm_bytes(self, data: bytes) -> None:
        """Parse a plain (P3) PPM byte string and replace the pixel array."""
        text = data.decode("ascii", errors="strict")
        tokens: list[str] = []
        for line in text.splitlines():
            tokens.extend(line.split("#", 1)[0].split())

        if not tokens or tokens[0] != "P3":
            raise ValueError("FrameBuffer.load_ppm_bytes: only plain (P3) PPM is supported")
        if len(tokens) < 4:
            raise ValueError("FrameBuffer.load_ppm_bytes: malformed PPM header")

        width = int(tokens[1])
        height = int(tokens[2])
        max_value = int(tokens[3])
        if width <= 0 or height <= 0:
            raise ValueError("FrameBuffer.load_ppm_bytes: non-positive dimensions")
        if max_value <= 0:
            raise ValueError("FrameBuffer.load_ppm_bytes: invalid maxval")

        body = tokens[4:]
        expected = width * height * 3
        if len(body) < expected:
            raise ValueError(
                f"FrameBuffer.load_ppm_bytes: truncated body (expected {expected}, got {len(body)})"
            )

        arr = np.array(body[:expected], dtype=np.float64).reshape(height, width, 3)
        if max_value != 255:
            arr = arr * (255.0 / max_value)
        self.pixels = np.clip(arr, 0.0, 255.0).astype(np.uint8)
        self.width = width
        self.height = height

    def save_ppm(self, path: str) -> None:
        """Write the buffer to ``path`` as a plain (P3) PPM."""
        if self.width <= 0 or self.height <= 0:
            raise ValueError("FrameBuffer.save_ppm: empty buffer")
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="ascii") as fh:
            fh.write("P3\n")
            fh.write(f"{self.width} {self.height}\n")
            fh.write("255\n")
            for row in self.pixels:
                fh.write(
                    " ".join(
                        f"{int(r)} {int(g)} {int(b)}" for r, g, b in row
                    )
                )
                fh.write("\n")

    # --- preview ----------------------------------------------------------

    def to_ascii_preview(self, max_width: int = 80) -> str:
        """Return a downsampled grayscale ASCII rendering."""
        if self.width <= 0 or self.height <= 0:
            return ""
        if max_width <= 0:
            raise ValueError("max_width must be positive")
        if self.width <= max_width:
            scaled = self.pixels
            scaled_w = self.width
            scaled_h = self.height
        else:
            scaled_w = max_width
            # account for the ~2:1 char height/width ratio so the preview
            # stays roughly proportional in a terminal
            scaled_h = max(1, self.height * scaled_w // self.width // 2)
            row_step = self.height / scaled_h
            col_step = self.width / scaled_w
            rows = (np.arange(scaled_h) * row_step).astype(np.int64)
            cols = (np.arange(scaled_w) * col_step).astype(np.int64)
            scaled = self.pixels[rows[:, None], cols[None, :]]

        luminance = (
            0.299 * scaled[..., 0]
            + 0.587 * scaled[..., 1]
            + 0.114 * scaled[..., 2]
        )
        levels = len(_ASCII_RAMP) - 1
        idx = np.clip(
            (luminance * levels / 255.0).astype(np.int64), 0, levels
        )
        ramp = np.array(list(_ASCII_RAMP))
        chars = ramp[idx]
        return "\n".join("".join(row) for row in chars)
