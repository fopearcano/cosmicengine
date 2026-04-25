"""Pack Gaussian points into a GPU-aligned float32 buffer.

Layout (8 floats, 32 bytes per point — matches WGSL's vec4 alignment):

    [pos.x, pos.y, pos.z, intensity, color.r, color.g, color.b, sigma]

The same packing is used for the CPU shadow copy in
:func:`WebGPUGaussianBuffer.pack_points` so tests can verify the
layout without a GPU.
"""

from __future__ import annotations

import numpy as np

from ai_viewer.neural_field.gaussian import GaussianPoint


# 8 floats × 4 bytes = 32 bytes per Gaussian.
FLOATS_PER_POINT: int = 8
BYTES_PER_POINT: int = FLOATS_PER_POINT * 4


def pack_points(points: list[GaussianPoint]) -> np.ndarray:
    """Pack ``points`` into a flat ``(N * FLOATS_PER_POINT,)`` float32 array."""
    n = len(points)
    out = np.zeros((n, FLOATS_PER_POINT), dtype=np.float32)
    for i, p in enumerate(points):
        out[i, 0] = p.position[0]
        out[i, 1] = p.position[1]
        out[i, 2] = p.position[2]
        out[i, 3] = p.intensity
        out[i, 4] = p.color[0]
        out[i, 5] = p.color[1]
        out[i, 6] = p.color[2]
        out[i, 7] = p.sigma
    return out.reshape(-1)


class WebGPUGaussianBuffer:
    """A GPU-side storage buffer holding the packed Gaussian field."""

    def __init__(self) -> None:
        self.gpu_buffer = None
        self.count: int = 0
        self._cpu_copy: np.ndarray | None = None

    def upload(
        self,
        points: list[GaussianPoint],
        device: "WebGPUDevice",
    ) -> None:
        """Pack ``points`` and upload to the device. Stores a CPU shadow."""
        packed = pack_points(points)
        self.count = len(points)
        self._cpu_copy = packed
        if device.device is None:
            return  # CPU-only fallback; the shadow is enough
        try:
            import wgpu

            usage = wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST
            byte_size = max(packed.nbytes, BYTES_PER_POINT)  # at least 1 dummy
            buf = device.device.create_buffer(size=byte_size, usage=usage)
            if packed.nbytes:
                device.queue.write_buffer(buf, 0, packed.tobytes())
            self.gpu_buffer = buf
        except Exception:
            # Fall back to CPU shadow only; renderer will detect via gpu_buffer is None.
            self.gpu_buffer = None

    def cpu_copy(self) -> np.ndarray:
        """Return the most-recently-uploaded packed array (or empty)."""
        if self._cpu_copy is None:
            return np.zeros((0,), dtype=np.float32)
        return self._cpu_copy

    def release(self) -> None:
        """Release the GPU buffer, if any. Safe to call repeatedly."""
        if self.gpu_buffer is not None:
            try:
                self.gpu_buffer.destroy()
            except Exception:
                pass
        self.gpu_buffer = None
        self._cpu_copy = None
        self.count = 0
