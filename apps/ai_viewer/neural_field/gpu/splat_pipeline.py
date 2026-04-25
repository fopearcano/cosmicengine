"""Backend-aware Gaussian splat pipeline.

The pipeline owns the CPU fallback and a vectorized "mock GPU"
renderer that stands in for a real Vulkan / WebGPU / CUDA backend.
The mock path uses NumPy throughout but vectorizes the entire
projection step (camera basis dot products, frustum culling,
screen-space sigma) before falling back to a per-point footprint
loop with a fixed splat radius — what a real implementation would
do via a parallel kernel.
"""

from __future__ import annotations

import logging
import math

import numpy as np

from ai_viewer.neural_field.gaussian import GaussianPoint
from ai_viewer.neural_field.gpu.buffer import GPUGaussianBuffer
from ai_viewer.neural_field.gpu.device import GPUDevice
from ai_viewer.neural_field.gpu.fallback import CPUSplatFallback
from cosmic_engine.rendering.simple_camera import SimpleCamera

_LOG = logging.getLogger(__name__)


class GaussianSplatPipeline:
    """Pick a backend and render a list of :class:`GaussianPoint`s."""

    def __init__(self, device: GPUDevice) -> None:
        self.device = device
        self._fallback = CPUSplatFallback()
        self._mock_gpu_announced = False

    def render(
        self,
        points: list[GaussianPoint],
        camera: SimpleCamera,
    ) -> np.ndarray:
        """Render via the configured backend, falling back to CPU on demand."""
        buffer = GPUGaussianBuffer.from_gaussian_points(points)
        backend = self.device.get_backend()
        if not self.device.is_available() or backend == "none":
            # No device available — fall back to CPU silently.
            return self._fallback.render(buffer, camera)
        if backend == "mock_gpu":
            return self._mock_gpu_render(buffer, camera)
        return self._fallback.render(buffer, camera)

    # --- backends --------------------------------------------------------

    def _mock_gpu_render(
        self,
        buffer: GPUGaussianBuffer,
        camera: SimpleCamera,
    ) -> np.ndarray:
        """Vectorized stand-in for a real GPU kernel."""
        if not self._mock_gpu_announced:
            _LOG.info(
                "GaussianSplatPipeline: simulating GPU path via vectorized "
                "NumPy (no real GPU bound yet)"
            )
            self._mock_gpu_announced = True

        camera.validate()
        width = camera.image_width
        height = camera.image_height
        n = len(buffer)
        if n == 0:
            return np.zeros((height, width, 3), dtype=np.uint8)

        cam_pos = np.array(
            [
                camera.position_m.x,
                camera.position_m.y,
                camera.position_m.z,
            ],
            dtype=np.float64,
        )
        forward = np.array(
            [camera.forward.x, camera.forward.y, camera.forward.z],
            dtype=np.float64,
        )
        up_raw = np.array(
            [camera.up.x, camera.up.y, camera.up.z], dtype=np.float64
        )
        forward = forward / np.linalg.norm(forward)
        right = np.cross(forward, up_raw)
        right_norm = np.linalg.norm(right)
        if right_norm == 0.0:
            return np.zeros((height, width, 3), dtype=np.uint8)
        right = right / right_norm
        up = np.cross(right, forward)

        half_fov = math.radians(camera.fov_degrees) / 2.0
        scale = math.tan(half_fov)
        if scale <= 0.0:
            return np.zeros((height, width, 3), dtype=np.uint8)
        aspect = height / width

        # Projection in one shot:
        d = buffer.positions - cam_pos
        f = d @ forward
        r = d @ right
        u = d @ up

        valid = f > 0.0
        safe_f = np.where(valid, f, 1.0)
        nx = (r / safe_f) / scale
        ny = (u / safe_f) / (scale * aspect)
        valid &= np.abs(nx) <= 2.0
        valid &= np.abs(ny) <= 2.0

        if not valid.any():
            return np.zeros((height, width, 3), dtype=np.uint8)

        screen_sigma_factor = width / (2.0 * scale)
        screen_sigma = np.where(
            valid,
            np.maximum(1.0, buffer.sigmas * screen_sigma_factor / safe_f),
            1.0,
        )

        px = ((nx + 1.0) * 0.5 * width).astype(np.int64)
        py = ((1.0 - ny) * 0.5 * height).astype(np.int64)

        color_buf = np.zeros((height, width, 3), dtype=np.float64)

        # Real GPUs would handle this in a kernel; the mock loops only
        # over visible points (valid array already culls everything else).
        valid_indices = np.where(valid)[0]
        for idx in valid_indices:
            sigma_s = float(screen_sigma[idx])
            radius = int(math.ceil(sigma_s * 3.0))
            cx = int(px[idx])
            cy = int(py[idx])
            x0 = max(0, cx - radius)
            x1 = min(width, cx + radius + 1)
            y0 = max(0, cy - radius)
            y1 = min(height, cy + radius + 1)
            if x1 <= x0 or y1 <= y0:
                continue

            xs = np.arange(x0, x1, dtype=np.float64) - cx
            ys = np.arange(y0, y1, dtype=np.float64) - cy
            dist_sq = ys[:, None] ** 2 + xs[None, :] ** 2
            kernel = float(buffer.intensities[idx]) * np.exp(
                -dist_sq / (2.0 * sigma_s * sigma_s)
            )
            color = buffer.colors[idx]
            color_buf[y0:y1, x0:x1, 0] += kernel * color[0]
            color_buf[y0:y1, x0:x1, 1] += kernel * color[1]
            color_buf[y0:y1, x0:x1, 2] += kernel * color[2]

        if not np.isfinite(color_buf).all():
            color_buf = np.nan_to_num(
                color_buf, nan=0.0, posinf=255.0, neginf=0.0
            )
        peak = float(color_buf.max())
        if peak <= 0.0:
            return color_buf.astype(np.uint8)
        normalized = color_buf / peak * 255.0
        return np.clip(normalized, 0.0, 255.0).astype(np.uint8)
