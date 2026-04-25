"""CPU Gaussian-splat renderer.

For each :class:`GaussianPoint`:

1. Project the world position through the camera basis to a depth
   ``f`` and a screen-space ``(nx, ny)`` in NDC.
2. Convert the world ``sigma`` to a screen-space sigma via
   ``sigma * (width / 2) / (f * tan(fov/2))``.
3. Splat an isotropic 2D Gaussian over a 3-σ bounding box, summing
   ``intensity * exp(-r²/(2σ²)) * color`` into a float accumulator.
4. Tone-map by max-channel normalization.

Loops over points; each point's footprint is computed in NumPy.
Acceptable for ~10k–50k points on CPU; GPU acceleration is the
next phase.
"""

from __future__ import annotations

import math

import numpy as np

from ai_viewer.neural_field.gaussian import GaussianPoint
from cosmic_engine.rendering.simple_camera import SimpleCamera


class GaussianSplatRenderer:
    """Project + splat + tone-map a list of Gaussian points to an RGB image."""

    def __init__(
        self,
        width: int,
        height: int,
        camera: SimpleCamera,
    ) -> None:
        if width <= 0 or height <= 0:
            raise ValueError("width and height must be positive")
        self.width = width
        self.height = height
        self.camera = camera
        camera.validate()

    # --- internals --------------------------------------------------------

    def _camera_basis(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        cam_pos = np.array(
            [
                self.camera.position_m.x,
                self.camera.position_m.y,
                self.camera.position_m.z,
            ],
            dtype=np.float64,
        )
        forward_raw = np.array(
            [
                self.camera.forward.x,
                self.camera.forward.y,
                self.camera.forward.z,
            ],
            dtype=np.float64,
        )
        up_raw = np.array(
            [self.camera.up.x, self.camera.up.y, self.camera.up.z],
            dtype=np.float64,
        )
        forward = forward_raw / np.linalg.norm(forward_raw)
        right = np.cross(forward, up_raw)
        right_norm = np.linalg.norm(right)
        if right_norm == 0.0:
            raise ValueError("camera forward and up are parallel")
        right = right / right_norm
        up = np.cross(right, forward)
        return cam_pos, forward, right, up

    # --- public API -------------------------------------------------------

    def render(self, points: list[GaussianPoint]) -> np.ndarray:
        """Render ``points`` to a ``(H, W, 3)`` ``uint8`` image."""
        color_buf = np.zeros((self.height, self.width, 3), dtype=np.float64)
        if not points:
            return color_buf.astype(np.uint8)

        cam_pos, forward, right, up = self._camera_basis()
        half_fov = math.radians(self.camera.fov_degrees) / 2.0
        scale = math.tan(half_fov)
        if scale <= 0.0:
            return color_buf.astype(np.uint8)
        aspect = self.height / self.width
        screen_sigma_factor = self.width / (2.0 * scale)

        for point in points:
            d = point.position - cam_pos
            f = float(np.dot(d, forward))
            if f <= 0.0:
                continue  # behind camera

            r = float(np.dot(d, right))
            u = float(np.dot(d, up))
            nx = (r / f) / scale
            ny = (u / f) / (scale * aspect)

            if abs(nx) > 2.0 or abs(ny) > 2.0:
                # 3-σ footprint of even a fairly large gaussian won't reach
                # the canvas from here; skip cheaply.
                continue

            px_center = (nx + 1.0) * 0.5 * self.width
            py_center = (1.0 - ny) * 0.5 * self.height
            px_int = int(round(px_center))
            py_int = int(round(py_center))

            screen_sigma = max(1.0, point.sigma * screen_sigma_factor / f)
            radius = int(math.ceil(screen_sigma * 3.0))
            x0 = max(0, px_int - radius)
            x1 = min(self.width, px_int + radius + 1)
            y0 = max(0, py_int - radius)
            y1 = min(self.height, py_int + radius + 1)
            if x1 <= x0 or y1 <= y0:
                continue

            xs = np.arange(x0, x1, dtype=np.float64)
            ys = np.arange(y0, y1, dtype=np.float64)
            dx = xs - px_center
            dy = ys - py_center
            dist_sq = dy[:, None] ** 2 + dx[None, :] ** 2
            kernel = point.intensity * np.exp(
                -dist_sq / (2.0 * screen_sigma * screen_sigma)
            )

            color_buf[y0:y1, x0:x1, 0] += kernel * point.color[0]
            color_buf[y0:y1, x0:x1, 1] += kernel * point.color[1]
            color_buf[y0:y1, x0:x1, 2] += kernel * point.color[2]

        peak = float(color_buf.max())
        if not math.isfinite(peak) or peak <= 0.0:
            return color_buf.astype(np.uint8)
        normalized = np.clip(color_buf / peak * 255.0, 0.0, 255.0)
        return normalized.astype(np.uint8)
