"""CPU fallback renderer for the GPU splat pipeline.

When the device backend is ``"cpu"`` (or the mock path is unavailable),
the pipeline routes through this thin shim that re-uses the Phase 24
:class:`GaussianSplatRenderer`. Keeping it isolated means the
pipeline's selection logic stays simple.
"""

from __future__ import annotations

import numpy as np

from ai_viewer.neural_field.gaussian import GaussianPoint
from ai_viewer.neural_field.gpu.buffer import GPUGaussianBuffer
from ai_viewer.neural_field.splat_renderer import GaussianSplatRenderer
from cosmic_engine.rendering.simple_camera import SimpleCamera


class CPUSplatFallback:
    """Render a :class:`GPUGaussianBuffer` on the CPU."""

    def render(
        self,
        buffer: GPUGaussianBuffer,
        camera: SimpleCamera,
    ) -> np.ndarray:
        """Reconstruct points and delegate to :class:`GaussianSplatRenderer`."""
        n = len(buffer)
        points: list[GaussianPoint] = []
        for i in range(n):
            points.append(
                GaussianPoint(
                    position=buffer.positions[i],
                    color=buffer.colors[i],
                    intensity=float(buffer.intensities[i]),
                    sigma=float(buffer.sigmas[i]),
                )
            )
        renderer = GaussianSplatRenderer(
            camera.image_width, camera.image_height, camera
        )
        return renderer.render(points)
