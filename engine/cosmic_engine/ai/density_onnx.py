"""ONNX-backed density-field enhancement model.

Mirrors the structure of :mod:`cosmic_engine.ai.onnx_warp` but for
volumetric data. The bundled mock model
(``data/density_upscaler.onnx``) takes ``(1, 1, 64, 64, 64)`` input
and returns ``(1, 1, 128, 128, 128)`` via a trilinear ``Resize`` op.

Any inference failure (shape mismatch, broken model, runtime error)
silently falls back to :class:`SimpleDensityEnhancer` so the public
``enhance`` always returns a valid grid.
"""

from __future__ import annotations

import numpy as np

from cosmic_engine.ai.density_model import (
    DensityFieldModel,
    SimpleDensityEnhancer,
)
from cosmic_engine.ai.density_utils import normalize_density_grid
from cosmic_engine.ai.onnx_model import ONNXModelWrapper


class ONNXDensityModel(DensityFieldModel):
    """Density enhancer that runs an ONNX session under the hood."""

    def __init__(self, model_path: str) -> None:
        self.model = ONNXModelWrapper(model_path)
        self._fallback = SimpleDensityEnhancer()
        self._last_inference_ok = True

    def _try_infer(self, grid: np.ndarray) -> np.ndarray:
        if grid.ndim != 3:
            raise ValueError(f"expected 3D grid; got shape {grid.shape}")
        arr = np.asarray(grid, dtype=np.float32).reshape(
            1, 1, *grid.shape
        )
        result = self.model.session.run(
            [self.model.output_name],
            {self.model.input_name: arr},
        )[0]
        out = np.asarray(result).reshape(result.shape[-3:]).astype(np.float64)
        return out

    def enhance(self, grid: np.ndarray) -> np.ndarray:
        try:
            enhanced = self._try_infer(grid)
            self._last_inference_ok = True
            return normalize_density_grid(enhanced)
        except Exception:
            self._last_inference_ok = False
            return self._fallback.enhance(grid)

    def confidence(self) -> float:
        return 0.8 if self._last_inference_ok else 0.3
