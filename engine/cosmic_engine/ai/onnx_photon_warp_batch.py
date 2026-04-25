"""Batch ONNX inference for photon-space neural warping.

Where :class:`cosmic_engine.ai.onnx_photon_warp.ONNXPhotonWarpModel`
runs one inference per photon, this class runs **one inference per
frame** over an entire ``(N, 9)`` batch — the speedup is significant
for large catalogs because per-call overhead in ``onnxruntime`` is
amortized across thousands of samples.

Schema (rows align with the per-photon spec):

    input column         meaning
    ------------------   ------------------------
    [0..2]               direction (x, y, z)
    [3]                  brightness
    [4..6]               color (r, g, b) in 0..255
    [7]                  observer beta = |v|/c
    [8]                  observer warp_factor

    output column        meaning
    ------------------   ------------------------
    [0..2]               new direction (renormalized)
    [3]                  new brightness (clamped >= 0)
    [4..6]               new color (clamped to 0..255)

This class **does not** silently fall back. Construction raises if
the model cannot be loaded; ``predict_batch`` raises on bad input
shape or inference failure. Higher-level integrations (see
:mod:`cosmic_engine.perception.vectorized_ai_transform`) catch the
exception and route to the deterministic vectorized transform.
"""

from __future__ import annotations

import numpy as np

from cosmic_engine.ai.onnx_model import ONNXModelWrapper


_BRIGHTNESS_CEILING = 1.0e18


class BatchONNXPhotonWarpModel:
    """Single-frame batch inference for photon-space warping."""

    def __init__(self, model_path: str) -> None:
        self.model_path = model_path
        self.last_error: str | None = None
        self._wrapper = ONNXModelWrapper(model_path)

    @property
    def session(self):
        return self._wrapper.session

    def predict_batch(self, input_array: np.ndarray) -> np.ndarray:
        """Run one ONNX inference over a ``(N, 9)`` array; return ``(N, 7)``.

        The returned directions are renormalized to unit length, the
        brightness column is clamped to ``[0, _BRIGHTNESS_CEILING]``,
        and color columns are clipped to ``[0, 255]``.
        """
        if input_array.ndim != 2 or input_array.shape[1] != 9:
            raise ValueError(
                f"BatchONNXPhotonWarpModel.predict_batch expects shape "
                f"(N, 9); got {input_array.shape}"
            )
        n = int(input_array.shape[0])
        if n == 0:
            return np.zeros((0, 7), dtype=np.float32)

        feed = np.ascontiguousarray(input_array, dtype=np.float32)
        try:
            raw = self._wrapper.session.run(
                [self._wrapper.output_name],
                {self._wrapper.input_name: feed},
            )[0]
        except Exception as e:
            self.last_error = f"inference failed: {e}"
            raise RuntimeError(
                f"BatchONNXPhotonWarpModel inference failed: {e}"
            ) from e

        out = np.asarray(raw, dtype=np.float32)
        if out.ndim != 2 or out.shape != (n, 7):
            self.last_error = f"unexpected output shape {out.shape}"
            raise RuntimeError(
                f"BatchONNXPhotonWarpModel produced unexpected output "
                f"shape {out.shape}; expected ({n}, 7)"
            )

        # Renormalize directions; rows that come back zero stay zero
        # (they will be filtered by the renderer).
        dirs = out[:, 0:3]
        norms = np.linalg.norm(dirs, axis=1, keepdims=True)
        safe = np.where(norms == 0.0, 1.0, norms)
        normalized = np.where(norms == 0.0, dirs, dirs / safe)

        brightness = np.clip(out[:, 3], 0.0, _BRIGHTNESS_CEILING)
        colors = np.clip(out[:, 4:7], 0.0, 255.0)

        result = np.empty((n, 7), dtype=np.float32)
        result[:, 0:3] = normalized
        result[:, 3] = brightness
        result[:, 4:7] = colors

        # Healthy run; clear any earlier error so confidence reflects state.
        self.last_error = None
        return result

    def confidence(self) -> float:
        return 0.3 if self.last_error else 0.85
