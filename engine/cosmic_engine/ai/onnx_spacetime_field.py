"""ONNX-backed implementation of :class:`SpacetimeFieldModel`.

Schema:

    input  shape (1, 6) = [pos.x, pos.y, pos.z, dir.x, dir.y, dir.z]
    output shape (1, 3) = [a.x, a.y, a.z]

If ``direction`` is omitted, zeros are fed in for the last three
components. The result is clipped to a finite magnitude and scrubbed
of NaNs so a degenerate model can't poison the geodesic integrator.

Construction errors propagate (so the caller can decide whether to
fall back); inference errors are raised as :class:`RuntimeError`
with ``last_error`` set.
"""

from __future__ import annotations

import math

import numpy as np

from cosmic_engine.ai.onnx_model import ONNXModelWrapper
from cosmic_engine.ai.spacetime_field import SpacetimeFieldModel


# Cap acceleration magnitude. The integrator clamps deflection per-step
# anyway; this is belt-and-braces for "model returned huge number".
_ACCEL_MAGNITUDE_CEILING: float = 1.0e30


class ONNXSpacetimeField(SpacetimeFieldModel):
    """Neural spacetime curvature backed by an ONNX session."""

    def __init__(self, model_path: str) -> None:
        self.model_path = model_path
        self.last_error: str | None = None
        self._wrapper = ONNXModelWrapper(model_path)

    def query_acceleration(
        self,
        position: np.ndarray,
        direction: np.ndarray | None = None,
    ) -> np.ndarray:
        pos = np.asarray(position, dtype=np.float32).reshape(3)
        if direction is None:
            dir_v = np.zeros(3, dtype=np.float32)
        else:
            dir_v = np.asarray(direction, dtype=np.float32).reshape(3)
        feed = np.empty((1, 6), dtype=np.float32)
        feed[0, 0:3] = pos
        feed[0, 3:6] = dir_v

        try:
            raw = self._wrapper.session.run(
                [self._wrapper.output_name],
                {self._wrapper.input_name: feed},
            )[0]
        except Exception as e:
            self.last_error = f"inference failed: {e}"
            raise RuntimeError(
                f"ONNXSpacetimeField inference failed: {e}"
            ) from e

        out = np.asarray(raw, dtype=np.float32)
        if out.ndim == 2:
            out = out[0]
        if out.shape != (3,):
            self.last_error = f"unexpected output shape {out.shape}"
            raise RuntimeError(
                f"ONNXSpacetimeField produced unexpected output shape "
                f"{out.shape}; expected (3,)"
            )
        # Clean up: clamp magnitude, scrub non-finite values.
        out = np.where(np.isfinite(out), out, 0.0)
        magnitude = float(np.linalg.norm(out))
        if magnitude > _ACCEL_MAGNITUDE_CEILING:
            out = out * (_ACCEL_MAGNITUDE_CEILING / magnitude)
        if not math.isfinite(magnitude):
            out = np.zeros(3, dtype=np.float32)

        self.last_error = None
        return out.astype(np.float64)

    def confidence(self) -> float:
        return 0.2 if self.last_error else 0.9
