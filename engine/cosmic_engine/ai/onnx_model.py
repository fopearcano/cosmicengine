"""Thin wrapper around ``onnxruntime.InferenceSession``.

Keeps ONNX specifics out of :mod:`cosmic_engine.ai.onnx_warp`: the
wrapper handles session construction, shape reconciliation, and error
normalization so callers deal in plain Python lists of floats.
"""

from __future__ import annotations

import os
from typing import Any


class ONNXModelWrapper:
    """Load an ONNX model and run single-sample inference.

    ``predict(input_vector)`` accepts a flat list of floats, reshapes it
    to the model's declared input shape (filling any dynamic / unknown
    dimensions with 1), runs the session, and returns the first output
    flattened to a Python list.

    Raises:
        FileNotFoundError: if the model file does not exist.
        ValueError: if the file exists but cannot be loaded as ONNX.
        RuntimeError: if inference fails at call time.
    """

    def __init__(self, model_path: str) -> None:
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"ONNX model not found: {model_path}")

        # Import lazily so the rest of the package stays importable even
        # when onnxruntime is missing.
        try:
            import onnxruntime as ort
        except ImportError as e:
            raise RuntimeError(
                "onnxruntime is required to use ONNXModelWrapper"
            ) from e

        try:
            self.session = ort.InferenceSession(
                model_path, providers=["CPUExecutionProvider"]
            )
        except Exception as e:
            raise ValueError(f"invalid ONNX model {model_path!r}: {e}") from e

        inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()
        if not inputs or not outputs:
            raise ValueError(
                f"ONNX model {model_path!r} has no inputs or outputs"
            )
        self.input_name: str = inputs[0].name
        self.output_name: str = outputs[0].name
        self._input_shape: list[Any] = list(inputs[0].shape)

    def _resolved_shape(self, length: int) -> list[int]:
        """Replace dynamic dimensions with 1, pad / validate against ``length``."""
        resolved = [d if isinstance(d, int) and d > 0 else 1 for d in self._input_shape]
        product = 1
        for d in resolved:
            product *= d
        if product != length:
            raise ValueError(
                f"input length {length} does not fit model shape {self._input_shape}"
            )
        return resolved

    def predict(self, input_vector: list[float]) -> list[float]:
        """Run inference on ``input_vector`` and return a flat list of outputs."""
        import numpy as np

        shape = self._resolved_shape(len(input_vector))
        arr = np.asarray(input_vector, dtype=np.float32).reshape(shape)
        try:
            result = self.session.run(
                [self.output_name], {self.input_name: arr}
            )
        except Exception as e:
            raise RuntimeError(f"ONNX inference failed: {e}") from e
        return result[0].flatten().tolist()
