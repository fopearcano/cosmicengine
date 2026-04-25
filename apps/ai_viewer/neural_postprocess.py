"""Neural frame postprocessor backed by ONNX Runtime.

Wraps :mod:`onnxruntime` for image-shaped tensors. Always returns a
valid image: missing files, malformed models, and inference failures
all flip ``self.last_error`` and produce the original frame unchanged.

Pillow handles the resize/colorspace path; NumPy handles the tensor
reshuffles. No PyTorch, no OpenCV, no GPU.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
from PIL import Image

from ai_viewer.postprocess import FramePostProcessor


class ONNXFrameProcessor(FramePostProcessor):
    """Image → ONNX session → image, with safe fallback to the input."""

    def __init__(
        self,
        model_path: str,
        input_size: tuple[int, int] | None = None,
        normalize: bool = True,
    ) -> None:
        self.model_path = model_path
        self.input_size = input_size
        self.normalize = normalize
        self.last_error: str | None = None
        self.session: Any = None
        self._input_name: str | None = None
        self._output_name: str | None = None

        if not os.path.isfile(model_path):
            self.last_error = f"model not found: {model_path}"
            return
        try:
            import onnxruntime as ort
        except Exception as e:
            self.last_error = f"onnxruntime unavailable: {e}"
            return
        try:
            self.session = ort.InferenceSession(
                model_path, providers=["CPUExecutionProvider"]
            )
            self._input_name = self.session.get_inputs()[0].name
            self._output_name = self.session.get_outputs()[0].name
        except Exception as e:
            self.last_error = f"failed to load model: {e}"
            self.session = None

    def process(self, image: Image.Image) -> Image.Image:
        if self.session is None:
            return image
        try:
            original_size = image.size  # (W, H)
            if image.mode != "RGB":
                image = image.convert("RGB")

            target = self.input_size if self.input_size is not None else original_size
            if image.size != target:
                model_input_image = image.resize(target, Image.LANCZOS)
            else:
                model_input_image = image

            arr = np.asarray(model_input_image, dtype=np.float32)
            if self.normalize:
                arr = arr / 255.0
            # (H, W, 3) → (1, 3, H, W)
            chw = arr.transpose(2, 0, 1)[None, ...].astype(np.float32)

            outputs = self.session.run(
                [self._output_name], {self._input_name: chw}
            )
            out = np.asarray(outputs[0])
            if out.ndim == 4:
                out = out[0]
            if out.ndim == 3 and out.shape[0] == 3:
                out = out.transpose(1, 2, 0)
            if out.ndim != 3 or out.shape[2] != 3:
                raise ValueError(f"unexpected output shape {out.shape}")

            if self.normalize:
                out = out * 255.0
            out = np.clip(out, 0.0, 255.0).astype(np.uint8)
            result = Image.fromarray(out, mode="RGB")
            if result.size != original_size:
                result = result.resize(original_size, Image.LANCZOS)
            return result
        except Exception as e:
            self.last_error = f"inference failed: {e}"
            return image
