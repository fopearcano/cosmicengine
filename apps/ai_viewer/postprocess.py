"""Modular CPU-only frame postprocessors.

Each processor takes a :class:`PIL.Image.Image` and returns one of the
same dimensions. The base class is a no-op so callers can wire it in
unconditionally; concrete processors apply contrast, brightness, color
shift, chain other processors, or wrap a fallible processor with safe
error handling. Neural processors live in
:mod:`ai_viewer.neural_postprocess`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from PIL import Image, ImageEnhance

if TYPE_CHECKING:
    from ai_viewer.config import AIViewerConfig


class FramePostProcessor:
    """Pluggable per-frame transform. The base class is a pass-through."""

    def process(self, image: Image.Image) -> Image.Image:
        return image


class ContrastBoostProcessor(FramePostProcessor):
    """Multiply contrast around mid-gray by ``factor`` (1.0 = no change)."""

    def __init__(self, factor: float = 1.3) -> None:
        if factor <= 0.0:
            raise ValueError("ContrastBoostProcessor.factor must be positive")
        self.factor = factor

    def process(self, image: Image.Image) -> Image.Image:
        return ImageEnhance.Contrast(image).enhance(self.factor)


class BrightnessProcessor(FramePostProcessor):
    """Scale every pixel by ``factor`` (1.0 = no change)."""

    def __init__(self, factor: float = 1.2) -> None:
        if factor < 0.0:
            raise ValueError("BrightnessProcessor.factor must be non-negative")
        self.factor = factor

    def process(self, image: Image.Image) -> Image.Image:
        return ImageEnhance.Brightness(image).enhance(self.factor)


class ColorShiftProcessor(FramePostProcessor):
    """Per-channel additive shift, clamped to ``[0, 255]`` via a LUT."""

    def __init__(
        self,
        r_shift: int = 0,
        g_shift: int = 0,
        b_shift: int = 0,
    ) -> None:
        self.shifts = (int(r_shift), int(g_shift), int(b_shift))

    @staticmethod
    def _lut(shift: int) -> list[int]:
        return [max(0, min(255, i + shift)) for i in range(256)]

    def process(self, image: Image.Image) -> Image.Image:
        if image.mode != "RGB":
            image = image.convert("RGB")
        r, g, b = image.split()
        r = r.point(self._lut(self.shifts[0]))
        g = g.point(self._lut(self.shifts[1]))
        b = b.point(self._lut(self.shifts[2]))
        return Image.merge("RGB", (r, g, b))


class CompositeProcessor(FramePostProcessor):
    """Apply each processor in order; an empty list is a no-op."""

    def __init__(self, processors: list[FramePostProcessor] | None = None) -> None:
        self.processors: list[FramePostProcessor] = list(processors or [])

    def process(self, image: Image.Image) -> Image.Image:
        for proc in self.processors:
            image = proc.process(image)
        return image


class SafeProcessor(FramePostProcessor):
    """Wrap another processor; any exception falls back to the input image.

    Records the most recent error in ``last_error`` so callers can
    surface it once rather than logging on every frame.
    """

    def __init__(self, wrapped: FramePostProcessor) -> None:
        self.wrapped = wrapped
        self.last_error: str | None = None

    def process(self, image: Image.Image) -> Image.Image:
        try:
            result = self.wrapped.process(image)
        except Exception as e:  # pragma: no cover - defensive
            self.last_error = str(e)
            return image
        # If the wrapped processor advertises its own last_error,
        # surface it through SafeProcessor too.
        inner_error = getattr(self.wrapped, "last_error", None)
        if inner_error:
            self.last_error = inner_error
        return result


def build_postprocessor_from_config(
    config: "AIViewerConfig",
) -> FramePostProcessor:
    """Pick the processor implied by the viewer config.

    - ``neural_model_path`` set: load an ONNX neural processor wrapped
      in :class:`SafeProcessor`. The wrapper guarantees the viewer
      never crashes on inference errors.
    - else if ``enable_postprocess`` is true: a deterministic
      :class:`CompositeProcessor` (gentle contrast + brightness boost).
    - else: a pass-through :class:`FramePostProcessor`.
    """
    # Lazy import keeps onnxruntime out of the loader path until needed.
    if config.neural_model_path:
        from ai_viewer.neural_postprocess import ONNXFrameProcessor

        size: tuple[int, int] | None = None
        if (
            config.neural_input_width is not None
            and config.neural_input_height is not None
        ):
            size = (config.neural_input_width, config.neural_input_height)
        return SafeProcessor(
            ONNXFrameProcessor(
                model_path=config.neural_model_path,
                input_size=size,
                normalize=config.neural_normalize,
            )
        )
    if config.enable_postprocess:
        return CompositeProcessor(
            [
                ContrastBoostProcessor(factor=1.2),
                BrightnessProcessor(factor=1.1),
            ]
        )
    return FramePostProcessor()
