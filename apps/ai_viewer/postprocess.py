"""Modular CPU-only frame postprocessors.

Each processor takes a :class:`PIL.Image.Image` and returns one of the
same dimensions. The base class is a no-op so callers can wire it in
unconditionally; concrete processors apply contrast, brightness, color
shift, or chain other processors. No GPU, no neural networks; that's
the next phase.
"""

from __future__ import annotations

from PIL import Image, ImageEnhance


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
