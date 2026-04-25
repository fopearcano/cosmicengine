"""Backend selection for the GPU splat pipeline.

No real GPU detection is performed yet. ``"none"`` flags an
explicitly-disabled device, ``"cpu"`` falls back to the Phase 24
NumPy renderer, and ``"mock_gpu"`` uses a vectorized NumPy
simulation that stands in for a future Vulkan / WebGPU / CUDA
implementation.
"""

from __future__ import annotations


_VALID_BACKENDS = ("none", "cpu", "mock_gpu", "webgpu")


class GPUDevice:
    """Resolve a rendering backend for the splat pipeline."""

    def __init__(self, backend: str = "auto") -> None:
        if backend == "auto":
            backend = self._detect_backend()
        if backend not in _VALID_BACKENDS:
            raise ValueError(
                f"unknown GPU backend {backend!r}; expected one of "
                f"{list(_VALID_BACKENDS)}"
            )
        self._backend = backend

    @staticmethod
    def _detect_backend() -> str:
        # No real GPU detection here. Phase 26 ships the architecture; the
        # hardware probe lands when a real backend is bound.
        return "cpu"

    def is_available(self) -> bool:
        """``True`` when the device can render (i.e. backend != ``"none"``)."""
        return self._backend != "none"

    def get_backend(self) -> str:
        """Return the resolved backend string."""
        return self._backend
