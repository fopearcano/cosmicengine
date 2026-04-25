"""WebGPU adapter / device handle for the splat renderer.

Sub-classes :class:`GPUDevice` from Phase 26. On construction we try
to acquire a real WebGPU adapter; if none is available (no GPU /
display / drivers) the device flips to ``backend = "cpu"`` and
:meth:`is_available` returns ``True`` so the pipeline still routes
through the CPU fallback rather than crashing.

Tests run on machines without a GPU — the graceful path is the
default. Real GPU initialization happens lazily only when
:meth:`initialize` is called.
"""

from __future__ import annotations

from ai_viewer.neural_field.gpu.device import GPUDevice


class WebGPUDevice(GPUDevice):
    """A :class:`GPUDevice` backed by ``wgpu`` when a real adapter is available."""

    def __init__(self, backend: str = "webgpu") -> None:
        # Start out claiming "webgpu" so the pipeline can dispatch; we'll
        # downgrade to "cpu" if real init fails.
        super().__init__(backend=backend if backend in ("webgpu", "cpu", "none") else "webgpu")
        self.adapter = None
        self.device = None
        self.queue = None
        self.last_error: str | None = None
        if self._backend == "webgpu":
            self.initialize()

    def initialize(self) -> None:
        """Attempt to acquire a WebGPU adapter + device. Falls back on failure."""
        try:
            import wgpu
        except Exception as exc:  # pragma: no cover - defensive
            self.last_error = f"wgpu unavailable: {exc}"
            self._backend = "cpu"
            return
        try:
            adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
        except Exception as exc:
            self.last_error = f"adapter request failed: {exc}"
            self._backend = "cpu"
            return
        if adapter is None:
            self.last_error = "no adapter available"
            self._backend = "cpu"
            return
        try:
            device = adapter.request_device_sync()
        except Exception as exc:
            self.last_error = f"device request failed: {exc}"
            self._backend = "cpu"
            return
        self.adapter = adapter
        self.device = device
        self.queue = device.queue

    def is_available(self) -> bool:
        return self._backend != "none"

    def get_backend(self) -> str:
        return self._backend
