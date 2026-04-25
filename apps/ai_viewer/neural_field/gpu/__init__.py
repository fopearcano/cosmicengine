"""GPU-acceleration foundation for Gaussian splatting.

Phase 26 contribution: scaffolding for a future Vulkan / WebGPU /
CUDA backend. No real GPU APIs are required at runtime — the
``mock_gpu`` backend is a vectorized NumPy stand-in that exercises
the data flow (point → :class:`GPUGaussianBuffer` → pipeline →
image) without taking on a heavy dependency. The CPU fallback uses
the Phase 24 :class:`GaussianSplatRenderer` directly.
"""

from ai_viewer.neural_field.gpu.buffer import GPUGaussianBuffer
from ai_viewer.neural_field.gpu.device import GPUDevice
from ai_viewer.neural_field.gpu.fallback import CPUSplatFallback
from ai_viewer.neural_field.gpu.splat_pipeline import GaussianSplatPipeline

__all__ = [
    "CPUSplatFallback",
    "GPUDevice",
    "GPUGaussianBuffer",
    "GaussianSplatPipeline",
]
