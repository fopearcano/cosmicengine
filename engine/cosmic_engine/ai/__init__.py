"""Pluggable AI-assisted perception layer.

Phase 5 contribution: defines the :class:`AIWarpModel` interface that
future PyTorch / ONNX / JAX models will implement, plus a deterministic
placeholder (:class:`SimpleNeuralWarp`) that stands in for a real model
so the rest of the pipeline can be exercised today.

Important: the deterministic perception transform remains the default.
AI is opt-in and fallback-safe — passing ``ai_model=None`` to
:func:`cosmic_engine.perception.transform_photon_field` yields the
exact same result as before this module existed.
"""

from cosmic_engine.ai.base import AIWarpModel
from cosmic_engine.ai.neural_warp import SimpleNeuralWarp
from cosmic_engine.ai.onnx_model import ONNXModelWrapper
from cosmic_engine.ai.onnx_warp import ONNXWarpModel

__all__ = [
    "AIWarpModel",
    "ONNXModelWrapper",
    "ONNXWarpModel",
    "SimpleNeuralWarp",
]
