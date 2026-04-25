"""Training pipeline for the Neural Spacetime Field.

Phase 31 contribution: dataset generation, a small MLP, an Adam
training loop, and an ONNX exporter that produces a file the
runtime :class:`cosmic_engine.ai.ONNXSpacetimeField` can consume.

PyTorch is **only** imported by submodules in this package — no
runtime / inference path takes a torch dependency.
"""

from cosmic_engine.ai.training.dataset import SpacetimeDataset

__all__ = ["SpacetimeDataset"]
