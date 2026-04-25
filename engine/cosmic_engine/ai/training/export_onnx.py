"""ONNX exporter for trained :class:`SpacetimeMLP` instances.

Writes a graph with input shape ``(N, 6)`` and output shape ``(N, 3)``,
exactly matching the schema :class:`cosmic_engine.ai.ONNXSpacetimeField`
consumes.
"""

from __future__ import annotations

from pathlib import Path

import torch

from cosmic_engine.ai.training.model import SpacetimeMLP


def export_to_onnx(
    model: SpacetimeMLP,
    output_path: str,
    *,
    opset_version: int = 17,
) -> str:
    """Export ``model`` to ONNX. Returns the absolute path of the file."""
    model.eval()
    target = Path(output_path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    dummy_input = torch.zeros(1, 6, dtype=torch.float32)
    torch.onnx.export(
        model,
        dummy_input,
        str(target),
        input_names=["input"],
        output_names=["output"],
        opset_version=opset_version,
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
    )
    return str(target)
