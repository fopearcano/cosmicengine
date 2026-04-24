"""One-shot generator for data/mock_warp_model.onnx.

Builds a tiny linear ONNX graph standing in for a trained perception
model. The produced .onnx is committed to the repo so the demo and
tests do not need the ``onnx`` build-time library at runtime —
``onnxruntime`` alone is enough to consume it.

Input tensor  "input"  : float32 [1, 9]
    [dx, dy, dz, beta, warp_factor, brightness, r, g, b]

Output tensor "output" : float32 [1, 7]
    [dx', dy', dz', brightness', r', g', b']

Run once with:
    python scripts/build_mock_onnx_model.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


_REPO_ROOT = Path(__file__).resolve().parent.parent
_OUT_PATH = _REPO_ROOT / "data" / "mock_warp_model.onnx"


def _weights() -> np.ndarray:
    """9x7 weight matrix with a deliberate, predictable structure."""
    w = np.zeros((9, 7), dtype=np.float32)
    # direction pass-through with a small boost from beta / warp_factor
    w[0, 0] = 1.0  # dx  -> dx'
    w[3, 0] = 0.5  # beta tilts dx'
    w[1, 1] = 1.0  # dy  -> dy'
    w[2, 2] = 1.0  # dz  -> dz'
    w[4, 2] = 0.05  # warp tilts dz'
    # brightness: amplified slightly by input brightness and warp
    w[5, 3] = 1.2
    w[4, 3] = 0.05
    # color: soft channel rotation
    w[6, 4] = 0.9
    w[7, 4] = 0.1
    w[7, 5] = 0.9
    w[8, 5] = 0.1
    w[8, 6] = 0.9
    w[6, 6] = 0.1
    return w


def build() -> onnx.ModelProto:
    input_info = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 9])
    output_info = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 7])

    weights = numpy_helper.from_array(_weights(), name="W")
    bias = numpy_helper.from_array(np.zeros((7,), dtype=np.float32), name="b")

    matmul = helper.make_node("MatMul", ["input", "W"], ["mm"])
    add = helper.make_node("Add", ["mm", "b"], ["output"])

    graph = helper.make_graph(
        nodes=[matmul, add],
        name="mock_warp_model",
        inputs=[input_info],
        outputs=[output_info],
        initializer=[weights, bias],
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 17)],
        producer_name="cosmic_engine",
    )
    onnx.checker.check_model(model)
    return model


def main() -> None:
    model = build()
    _OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(_OUT_PATH))
    print(f"wrote {_OUT_PATH} ({_OUT_PATH.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
