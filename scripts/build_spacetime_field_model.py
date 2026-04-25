"""One-shot generator for data/spacetime_field_identity.onnx.

A 6 -> 3 linear ONNX graph following the Phase 30 spacetime field
schema. The hand-tuned weights produce an acceleration that points
toward the origin with magnitude proportional to ``|position|``,
mimicking a soft gravitational pull. This is enough to exercise the
neural geodesic integration path without shipping a real trained
model.

Run once with:
    python scripts/build_spacetime_field_model.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


_REPO_ROOT = Path(__file__).resolve().parent.parent
_OUT_PATH = _REPO_ROOT / "data" / "spacetime_field_identity.onnx"


def _weights() -> np.ndarray:
    """6x3 matrix: a = -k * position (toward origin); ignore direction."""
    w = np.zeros((6, 3), dtype=np.float32)
    # Mild "spring" pulling toward origin; magnitude scaled so a
    # 1e23 m position yields a 1e23 * 1e-13 = 1e10 m/s^2 acceleration.
    k = -1.0e-13
    w[0, 0] = k
    w[1, 1] = k
    w[2, 2] = k
    return w


def build() -> onnx.ModelProto:
    input_info = helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, [1, 6]
    )
    output_info = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, [1, 3]
    )
    weights = numpy_helper.from_array(_weights(), name="W")
    bias = numpy_helper.from_array(np.zeros((3,), dtype=np.float32), name="b")
    matmul = helper.make_node("MatMul", ["input", "W"], ["mm"])
    add = helper.make_node("Add", ["mm", "b"], ["output"])
    graph = helper.make_graph(
        nodes=[matmul, add],
        name="spacetime_field_identity",
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
