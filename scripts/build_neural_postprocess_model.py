"""One-shot generator for data/neural_postprocess_identity.onnx.

A 4-D image-shaped Identity ONNX graph used by the AI Viewer demos and
tests to exercise the neural-postprocess path without shipping a real
trained model. Input/output shape: ``(1, 3, 32, 32)`` float32.

Run once with:
    python scripts/build_neural_postprocess_model.py
"""

from __future__ import annotations

from pathlib import Path

import onnx
from onnx import TensorProto, helper


_REPO_ROOT = Path(__file__).resolve().parent.parent
_OUT_PATH = _REPO_ROOT / "data" / "neural_postprocess_identity.onnx"

_INPUT_SHAPE = [1, 3, 32, 32]


def build() -> onnx.ModelProto:
    input_info = helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, _INPUT_SHAPE
    )
    output_info = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, _INPUT_SHAPE
    )
    node = helper.make_node(
        "Identity", inputs=["input"], outputs=["output"], name="identity"
    )
    graph = helper.make_graph(
        nodes=[node],
        name="neural_postprocess_identity",
        inputs=[input_info],
        outputs=[output_info],
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
