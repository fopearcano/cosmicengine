"""One-shot generator for data/density_upscaler.onnx.

Builds a tiny ONNX graph that trilinearly upsamples a 64³ density grid
to 128³ via a single ``Resize`` op. The result is committed so demos
and tests run without the build-time ``onnx`` library.

Run once with:
    python scripts/build_density_onnx_model.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


_REPO_ROOT = Path(__file__).resolve().parent.parent
_OUT_PATH = _REPO_ROOT / "data" / "density_upscaler.onnx"

_GRID = 64
_SCALE = 2


def build() -> onnx.ModelProto:
    input_info = helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, [1, 1, _GRID, _GRID, _GRID]
    )
    output_info = helper.make_tensor_value_info(
        "output",
        TensorProto.FLOAT,
        [1, 1, _GRID * _SCALE, _GRID * _SCALE, _GRID * _SCALE],
    )

    scales = numpy_helper.from_array(
        np.array([1.0, 1.0, _SCALE, _SCALE, _SCALE], dtype=np.float32),
        name="scales",
    )

    resize = helper.make_node(
        "Resize",
        inputs=["input", "", "scales"],
        outputs=["output"],
        mode="linear",
        coordinate_transformation_mode="half_pixel",
        name="resize_2x",
    )

    graph = helper.make_graph(
        nodes=[resize],
        name="density_upscaler",
        inputs=[input_info],
        outputs=[output_info],
        initializer=[scales],
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
