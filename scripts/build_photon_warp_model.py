"""One-shot generator for data/photon_warp_model.onnx.

A 9-float -> 7-float linear ONNX graph following the Phase 21 photon
warp schema (see :mod:`cosmic_engine.ai.onnx_photon_warp` for the
input / output layout). Hand-tuned weights produce a visibly
different but plausible warp compared to the deterministic transform.

Run once with:
    python scripts/build_photon_warp_model.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


_REPO_ROOT = Path(__file__).resolve().parent.parent
_OUT_PATH = _REPO_ROOT / "data" / "photon_warp_model.onnx"


def _weights() -> np.ndarray:
    """9x7 weight matrix following the Phase 21 photon warp schema.

    Input columns: [dx, dy, dz, brightness, r, g, b, beta, warp_factor]
    Output rows:   [dx', dy', dz', brightness', r', g', b']
    """
    w = np.zeros((9, 7), dtype=np.float32)
    # direction: mostly preserve, slight pull from beta on dx, slight z-warp
    w[0, 0] = 0.95
    w[7, 0] = 0.5    # beta tilts dx'
    w[1, 1] = 1.0
    w[2, 2] = 1.0
    w[8, 2] = 0.05   # warp_factor tilts dz'
    # brightness amplification with mild coupling to beta and warp
    w[3, 3] = 1.3
    w[7, 3] = 0.1
    w[8, 3] = 0.02
    # color: gentle channel rotation
    w[4, 4] = 0.95
    w[5, 4] = 0.05
    w[5, 5] = 0.9
    w[6, 5] = 0.1
    w[6, 6] = 0.95
    w[4, 6] = 0.05
    return w


def build() -> onnx.ModelProto:
    # The leading dimension is symbolic ("batch") so this single graph
    # serves both the scalar (1, 9) and batch (N, 9) inference paths.
    input_info = helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, ["batch", 9]
    )
    output_info = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, ["batch", 7]
    )
    weights = numpy_helper.from_array(_weights(), name="W")
    bias = numpy_helper.from_array(np.zeros((7,), dtype=np.float32), name="b")
    matmul = helper.make_node("MatMul", ["input", "W"], ["mm"])
    add = helper.make_node("Add", ["mm", "b"], ["output"])
    graph = helper.make_graph(
        nodes=[matmul, add],
        name="photon_warp_model",
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
