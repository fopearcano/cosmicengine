"""Demo: load a trained ONNX spacetime model and compare to analytical truth."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from cosmic_engine.ai import ONNXSpacetimeField
from cosmic_engine.ai.training.dataset import (
    SpacetimeDataset,
    schwarzschild_acceleration_batch,
)


_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_MODEL = _REPO_ROOT / "models" / "spacetime_field.onnx"
_MASS_KG = 1.0e30


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default=str(_DEFAULT_MODEL),
        help="Path to a trained ONNX spacetime field.",
    )
    parser.add_argument("--samples", type=int, default=2_000)
    args = parser.parse_args()

    if not Path(args.model).is_file():
        print(f"model not found: {args.model}")
        print(
            "run examples/train_spacetime_model.py first or pass --model "
            "to point at a trained file"
        )
        return

    # Match the training script's radius range so the test is in-distribution.
    dataset = SpacetimeDataset(
        num_samples=args.samples,
        mass_kg=_MASS_KG,
        radius_range=(2.0e10, 5.0e10),
        seed=99,  # different from training seed
    )
    x_test, y_true = dataset.generate()
    print(f"loaded test set   : {len(x_test)} samples")

    field = ONNXSpacetimeField(args.model)
    print(f"loaded model      : {args.model}")
    print(f"model confidence  : {field.confidence():.2f}")

    # Per-sample inference (matches the integrator's actual call pattern).
    t0 = time.perf_counter()
    y_pred = np.zeros_like(y_true)
    for i in range(len(x_test)):
        y_pred[i] = field.query_acceleration(
            x_test[i, :3], direction=x_test[i, 3:6]
        )
    infer_seconds = time.perf_counter() - t0

    # Errors in normalized magnitudes so the number is human-readable
    # across the wide dynamic range of accelerations.
    truth_mag = np.linalg.norm(y_true, axis=1)
    pred_mag = np.linalg.norm(y_pred, axis=1)
    safe_truth = np.where(truth_mag > 0, truth_mag, 1.0)
    rel_err = np.abs(pred_mag - truth_mag) / safe_truth
    cos_sim = (y_true * y_pred).sum(axis=1) / (
        truth_mag * np.where(pred_mag > 0, pred_mag, 1.0) + 1e-30
    )

    print()
    print(f"inference time    : {infer_seconds * 1000:.1f} ms")
    print(f"avg |err/truth|   : {float(np.mean(rel_err)):.4f}")
    print(f"median |err/truth|: {float(np.median(rel_err)):.4f}")
    print(f"avg cos similarity: {float(np.mean(cos_sim)):.4f}  (1 = perfect)")
    print(f"finite outputs    : {bool(np.isfinite(y_pred).all())}")


if __name__ == "__main__":
    main()
