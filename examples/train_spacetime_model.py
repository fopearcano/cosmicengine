"""Demo: train a small Neural Spacetime Field and export it to ONNX."""

from __future__ import annotations

import time
from pathlib import Path

from cosmic_engine.ai.training import SpacetimeDataset
from cosmic_engine.ai.training.export_onnx import export_to_onnx
from cosmic_engine.ai.training.train_spacetime import train_model


_REPO_ROOT = Path(__file__).resolve().parent.parent
_OUTPUT_PATH = _REPO_ROOT / "models" / "spacetime_field.onnx"


def main() -> None:
    # A relatively tight radius range keeps the dynamic range of the
    # 1/r^2 acceleration manageable for a tiny CPU MLP. Demos that need
    # broader coverage should widen this and train for more epochs.
    dataset = SpacetimeDataset(
        num_samples=100_000,
        mass_kg=1.0e30,
        radius_range=(2.0e10, 5.0e10),
    )
    print(
        f"dataset           : {dataset.num_samples} samples, "
        f"mass={dataset.mass_kg:.2e} kg, "
        f"radius={dataset.radius_range[0]:.0e}..{dataset.radius_range[1]:.0e} m"
    )

    t0 = time.perf_counter()
    model, losses = train_model(
        dataset,
        epochs=20,
        batch_size=512,
        learning_rate=1.0e-3,
    )
    train_seconds = time.perf_counter() - t0

    print()
    print(f"training time     : {train_seconds:.2f} s")
    print(f"first epoch loss  : {losses[0]:.6e}")
    print(f"final epoch loss  : {losses[-1]:.6e}")
    print(
        f"loss reduction    : "
        f"{losses[0] / max(losses[-1], 1.0e-30):.1f}x"
    )

    out = export_to_onnx(model, str(_OUTPUT_PATH))
    print()
    print(f"exported ONNX to  : {out}")


if __name__ == "__main__":
    main()
