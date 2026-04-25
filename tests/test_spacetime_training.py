"""Tests for Phase 31 Neural Spacetime Field training pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from cosmic_engine.ai import ONNXSpacetimeField  # noqa: E402
from cosmic_engine.ai.training import SpacetimeDataset  # noqa: E402
from cosmic_engine.ai.training.dataset import (  # noqa: E402
    schwarzschild_acceleration_batch,
)
from cosmic_engine.ai.training.export_onnx import export_to_onnx  # noqa: E402
from cosmic_engine.ai.training.model import SpacetimeMLP  # noqa: E402
from cosmic_engine.ai.training.train_spacetime import train_model  # noqa: E402


# --- SpacetimeDataset ---


def test_dataset_generate_shapes_and_dtype():
    ds = SpacetimeDataset(
        num_samples=128, mass_kg=1.0e30, radius_range=(1.0e10, 1.0e12)
    )
    x, y = ds.generate()
    assert x.shape == (128, 6)
    assert y.shape == (128, 3)
    assert x.dtype == np.float32
    assert y.dtype == np.float32


def test_dataset_directions_unit_length():
    ds = SpacetimeDataset(
        num_samples=64, mass_kg=1.0e30, radius_range=(1.0e10, 1.0e12)
    )
    x, _ = ds.generate()
    norms = np.linalg.norm(x[:, 3:6], axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-6)


def test_dataset_positions_within_radius_range():
    r_min, r_max = 1.0e10, 1.0e14
    ds = SpacetimeDataset(num_samples=256, mass_kg=1.0e30, radius_range=(r_min, r_max))
    x, _ = ds.generate()
    radii = np.linalg.norm(x[:, :3], axis=1)
    assert (radii >= r_min * (1.0 - 1e-9)).all()
    assert (radii <= r_max * (1.0 + 1e-9)).all()


def test_dataset_targets_match_analytical_formula():
    ds = SpacetimeDataset(
        num_samples=32, mass_kg=1.0e30, radius_range=(1.0e10, 1.0e12)
    )
    x, y = ds.generate()
    truth = schwarzschild_acceleration_batch(
        x[:, :3].astype(np.float64), ds.mass_kg
    )
    np.testing.assert_allclose(y.astype(np.float64), truth, rtol=1e-6, atol=1e-30)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"num_samples": 0},
        {"mass_kg": 0.0},
        {"radius_range": (1.0e10, 0.0)},
        {"radius_range": (1.0e12, 1.0e10)},
    ],
)
def test_dataset_validates_constructor_args(kwargs):
    base = dict(num_samples=32, mass_kg=1.0e30, radius_range=(1.0e10, 1.0e12))
    base.update(kwargs)
    with pytest.raises(ValueError):
        SpacetimeDataset(**base)


def test_dataset_seed_is_deterministic():
    ds_a = SpacetimeDataset(num_samples=100, mass_kg=1.0e30, radius_range=(1.0e10, 1.0e12), seed=7)
    ds_b = SpacetimeDataset(num_samples=100, mass_kg=1.0e30, radius_range=(1.0e10, 1.0e12), seed=7)
    x_a, y_a = ds_a.generate()
    x_b, y_b = ds_b.generate()
    np.testing.assert_array_equal(x_a, x_b)
    np.testing.assert_array_equal(y_a, y_b)


# --- SpacetimeMLP ---


def test_model_forward_shape():
    model = SpacetimeMLP()
    x = torch.zeros(8, 6)
    y = model(x)
    assert y.shape == (8, 3)


def test_model_forward_finite():
    model = SpacetimeMLP()
    x = torch.rand(16, 6) * 1.0e10
    y = model(x)
    assert torch.isfinite(y).all().item()


# --- training ---


def test_training_reduces_loss():
    """A short training run should drive the loss down meaningfully."""
    # Tight radius range keeps the dynamic range of acceleration
    # manageable for a 5-epoch run with a tiny MLP.
    ds = SpacetimeDataset(
        num_samples=2_000, mass_kg=1.0e30, radius_range=(2.0e10, 5.0e10)
    )
    _, losses = train_model(
        ds, epochs=5, batch_size=128, learning_rate=5.0e-3, verbose=False
    )
    assert len(losses) == 5
    assert losses[-1] < losses[0]  # learned something


def test_training_returns_model_with_correct_shape():
    ds = SpacetimeDataset(
        num_samples=512, mass_kg=1.0e30, radius_range=(1.0e10, 1.0e12)
    )
    model, _ = train_model(
        ds, epochs=2, batch_size=128, verbose=False
    )
    out = model(torch.zeros(4, 6))
    assert out.shape == (4, 3)


# --- ONNX export ---


def test_export_creates_onnx_file(tmp_path: Path):
    model = SpacetimeMLP()
    out = export_to_onnx(model, str(tmp_path / "spacetime.onnx"))
    assert Path(out).is_file()
    assert Path(out).stat().st_size > 0


def test_exported_onnx_loads_into_runtime_field(tmp_path: Path):
    model = SpacetimeMLP()
    onnx_path = export_to_onnx(model, str(tmp_path / "spacetime.onnx"))
    field = ONNXSpacetimeField(onnx_path)
    out = field.query_acceleration(
        np.array([1.0e10, 0.0, 0.0]), direction=np.array([0.0, 1.0, 0.0])
    )
    assert out.shape == (3,)
    assert np.isfinite(out).all()
    assert field.confidence() == pytest.approx(0.9)


def test_exported_trained_model_roughly_matches_truth(tmp_path: Path):
    """End-to-end smoke test: train a tiny model and check ONNX output magnitude."""
    ds = SpacetimeDataset(
        num_samples=4_000, mass_kg=1.0e30, radius_range=(1.0e10, 1.0e12)
    )
    model, _ = train_model(
        ds, epochs=8, batch_size=256, learning_rate=2.0e-3, verbose=False
    )
    onnx_path = export_to_onnx(model, str(tmp_path / "trained.onnx"))
    field = ONNXSpacetimeField(onnx_path)

    pos = np.array([5.0e10, 0.0, 0.0])
    pred = field.query_acceleration(pos)
    truth = schwarzschild_acceleration_batch(pos.reshape(1, 3), ds.mass_kg)[0]
    pred_mag = float(np.linalg.norm(pred))
    truth_mag = float(np.linalg.norm(truth))
    # Trained on this radius range, the prediction should at least be the
    # right order of magnitude.
    assert pred_mag > 0.0
    assert truth_mag > 0.0
    assert pred_mag / truth_mag < 100.0
    assert pred_mag / truth_mag > 0.01
