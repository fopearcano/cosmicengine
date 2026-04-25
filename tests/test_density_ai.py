"""Tests for Phase 9 AI density-field reconstruction."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from cosmic_engine.ai import (
    DensityFieldModel,
    ONNXDensityModel,
    SimpleDensityEnhancer,
    enhance_density_field,
)
from cosmic_engine.ai.density_utils import (
    normalize_density_grid,
    project_density_to_2d,
    render_density_image_to_ppm,
    smooth_density_grid,
    upscale_density_grid_nearest,
)


_REPO_ROOT = Path(__file__).resolve().parent.parent
_MODEL = _REPO_ROOT / "data" / "density_upscaler.onnx"


def _toy_grid(size: int = 8) -> np.ndarray:
    g = np.zeros((size, size, size), dtype=np.float64)
    g[1, 2, 3] = 1.0
    g[5, 5, 5] = 0.5
    g[size - 1, size - 1, size - 1] = 0.25
    return g


# --- density_utils ---


def test_normalize_density_grid_scales_to_unit_max():
    g = np.array([[[2.0, 4.0], [0.0, 1.0]]])
    out = normalize_density_grid(g)
    assert out.max() == pytest.approx(1.0)
    assert out.min() == pytest.approx(0.0)


def test_normalize_density_grid_handles_all_zero():
    g = np.zeros((3, 3, 3))
    out = normalize_density_grid(g)
    assert out.shape == g.shape
    assert (out == 0).all()


def test_upscale_density_grid_nearest_shape():
    g = np.arange(8).reshape(2, 2, 2).astype(np.float64)
    out = upscale_density_grid_nearest(g, scale=3)
    assert out.shape == (6, 6, 6)
    # corner cell of original maps to a 3x3x3 block of the output
    assert (out[:3, :3, :3] == g[0, 0, 0]).all()


def test_upscale_density_grid_scale_one_is_identity():
    g = _toy_grid()
    np.testing.assert_array_equal(upscale_density_grid_nearest(g, 1), g)


@pytest.mark.parametrize("bad_scale", [0, -1])
def test_upscale_density_rejects_bad_scale(bad_scale):
    with pytest.raises(ValueError):
        upscale_density_grid_nearest(_toy_grid(), bad_scale)


def test_smooth_density_grid_shape_preserved():
    g = _toy_grid(8)
    out = smooth_density_grid(g, kernel_size=3)
    assert out.shape == g.shape


def test_smooth_density_grid_kernel_one_is_identity():
    g = _toy_grid(8)
    np.testing.assert_array_equal(smooth_density_grid(g, kernel_size=1), g)


def test_smooth_density_grid_constant_input_is_constant():
    g = np.full((6, 6, 6), 0.7)
    np.testing.assert_allclose(smooth_density_grid(g, 3), g)


@pytest.mark.parametrize("bad_kernel", [0, -1, 2, 4])
def test_smooth_density_rejects_bad_kernel(bad_kernel):
    with pytest.raises(ValueError):
        smooth_density_grid(_toy_grid(), kernel_size=bad_kernel)


def test_project_density_to_2d_bounds_and_shape():
    g = _toy_grid(8)
    img = project_density_to_2d(g)
    assert img.shape == (8, 8)
    assert img.min() >= 0.0
    assert img.max() <= 255.0
    # at least one nonzero pixel since the grid has nonzero entries
    assert img.max() > 0.0


def test_project_density_zero_grid_is_zero_image():
    img = project_density_to_2d(np.zeros((4, 4, 4)))
    assert img.shape == (4, 4)
    assert (img == 0).all()


def test_render_density_image_to_ppm(tmp_path: Path):
    img = np.full((4, 4), 100.0)
    out = tmp_path / "img.ppm"
    render_density_image_to_ppm(img, str(out))
    text = out.read_text().splitlines()
    assert text[0] == "P3"
    assert text[1] == "4 4"
    assert text[2] == "255"


# --- DensityFieldModel base ---


def test_base_model_raises_not_implemented():
    model = DensityFieldModel()
    with pytest.raises(NotImplementedError):
        model.enhance(_toy_grid())
    with pytest.raises(NotImplementedError):
        model.confidence()


# --- SimpleDensityEnhancer ---


def test_simple_enhancer_doubles_resolution():
    g = _toy_grid(8)
    out = SimpleDensityEnhancer().enhance(g)
    assert out.shape == (16, 16, 16)
    assert out.min() >= 0.0
    assert out.max() <= 1.0


def test_simple_enhancer_normalizes():
    g = _toy_grid(8) * 100.0  # large input values
    out = SimpleDensityEnhancer().enhance(g)
    assert out.max() == pytest.approx(1.0)


def test_simple_enhancer_confidence():
    assert SimpleDensityEnhancer().confidence() == 0.4


# --- enhance_density_field ---


def test_enhance_density_field_none_uses_simple():
    g = _toy_grid(8)
    out = enhance_density_field(g, None)
    np.testing.assert_array_equal(out, SimpleDensityEnhancer().enhance(g))


class _BrokenModel(DensityFieldModel):
    def enhance(self, grid):
        raise RuntimeError("kaboom")

    def confidence(self):
        return 0.0


def test_enhance_density_field_falls_back_when_model_raises():
    g = _toy_grid(8)
    out = enhance_density_field(g, _BrokenModel())
    np.testing.assert_array_equal(out, SimpleDensityEnhancer().enhance(g))


# --- ONNX density model ---


def test_onnx_density_model_loads_and_enhances_correct_size():
    model = ONNXDensityModel(str(_MODEL))
    g = np.random.default_rng(0).random((64, 64, 64))
    out = model.enhance(g)
    assert out.shape == (128, 128, 128)
    assert out.min() >= 0.0
    assert out.max() <= 1.0
    assert model.confidence() == pytest.approx(0.8)


def test_onnx_density_model_falls_back_on_wrong_size():
    model = ONNXDensityModel(str(_MODEL))
    g = _toy_grid(8)  # mock model expects 64^3
    out = model.enhance(g)
    # SimpleDensityEnhancer fallback: 2x upscale -> 16^3
    assert out.shape == (16, 16, 16)
    assert model.confidence() == pytest.approx(0.3)


def test_onnx_density_model_missing_file_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        ONNXDensityModel(str(tmp_path / "nope.onnx"))


def test_onnx_density_recovers_after_failure():
    model = ONNXDensityModel(str(_MODEL))
    # first call: wrong size -> fallback, confidence drops
    model.enhance(_toy_grid(8))
    assert model.confidence() == pytest.approx(0.3)
    # second call: correct size -> success, confidence restored
    model.enhance(np.random.default_rng(1).random((64, 64, 64)))
    assert model.confidence() == pytest.approx(0.8)
