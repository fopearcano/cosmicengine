"""Density-field enhancement model interface and deterministic fallback."""

from __future__ import annotations

import numpy as np

from cosmic_engine.ai.density_utils import (
    normalize_density_grid,
    smooth_density_grid,
    upscale_density_grid_nearest,
)


class DensityFieldModel:
    """Abstract interface for models that enhance a 3D density grid.

    Subclasses replace the default behavior; the base raises
    :class:`NotImplementedError` so a forgotten override fails loudly.
    """

    def enhance(self, grid: np.ndarray) -> np.ndarray:
        """Return an enhanced (typically higher-resolution) density grid."""
        raise NotImplementedError

    def confidence(self) -> float:
        """Self-rated trust score in ``[0, 1]``."""
        raise NotImplementedError


class SimpleDensityEnhancer(DensityFieldModel):
    """Pure-NumPy fallback: 2× nearest upsample + 3³ box blur + normalize."""

    def enhance(self, grid: np.ndarray) -> np.ndarray:
        upscaled = upscale_density_grid_nearest(grid, scale=2)
        smoothed = smooth_density_grid(upscaled, kernel_size=3)
        return normalize_density_grid(smoothed)

    def confidence(self) -> float:
        return 0.4


def enhance_density_field(
    grid: np.ndarray,
    model: DensityFieldModel | None,
) -> np.ndarray:
    """Run AI enhancement on ``grid`` with deterministic fallback.

    ``model is None`` → :class:`SimpleDensityEnhancer`. If a provided
    model raises during inference, the simple enhancer is used instead
    so the pipeline never crashes on a bad model.
    """
    if model is None:
        return SimpleDensityEnhancer().enhance(grid)
    try:
        return model.enhance(grid)
    except Exception:
        return SimpleDensityEnhancer().enhance(grid)
