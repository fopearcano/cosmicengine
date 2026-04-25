"""NumPy-only utilities for manipulating 3D density grids.

Stays scipy-free on purpose — everything (normalization, nearest-neighbor
upscaling, box-blur smoothing, axis projection, grayscale PPM writing)
is plain NumPy with `sliding_window_view` for the convolution.
"""

from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def normalize_density_grid(grid: np.ndarray) -> np.ndarray:
    """Return a copy of ``grid`` scaled so its maximum is 1.0.

    All-zero (or non-positive-peak) grids are returned unchanged so the
    function is safe on empty inputs.
    """
    arr = np.asarray(grid, dtype=np.float64)
    peak = arr.max() if arr.size else 0.0
    if peak <= 0.0:
        return arr.copy()
    return arr / peak


def upscale_density_grid_nearest(
    grid: np.ndarray,
    scale: int,
) -> np.ndarray:
    """Nearest-neighbor 3D upsample: ``(G, G, G)`` → ``(G*scale,)*3``."""
    if scale < 1:
        raise ValueError(f"scale must be >= 1; got {scale}")
    if grid.ndim != 3:
        raise ValueError(f"expected 3D grid; got shape {grid.shape}")
    if scale == 1:
        return np.asarray(grid).copy()
    return (
        np.asarray(grid)
        .repeat(scale, axis=0)
        .repeat(scale, axis=1)
        .repeat(scale, axis=2)
    )


def smooth_density_grid(
    grid: np.ndarray,
    kernel_size: int = 3,
) -> np.ndarray:
    """3D box blur of side ``kernel_size`` (odd integer >= 1)."""
    if kernel_size < 1:
        raise ValueError(f"kernel_size must be >= 1; got {kernel_size}")
    if kernel_size % 2 == 0:
        raise ValueError(f"kernel_size must be odd; got {kernel_size}")
    if grid.ndim != 3:
        raise ValueError(f"expected 3D grid; got shape {grid.shape}")
    if kernel_size == 1:
        return np.asarray(grid, dtype=np.float64).copy()

    arr = np.asarray(grid, dtype=np.float64)
    pad = kernel_size // 2
    padded = np.pad(arr, pad, mode="edge")
    windows = sliding_window_view(
        padded, (kernel_size, kernel_size, kernel_size)
    )
    return windows.mean(axis=(-3, -2, -1))


def project_density_to_2d(grid: np.ndarray) -> np.ndarray:
    """Sum a 3D grid along its last axis and rescale to ``[0, 255]``.

    Returns a 2D float array suitable for grayscale PPM export.
    """
    if grid.ndim != 3:
        raise ValueError(f"expected 3D grid; got shape {grid.shape}")
    projection = np.asarray(grid, dtype=np.float64).sum(axis=2)
    peak = projection.max() if projection.size else 0.0
    if peak <= 0.0:
        return np.zeros_like(projection)
    return np.clip(projection / peak * 255.0, 0.0, 255.0)


def render_density_image_to_ppm(
    image_2d: np.ndarray,
    output_path: str,
) -> None:
    """Write a 2D float image (values in roughly ``[0, 255]``) as grayscale PPM."""
    if image_2d.ndim != 2:
        raise ValueError(f"expected 2D image; got shape {image_2d.shape}")
    arr = np.clip(np.asarray(image_2d, dtype=np.float64), 0.0, 255.0)
    pixels = arr.astype(np.int64)
    h, w = pixels.shape
    with open(output_path, "w", encoding="ascii") as fh:
        fh.write("P3\n")
        fh.write(f"{w} {h}\n")
        fh.write("255\n")
        for row in pixels:
            fh.write(" ".join(f"{v} {v} {v}" for v in row))
            fh.write("\n")
