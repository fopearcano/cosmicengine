"""Build :class:`GaussianPoint` lists from engine data structures."""

from __future__ import annotations

import numpy as np

from ai_viewer.neural_field.gaussian import GaussianPoint
from cosmic_engine.rendering.vectorized_galaxy_field import GalaxyFieldBatch


def build_gaussian_field_from_galaxy_batch(
    batch: GalaxyFieldBatch,
    *,
    sigma_scale: float = 1.0,
    min_sigma_m: float = 1.0e21,
) -> list[GaussianPoint]:
    """Map every galaxy in ``batch`` to a :class:`GaussianPoint`.

    ``sigma`` scales with the galaxy's distance from the camera (1%
    of distance, multiplied by ``sigma_scale``) so faraway galaxies
    cover more pixels relative to their visual budget. Color and
    intensity come straight from the batch (set up by Phase 8 with
    redshift-driven color).
    """
    n = len(batch)
    if n == 0:
        return []
    if sigma_scale <= 0.0:
        raise ValueError("sigma_scale must be positive")

    points: list[GaussianPoint] = []
    for i in range(n):
        distance = float(batch.distances_m[i])
        sigma = max(min_sigma_m, sigma_scale * distance * 0.01)
        points.append(
            GaussianPoint(
                position=batch.positions_m[i],
                color=batch.colors_rgb[i],
                intensity=float(batch.brightness[i]),
                sigma=sigma,
            )
        )
    return points


def build_gaussian_field_from_density(
    grid: np.ndarray,
    *,
    threshold: float = 0.01,
    cell_size_m: float = 1.0e22,
) -> list[GaussianPoint]:
    """Sample a 3D density grid into Gaussian points.

    Cells with density at or below ``threshold`` are skipped so an
    all-zero or very sparse grid produces an empty / tiny field.
    Density values are mapped to a grayscale color and used as the
    point's intensity.
    """
    if grid.ndim != 3:
        raise ValueError(
            f"build_gaussian_field_from_density expects a 3D grid; got {grid.shape}"
        )
    if cell_size_m <= 0.0:
        raise ValueError("cell_size_m must be positive")

    indices = np.argwhere(grid > threshold)
    if indices.size == 0:
        return []

    nx, ny, nz = grid.shape
    cx = (nx - 1) * 0.5
    cy = (ny - 1) * 0.5
    cz = (nz - 1) * 0.5

    points: list[GaussianPoint] = []
    for ix, iy, iz in indices:
        density = float(grid[ix, iy, iz])
        position = np.array(
            [
                (ix - cx) * cell_size_m,
                (iy - cy) * cell_size_m,
                (iz - cz) * cell_size_m,
            ],
            dtype=np.float64,
        )
        gray = max(0.0, min(255.0, density * 255.0))
        points.append(
            GaussianPoint(
                position=position,
                color=np.array([gray, gray, gray], dtype=np.float64),
                intensity=density,
                sigma=cell_size_m,
            )
        )
    return points
