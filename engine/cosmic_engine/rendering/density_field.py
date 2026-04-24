"""3D density grid construction from a :class:`GalaxyFieldBatch`.

The grid is the scaffolding for future AI super-resolution work: the
raw counts are a coarse estimator of large-scale structure that a
neural model can later refine. Today it is just a histogram.
"""

from __future__ import annotations

import numpy as np

from cosmic_engine.rendering.vectorized_galaxy_field import GalaxyFieldBatch


def galaxy_batch_to_density_grid(
    batch: GalaxyFieldBatch,
    grid_size: int,
    extent_m: float,
) -> np.ndarray:
    """Bin galaxy positions into a ``(grid_size, grid_size, grid_size)`` cube.

    Positions are expressed in the batch's camera-relative frame.
    ``extent_m`` is the **half-extent**: the cube spans
    ``[-extent_m, +extent_m]`` on each axis, and galaxies outside that
    cube are dropped. The grid is normalized so its maximum count is
    1.0 (an all-zero grid is returned unchanged).
    """
    if grid_size <= 0:
        raise ValueError(f"grid_size must be positive; got {grid_size}")
    if extent_m <= 0.0:
        raise ValueError(f"extent_m must be positive; got {extent_m}")

    grid = np.zeros((grid_size, grid_size, grid_size), dtype=np.float64)
    if len(batch) == 0:
        return grid

    positions = batch.positions_m
    inside = np.all(np.abs(positions) < extent_m, axis=1)
    kept = positions[inside]
    if kept.size == 0:
        return grid

    # Map each coordinate from [-extent, +extent] to [0, grid_size).
    normalized = (kept / extent_m + 1.0) * 0.5 * grid_size
    indices = np.clip(normalized.astype(np.int64), 0, grid_size - 1)
    np.add.at(
        grid,
        (indices[:, 0], indices[:, 1], indices[:, 2]),
        1.0,
    )

    peak = grid.max()
    if peak > 0.0:
        grid /= peak
    return grid
