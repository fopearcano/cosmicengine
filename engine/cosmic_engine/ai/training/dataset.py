"""Supervised training data for the Neural Spacetime Field.

Inputs are ``(pos.x, pos.y, pos.z, dir.x, dir.y, dir.z)`` in SI units.
Targets are the analytical Schwarzschild acceleration around a point
mass at the origin, computed by
:func:`cosmic_engine.ai.training.dataset.schwarzschild_acceleration_batch`
— the vectorized form of the same approximation the runtime uses,
so the trained model can be a drop-in replacement.
"""

from __future__ import annotations

import numpy as np

from cosmic_engine.core.units import SPEED_OF_LIGHT_M_S
from cosmic_engine.physics.nbody import GRAVITATIONAL_CONSTANT


def schwarzschild_acceleration_batch(
    positions: np.ndarray,
    mass_kg: float,
) -> np.ndarray:
    """Vectorized analytical acceleration: ``a = -2GM/r² · r̂`` per row.

    Mirrors :func:`ai_viewer.neural_field.gr.geodesic.schwarzschild_acceleration`
    but operates on a batch.  Returns a zero vector for any row whose
    position coincides with the origin.
    """
    positions = np.asarray(positions, dtype=np.float64)
    r = np.linalg.norm(positions, axis=1, keepdims=True)
    safe_r = np.where(r == 0.0, 1.0, r)
    r_hat = -positions / safe_r
    magnitude = (
        2.0 * GRAVITATIONAL_CONSTANT * float(mass_kg) / (safe_r * safe_r)
    )
    accel = r_hat * magnitude
    accel = np.where(r == 0.0, 0.0, accel)
    return accel


class SpacetimeDataset:
    """Generate ``(input, target)`` pairs by sampling and analytical truth."""

    def __init__(
        self,
        num_samples: int,
        mass_kg: float,
        radius_range: tuple[float, float],
        seed: int = 42,
    ) -> None:
        if num_samples <= 0:
            raise ValueError("num_samples must be positive")
        if mass_kg <= 0.0:
            raise ValueError("mass_kg must be positive")
        if (
            len(radius_range) != 2
            or radius_range[0] <= 0.0
            or radius_range[1] <= radius_range[0]
        ):
            raise ValueError(
                "radius_range must be (r_min, r_max) with 0 < r_min < r_max"
            )
        self.num_samples = int(num_samples)
        self.mass_kg = float(mass_kg)
        self.radius_range = (float(radius_range[0]), float(radius_range[1]))
        self.seed = int(seed)

    def generate(self) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(X, Y)`` of shapes ``(N, 6)`` and ``(N, 3)``, float32."""
        rng = np.random.default_rng(self.seed)
        n = self.num_samples
        r_min, r_max = self.radius_range
        # Uniform-in-log radius gives a more even distribution across
        # decades of magnitude than uniform-in-r, which is otherwise
        # heavily biased toward the upper end.
        log_r = rng.uniform(np.log(r_min), np.log(r_max), n)
        r = np.exp(log_r)
        cos_theta = rng.uniform(-1.0, 1.0, n)
        sin_theta = np.sqrt(np.maximum(0.0, 1.0 - cos_theta * cos_theta))
        phi = rng.uniform(0.0, 2.0 * np.pi, n)
        positions = np.empty((n, 3), dtype=np.float64)
        positions[:, 0] = r * sin_theta * np.cos(phi)
        positions[:, 1] = r * sin_theta * np.sin(phi)
        positions[:, 2] = r * cos_theta

        # Random unit directions via standard-normal projection.
        directions = rng.standard_normal((n, 3))
        norms = np.linalg.norm(directions, axis=1, keepdims=True)
        norms = np.where(norms == 0.0, 1.0, norms)
        directions /= norms

        accel = schwarzschild_acceleration_batch(positions, self.mass_kg)

        x = np.concatenate([positions, directions], axis=1).astype(np.float32)
        y = accel.astype(np.float32)
        return x, y
