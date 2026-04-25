"""Structured Gaussian-point buffer ready for GPU upload.

Mirrors what a Vulkan / WebGPU / CUDA implementation will eventually
push to device memory: four parallel contiguous arrays keyed by
index. Phase 26 keeps everything host-side as NumPy arrays.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ai_viewer.neural_field.gaussian import GaussianPoint


@dataclass
class GPUGaussianBuffer:
    """Plain-data view of a Gaussian field for the splat pipeline."""

    positions: np.ndarray   # shape (N, 3)
    colors: np.ndarray      # shape (N, 3)
    intensities: np.ndarray # shape (N,)
    sigmas: np.ndarray      # shape (N,)

    def __len__(self) -> int:
        return int(self.positions.shape[0])

    @classmethod
    def from_gaussian_points(
        cls,
        points: list[GaussianPoint],
    ) -> "GPUGaussianBuffer":
        """Pack a list of :class:`GaussianPoint` into parallel NumPy arrays."""
        n = len(points)
        if n == 0:
            return cls(
                positions=np.zeros((0, 3), dtype=np.float64),
                colors=np.zeros((0, 3), dtype=np.float64),
                intensities=np.zeros((0,), dtype=np.float64),
                sigmas=np.zeros((0,), dtype=np.float64),
            )
        positions = np.empty((n, 3), dtype=np.float64)
        colors = np.empty((n, 3), dtype=np.float64)
        intensities = np.empty((n,), dtype=np.float64)
        sigmas = np.empty((n,), dtype=np.float64)
        for i, point in enumerate(points):
            positions[i] = point.position
            colors[i] = point.color
            intensities[i] = point.intensity
            sigmas[i] = point.sigma
        return cls(
            positions=positions,
            colors=colors,
            intensities=intensities,
            sigmas=sigmas,
        )

    def validate(self) -> None:
        """Raise :class:`ValueError` if any array's shape is inconsistent."""
        n = len(self)
        if self.positions.shape != (n, 3):
            raise ValueError(
                f"positions shape must be ({n}, 3); got {self.positions.shape}"
            )
        if self.colors.shape != (n, 3):
            raise ValueError(
                f"colors shape must be ({n}, 3); got {self.colors.shape}"
            )
        if self.intensities.shape != (n,):
            raise ValueError(
                f"intensities shape must be ({n},); got {self.intensities.shape}"
            )
        if self.sigmas.shape != (n,):
            raise ValueError(
                f"sigmas shape must be ({n},); got {self.sigmas.shape}"
            )

    def to_numpy(self) -> dict[str, np.ndarray]:
        """Return a copy of every array as a plain dict for serialization."""
        return {
            "positions": self.positions.copy(),
            "colors": self.colors.copy(),
            "intensities": self.intensities.copy(),
            "sigmas": self.sigmas.copy(),
        }
