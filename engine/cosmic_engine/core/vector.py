"""Minimal 3D vector type.

Intentionally dependency-free. A heavier numerical representation
(NumPy, JAX, etc.) will arrive in later phases; core state stays plain
Python so serialization and comparison remain trivial.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Vector3:
    """An immutable 3D vector of floats."""

    x: float
    y: float
    z: float

    def to_list(self) -> list[float]:
        """Return the vector components as a plain list ``[x, y, z]``."""
        return [self.x, self.y, self.z]

    @classmethod
    def from_list(cls, values: list[float]) -> Vector3:
        """Build a :class:`Vector3` from a 3-element sequence."""
        if len(values) != 3:
            raise ValueError(
                f"Vector3.from_list expects 3 values, got {len(values)}"
            )
        x, y, z = values
        return cls(float(x), float(y), float(z))

    @classmethod
    def zero(cls) -> Vector3:
        """Return the zero vector."""
        return cls(0.0, 0.0, 0.0)
