"""Declarative description of a universe to synthesize."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# Whitelist of physics / spacetime model labels the generator
# recognises. Keep these as plain strings (not enums) so external
# tooling can serialize a UniverseSpec to JSON without engine imports.
_PHYSICS_MODELS = frozenset({
    "newtonian", "lambda_cdm", "n_body_only", "none",
})
_SPACETIME_MODELS = frozenset({
    "analytical", "neural", "schwarzschild", "none",
})


@dataclass
class UniverseSpec:
    """All knobs needed to deterministically generate a universe.

    ``initial_conditions`` keys recognised by the generator:
      - ``object_count`` (int): how many objects to seed
      - ``mass_range_kg`` (tuple[float, float]): per-object mass band
      - ``velocity_dispersion_m_s`` (float): isotropic Gaussian σ
      - ``include_central_mass`` (bool): drop one massive body at origin
      - ``central_mass_kg`` (float): mass for the optional central body

    ``constraints`` keys recognised:
      - ``max_mass_kg`` (float)
      - ``max_velocity_m_s`` (float)

    Every field has a default so ``UniverseSpec(id=..., seed=...)``
    is enough to round-trip through the generator.
    """

    id: str
    seed: int
    description: str = ""
    scale_limits: tuple[float, float] = (1.0e9, 1.0e22)
    initial_conditions: dict[str, Any] = field(default_factory=dict)
    physics_model: str = "newtonian"
    spacetime_model: str = "analytical"
    rule_ids: list[str] = field(default_factory=list)
    constraints: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        """Raise :class:`ValueError` if any field is invalid."""
        if not self.id:
            raise ValueError("UniverseSpec.id must be a non-empty string")
        if self.seed is None:
            raise ValueError("UniverseSpec.seed must be set")
        try:
            seed_int = int(self.seed)
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"UniverseSpec.seed must be int-coercible; got {self.seed!r}"
            ) from e
        if seed_int < 0:
            raise ValueError("UniverseSpec.seed must be non-negative")
        if (
            len(self.scale_limits) != 2
            or self.scale_limits[0] <= 0.0
            or self.scale_limits[1] <= self.scale_limits[0]
        ):
            raise ValueError(
                "scale_limits must be (lo, hi) with 0 < lo < hi; got "
                f"{self.scale_limits}"
            )
        if self.physics_model not in _PHYSICS_MODELS:
            raise ValueError(
                f"unknown physics_model {self.physics_model!r}; "
                f"expected one of {sorted(_PHYSICS_MODELS)}"
            )
        if self.spacetime_model not in _SPACETIME_MODELS:
            raise ValueError(
                f"unknown spacetime_model {self.spacetime_model!r}; "
                f"expected one of {sorted(_SPACETIME_MODELS)}"
            )
        # Initial-conditions sanity
        ic = self.initial_conditions
        if "object_count" in ic and int(ic["object_count"]) < 0:
            raise ValueError("initial_conditions.object_count must be >= 0")
        if "mass_range_kg" in ic:
            mr = ic["mass_range_kg"]
            if (
                len(mr) != 2
                or mr[0] <= 0.0
                or mr[1] < mr[0]
            ):
                raise ValueError(
                    "mass_range_kg must be (lo, hi) with 0 < lo <= hi"
                )
        if (
            "velocity_dispersion_m_s" in ic
            and ic["velocity_dispersion_m_s"] < 0.0
        ):
            raise ValueError("velocity_dispersion_m_s must be non-negative")
