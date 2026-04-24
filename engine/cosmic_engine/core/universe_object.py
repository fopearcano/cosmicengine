"""The :class:`UniverseObject` — the universal record type of the engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from cosmic_engine.core.object_types import CosmicObjectType
from cosmic_engine.core.truth import TruthLevel
from cosmic_engine.core.vector import Vector3


@dataclass
class UniverseObject:
    """A single entity in the universe state.

    All physical quantities are stored in SI units. Optional scalar
    fields are ``None`` when unknown rather than zero, so callers can
    distinguish "missing" from "measured as zero".
    """

    id: str
    name: str
    object_type: CosmicObjectType
    position_m: Vector3
    velocity_m_s: Vector3
    truth_level: TruthLevel
    source: str | None = None
    mass_kg: float | None = None
    radius_m: float | None = None
    luminosity_w: float | None = None
    spectral_class: str | None = None
    redshift_z: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        """Raise :class:`ValueError` if any field violates the invariants."""
        if not self.id:
            raise ValueError("UniverseObject.id must not be empty")
        if not self.name:
            raise ValueError("UniverseObject.name must not be empty")
        if self.mass_kg is not None and self.mass_kg < 0:
            raise ValueError("mass_kg must not be negative")
        if self.radius_m is not None and self.radius_m < 0:
            raise ValueError("radius_m must not be negative")
        if self.luminosity_w is not None and self.luminosity_w < 0:
            raise ValueError("luminosity_w must not be negative")
        if self.redshift_z is not None and self.redshift_z < 0:
            raise ValueError("redshift_z must not be negative")

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict with enums/vectors reduced to primitives."""
        return {
            "id": self.id,
            "name": self.name,
            "object_type": self.object_type.value,
            "position_m": self.position_m.to_list(),
            "velocity_m_s": self.velocity_m_s.to_list(),
            "truth_level": self.truth_level.value,
            "source": self.source,
            "mass_kg": self.mass_kg,
            "radius_m": self.radius_m,
            "luminosity_w": self.luminosity_w,
            "spectral_class": self.spectral_class,
            "redshift_z": self.redshift_z,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> UniverseObject:
        """Rebuild a :class:`UniverseObject` from its :meth:`to_dict` form."""
        return cls(
            id=data["id"],
            name=data["name"],
            object_type=CosmicObjectType(data["object_type"]),
            position_m=Vector3.from_list(data["position_m"]),
            velocity_m_s=Vector3.from_list(data["velocity_m_s"]),
            truth_level=TruthLevel(data["truth_level"]),
            source=data.get("source"),
            mass_kg=data.get("mass_kg"),
            radius_m=data.get("radius_m"),
            luminosity_w=data.get("luminosity_w"),
            spectral_class=data.get("spectral_class"),
            redshift_z=data.get("redshift_z"),
            metadata=dict(data.get("metadata", {})),
        )
