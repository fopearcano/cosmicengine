"""Galaxy-specific properties and :class:`UniverseObject` factory."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from cosmic_engine.core.object_types import CosmicObjectType
from cosmic_engine.core.truth import TruthLevel
from cosmic_engine.core.universe_object import UniverseObject
from cosmic_engine.core.vector import Vector3


@dataclass
class GalaxyProperties:
    """Optional physical and morphological fields for a galaxy.

    Every field is nullable — different catalogs populate different
    subsets, and the engine treats ``None`` as "unknown" rather than
    zero.
    """

    morphology: str | None = None
    redshift_z: float | None = None
    apparent_magnitude: float | None = None
    color_index: float | None = None
    stellar_mass_kg: float | None = None
    halo_mass_kg: float | None = None


def create_galaxy_object(
    id: str,
    name: str,
    position_m: Vector3,
    properties: GalaxyProperties,
    source: str = "synthetic",
) -> UniverseObject:
    """Construct a :class:`UniverseObject` representing a galaxy.

    ``truth_level`` is :attr:`TruthLevel.PROCEDURAL_APPROXIMATION` when
    ``source`` starts with ``"synthetic"`` (covers the plain
    ``"synthetic"`` sentinel and specialized variants like
    ``"synthetic_desi_like"``), otherwise :attr:`TruthLevel.CATALOG_IMPORTED`.
    """
    truth_level = (
        TruthLevel.PROCEDURAL_APPROXIMATION
        if source.startswith("synthetic")
        else TruthLevel.CATALOG_IMPORTED
    )
    metadata: dict[str, Any] = {
        "morphology": properties.morphology,
        "apparent_magnitude": properties.apparent_magnitude,
        "color_index": properties.color_index,
        "stellar_mass_kg": properties.stellar_mass_kg,
        "halo_mass_kg": properties.halo_mass_kg,
    }
    return UniverseObject(
        id=id,
        name=name,
        object_type=CosmicObjectType.GALAXY,
        position_m=position_m,
        velocity_m_s=Vector3.zero(),
        truth_level=truth_level,
        source=source,
        mass_kg=properties.stellar_mass_kg,
        redshift_z=properties.redshift_z,
        metadata=metadata,
    )
