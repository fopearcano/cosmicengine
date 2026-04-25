"""JPL ephemeris placeholder.

A minimal stand-in for a future SPICE / DE-series ephemeris ingest. We
hardcode the Sun, Earth, and Mars at approximate fixed positions so
downstream code can rely on a non-empty solar-system registry today.

Real JPL integration (SPICE kernels, time-resolved state vectors,
osculating elements) is out of scope for Phase 12.
"""

from __future__ import annotations

from cosmic_engine.core.object_types import CosmicObjectType
from cosmic_engine.core.registry import UniverseRegistry
from cosmic_engine.core.truth import TruthLevel
from cosmic_engine.core.units import AU_IN_METERS
from cosmic_engine.core.universe_object import UniverseObject
from cosmic_engine.core.vector import Vector3
from cosmic_engine.data.sources import DataSource, tag_source


# Approximate semi-major axes (AU) along +x for a frozen snapshot.
_BODIES = (
    ("sun", "Sun", CosmicObjectType.STAR, 0.0, 1.989e30, 6.957e8),
    ("earth", "Earth", CosmicObjectType.PLANET, 1.0, 5.972e24, 6.371e6),
    ("mars", "Mars", CosmicObjectType.PLANET, 1.524, 6.39e23, 3.3895e6),
)


def load_jpl_ephemeris_placeholder() -> list[UniverseObject]:
    """Return a minimal Sun / Earth / Mars list at frozen positions.

    All bodies sit on the +x axis at their semi-major-axis distance.
    Velocities are zero (snapshot only). ``truth_level`` is
    :attr:`TruthLevel.EPHEMERIS_REAL` because the values themselves
    come from JPL even though the time evolution is omitted.
    """
    objects: list[UniverseObject] = []
    for obj_id, name, obj_type, semi_major_au, mass_kg, radius_m in _BODIES:
        position = Vector3(semi_major_au * AU_IN_METERS, 0.0, 0.0)
        obj = UniverseObject(
            id=obj_id,
            name=name,
            object_type=obj_type,
            position_m=position,
            velocity_m_s=Vector3.zero(),
            truth_level=TruthLevel.EPHEMERIS_REAL,
            mass_kg=mass_kg,
            radius_m=radius_m,
            metadata={"semi_major_au": semi_major_au},
        )
        tag_source(obj, DataSource.JPL)
        objects.append(obj)
    return objects


def load_jpl_into_registry(registry: UniverseRegistry) -> None:
    """Add the placeholder Sun / Earth / Mars to ``registry`` if absent."""
    for obj in load_jpl_ephemeris_placeholder():
        if registry.get_object(obj.id) is None:
            registry.add_object(obj)
