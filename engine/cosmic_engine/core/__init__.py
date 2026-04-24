"""Core data model for CosmicEngine.

Every future subsystem (data ingestion, physics, rendering, AI) must build
on the primitives defined here: truth levels, cosmic object types, vectors,
universe objects, and the registry that holds them.
"""

from cosmic_engine.core.object_types import CosmicObjectType
from cosmic_engine.core.registry import UniverseRegistry
from cosmic_engine.core.truth import TruthLevel
from cosmic_engine.core.universe_object import UniverseObject
from cosmic_engine.core.vector import Vector3

__all__ = [
    "CosmicObjectType",
    "TruthLevel",
    "UniverseObject",
    "UniverseRegistry",
    "Vector3",
]
