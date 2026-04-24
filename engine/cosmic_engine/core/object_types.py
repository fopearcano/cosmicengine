"""Enumeration of cosmic object categories recognized by the engine."""

from enum import Enum


class CosmicObjectType(str, Enum):
    """Category of a :class:`UniverseObject`.

    The taxonomy is intentionally coarse. Finer classification (spectral
    class, galaxy morphology, etc.) lives on the object itself or in its
    metadata, not in this enum.
    """

    STAR = "star"
    PLANET = "planet"
    MOON = "moon"
    ASTEROID = "asteroid"
    COMET = "comet"
    BLACK_HOLE = "black_hole"
    NEUTRON_STAR = "neutron_star"
    WHITE_DWARF = "white_dwarf"
    GALAXY = "galaxy"
    NEBULA = "nebula"
    STAR_CLUSTER = "star_cluster"
    COSMIC_WEB_NODE = "cosmic_web_node"
    DARK_MATTER_HALO = "dark_matter_halo"
    PROCEDURAL_FIELD = "procedural_field"
    UNKNOWN = "unknown"
