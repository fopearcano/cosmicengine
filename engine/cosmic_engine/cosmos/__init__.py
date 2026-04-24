"""Cosmological object types beyond the core model.

Phase 8 contribution: domain-specific helpers for building large-scale
objects (galaxies today, halos / filaments / nebulae later) into the
shared :class:`UniverseObject` record type.
"""

from cosmic_engine.cosmos.galaxy import GalaxyProperties, create_galaxy_object

__all__ = ["GalaxyProperties", "create_galaxy_object"]
