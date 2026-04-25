"""Physics utilities for CosmicEngine.

Phase 10 contribution: a tiny cosmology helper that turns redshift
into distance via the linear Hubble law. Explicitly an approximation
(valid only at low z, ignores curvature, dark energy, and matter
density). Future phases can replace it with a full ΛCDM integrator
without touching call sites.
"""

from cosmic_engine.physics.cosmology import (
    HUBBLE_CONSTANT_KM_S_MPC,
    HUBBLE_CONSTANT_PER_S,
    redshift_to_distance_lightyears,
    redshift_to_distance_m,
    redshift_to_velocity,
)

__all__ = [
    "HUBBLE_CONSTANT_KM_S_MPC",
    "HUBBLE_CONSTANT_PER_S",
    "redshift_to_distance_lightyears",
    "redshift_to_distance_m",
    "redshift_to_velocity",
]
