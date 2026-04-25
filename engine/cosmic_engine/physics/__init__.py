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
    angular_diameter_distance_m,
    comoving_distance_m,
    distance_modulus,
    get_cosmology_mode,
    luminosity_distance_m,
    redshift_to_distance_lightyears,
    redshift_to_distance_m,
    redshift_to_velocity,
    set_cosmology_mode,
)
from cosmic_engine.physics.orbital import (
    OrbitalElements,
    mean_motion,
    orbital_position_from_elements,
    solve_kepler_equation,
)
from cosmic_engine.physics.solar_system import create_solar_system_objects

__all__ = [
    "HUBBLE_CONSTANT_KM_S_MPC",
    "HUBBLE_CONSTANT_PER_S",
    "OrbitalElements",
    "angular_diameter_distance_m",
    "comoving_distance_m",
    "create_solar_system_objects",
    "distance_modulus",
    "get_cosmology_mode",
    "luminosity_distance_m",
    "mean_motion",
    "orbital_position_from_elements",
    "redshift_to_distance_lightyears",
    "redshift_to_distance_m",
    "redshift_to_velocity",
    "set_cosmology_mode",
    "solve_kepler_equation",
]
