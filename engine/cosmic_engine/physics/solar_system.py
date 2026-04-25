"""Simplified Keplerian solar-system snapshot.

Element values are approximate J2000.0 mean orbits — good enough for
dynamics demos and scaffolding tests, **not** good enough for mission
planning. For real ephemerides, swap in a SPICE / DE4xx integrator
once those become acceptable dependencies.

Truth level for everything here is :attr:`TruthLevel.PHYSICS_SIMULATED`
— positions are computed from elements, not looked up from JPL.
"""

from __future__ import annotations

from dataclasses import dataclass

from cosmic_engine.core.object_types import CosmicObjectType
from cosmic_engine.core.truth import TruthLevel
from cosmic_engine.core.units import AU_IN_METERS
from cosmic_engine.core.universe_object import UniverseObject
from cosmic_engine.core.vector import Vector3
from cosmic_engine.physics.orbital import (
    OrbitalElements,
    orbital_position_from_elements,
)


# J2000.0 epoch.
_J2000_JD = 2_451_545.0
_YEAR_SECONDS = 365.25 * 86_400.0


@dataclass(frozen=True)
class _PlanetSpec:
    obj_id: str
    name: str
    a_au: float
    e: float
    i_deg: float
    big_omega_deg: float           # longitude of ascending node Ω
    long_periapsis_deg: float      # longitude of periapsis ϖ = Ω + ω
    mean_longitude_deg: float      # L = Ω + ω + M
    period_years: float
    mass_kg: float
    radius_m: float


# Approximate mean elements at J2000.0 (heliocentric ecliptic).
_PLANETS: tuple[_PlanetSpec, ...] = (
    _PlanetSpec("mercury", "Mercury", 0.387098, 0.205630, 7.0050,
                48.331, 77.456, 252.251, 0.2408467, 3.301e23, 2.4397e6),
    _PlanetSpec("venus", "Venus", 0.723332, 0.006772, 3.3946,
                76.680, 131.564, 181.980, 0.6151973, 4.867e24, 6.0518e6),
    _PlanetSpec("earth", "Earth", 1.000000, 0.016710, 0.0000,
                -11.260, 102.947, 100.464, 1.0000174, 5.972e24, 6.371e6),
    _PlanetSpec("mars", "Mars", 1.523679, 0.093400, 1.8500,
                49.578, 336.080, 355.453, 1.8808476, 6.39e23, 3.3895e6),
    _PlanetSpec("jupiter", "Jupiter", 5.202887, 0.048386, 1.3047,
                100.464, 14.331, 34.404, 11.862615, 1.898e27, 6.9911e7),
    _PlanetSpec("saturn", "Saturn", 9.536676, 0.053862, 2.4853,
                113.665, 93.057, 49.944, 29.447498, 5.683e26, 5.8232e7),
    _PlanetSpec("uranus", "Uranus", 19.189164, 0.047257, 0.7733,
                74.006, 173.005, 313.232, 84.016846, 8.681e25, 2.5362e7),
    _PlanetSpec("neptune", "Neptune", 30.069923, 0.008590, 1.7700,
                131.784, 48.123, -55.120, 164.79132, 1.024e26, 2.4622e7),
)


def _elements_for(spec: _PlanetSpec) -> OrbitalElements:
    omega_deg = spec.long_periapsis_deg - spec.big_omega_deg
    mean_anomaly_deg = spec.mean_longitude_deg - spec.long_periapsis_deg
    return OrbitalElements(
        semi_major_axis_m=spec.a_au * AU_IN_METERS,
        eccentricity=spec.e,
        inclination_deg=spec.i_deg,
        longitude_ascending_node_deg=spec.big_omega_deg,
        argument_of_periapsis_deg=omega_deg,
        mean_anomaly_at_epoch_deg=mean_anomaly_deg,
        epoch_julian_date=_J2000_JD,
        orbital_period_seconds=spec.period_years * _YEAR_SECONDS,
    )


def _make_sun() -> UniverseObject:
    return UniverseObject(
        id="sun",
        name="Sun",
        object_type=CosmicObjectType.STAR,
        position_m=Vector3.zero(),
        velocity_m_s=Vector3.zero(),
        truth_level=TruthLevel.PHYSICS_SIMULATED,
        source="approx_solar_system",
        mass_kg=1.989e30,
        radius_m=6.957e8,
        metadata={"model": "simplified_keplerian"},
    )


def _make_planet(spec: _PlanetSpec, julian_date: float) -> UniverseObject:
    elements = _elements_for(spec)
    position = orbital_position_from_elements(elements, julian_date)
    return UniverseObject(
        id=spec.obj_id,
        name=spec.name,
        object_type=CosmicObjectType.PLANET,
        position_m=position,
        velocity_m_s=Vector3.zero(),
        truth_level=TruthLevel.PHYSICS_SIMULATED,
        source="approx_solar_system",
        mass_kg=spec.mass_kg,
        radius_m=spec.radius_m,
        metadata={
            "model": "simplified_keplerian",
            "semi_major_axis_au": spec.a_au,
            "eccentricity": spec.e,
            "orbital_period_years": spec.period_years,
        },
    )


def create_solar_system_objects(julian_date: float) -> list[UniverseObject]:
    """Return the Sun and the eight major planets at ``julian_date``.

    Positions for the planets are computed via the simplified Keplerian
    solver in :mod:`cosmic_engine.physics.orbital`; the Sun sits at the
    heliocentric origin by convention.
    """
    objects: list[UniverseObject] = [_make_sun()]
    for spec in _PLANETS:
        objects.append(_make_planet(spec, julian_date))
    return objects
