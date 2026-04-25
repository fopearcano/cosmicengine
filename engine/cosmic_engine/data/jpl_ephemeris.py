"""JPL ephemeris placeholder.

Phase 13 swap: the placeholder now delegates to
:func:`cosmic_engine.physics.solar_system.create_solar_system_objects`,
which advances mean anomalies via a simple Keplerian solver. The
positions are still **not** real JPL ephemerides — they are computed
from approximate J2000.0 mean elements with no perturbations or
relativistic corrections — but they at least move consistently with
time.

A full SPICE / DE4xx ingest is out of scope.
"""

from __future__ import annotations

from cosmic_engine.core.registry import UniverseRegistry
from cosmic_engine.core.universe_object import UniverseObject
from cosmic_engine.data.sources import DataSource, tag_source
from cosmic_engine.physics.solar_system import create_solar_system_objects


# J2000.0 fallback so call sites that don't care about time still work.
_DEFAULT_JD = 2_451_545.0


def load_jpl_ephemeris_placeholder(
    julian_date: float = _DEFAULT_JD,
) -> list[UniverseObject]:
    """Return the Sun and the eight major planets tagged as the JPL source.

    Uses :func:`create_solar_system_objects` internally; positions are
    computed via the simplified Keplerian model and tagged
    ``truth_level=PHYSICS_SIMULATED``. Source is overwritten from
    ``"approx_solar_system"`` to ``"jpl"`` so consumers see a single
    provenance string.
    """
    objects = create_solar_system_objects(julian_date)
    for obj in objects:
        tag_source(obj, DataSource.JPL)
    return objects


def load_jpl_into_registry(
    registry: UniverseRegistry,
    julian_date: float = _DEFAULT_JD,
) -> None:
    """Add the placeholder solar-system snapshot to ``registry`` if absent."""
    for obj in load_jpl_ephemeris_placeholder(julian_date):
        if registry.get_object(obj.id) is None:
            registry.add_object(obj)
