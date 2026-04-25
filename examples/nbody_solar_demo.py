"""Demo: integrate a Sun + Earth + Mars 3-body system with leapfrog."""

from __future__ import annotations

import math

from cosmic_engine.core.units import AU_IN_METERS
from cosmic_engine.core.universe_object import UniverseObject
from cosmic_engine.core.vector import Vector3
from cosmic_engine.physics.nbody import (
    GRAVITATIONAL_CONSTANT,
    NBodySimulator,
    apply_nbody_state_to_objects,
    objects_to_nbody_state,
)
from cosmic_engine.physics.solar_system import create_solar_system_objects


_SECONDS_PER_DAY = 86_400.0


def _circular_velocity(distance_m: float, central_mass_kg: float) -> float:
    """v_circ = √(G·M/r)."""
    return math.sqrt(GRAVITATIONAL_CONSTANT * central_mass_kg / distance_m)


def _bootstrap_solar_system_velocities(
    objects: list[UniverseObject],
) -> None:
    """Assign each planet a circular orbital velocity around the Sun.

    The bundled solar-system snapshot stores zero velocity (Phase 13 only
    cares about positions). For a meaningful N-body simulation we need
    an initial velocity that approximately balances gravity at t=0.
    """
    sun = next(o for o in objects if o.id == "sun")
    sun_mass = sun.mass_kg
    if sun_mass is None:
        raise RuntimeError("Sun mass missing")
    for obj in objects:
        if obj is sun or obj.mass_kg is None:
            continue
        rx = obj.position_m.x - sun.position_m.x
        ry = obj.position_m.y - sun.position_m.y
        rz = obj.position_m.z - sun.position_m.z
        r = math.sqrt(rx * rx + ry * ry + rz * rz)
        if r == 0.0:
            continue
        speed = _circular_velocity(r, sun_mass)
        # circular orbit perpendicular to r, in the xy plane:
        # tangent = (-ry, rx, 0) / |xy| projected onto unit tangent
        norm_xy = math.sqrt(rx * rx + ry * ry)
        if norm_xy == 0.0:
            continue
        tx = -ry / norm_xy
        ty = rx / norm_xy
        obj.velocity_m_s = Vector3(speed * tx, speed * ty, 0.0)


def _earth_sun_distance(objects: list[UniverseObject]) -> float:
    earth = next(o for o in objects if o.id == "earth")
    sun = next(o for o in objects if o.id == "sun")
    return math.sqrt(
        (earth.position_m.x - sun.position_m.x) ** 2
        + (earth.position_m.y - sun.position_m.y) ** 2
        + (earth.position_m.z - sun.position_m.z) ** 2
    )


def main() -> None:
    full_system = create_solar_system_objects(2_451_545.0)
    objects = [o for o in full_system if o.id in ("sun", "earth", "mars")]
    _bootstrap_solar_system_velocities(objects)

    state = objects_to_nbody_state(objects)
    sim = NBodySimulator(state, softening_m=0.0, integrator="leapfrog")

    initial_distance_au = _earth_sun_distance(objects) / AU_IN_METERS
    earth_initial = next(o for o in objects if o.id == "earth").position_m
    mars_initial = next(o for o in objects if o.id == "mars").position_m

    days = 90
    dt_seconds = 3_600.0  # one-hour steps
    steps = int(days * _SECONDS_PER_DAY / dt_seconds)
    sim.run(steps, dt_seconds)

    apply_nbody_state_to_objects(objects, sim.get_state())
    final_distance_au = _earth_sun_distance(objects) / AU_IN_METERS
    earth_final = next(o for o in objects if o.id == "earth").position_m
    mars_final = next(o for o in objects if o.id == "mars").position_m

    print(f"bodies              : {len(state)}")
    print(f"integrator          : {sim.integrator}")
    print(f"step size           : {dt_seconds} s ({dt_seconds / 3600.0:.0f} h)")
    print(f"total steps         : {steps}  ({days} days)")
    print()
    print(f"Earth-Sun distance (AU)")
    print(f"  initial : {initial_distance_au:.6f}")
    print(f"  final   : {final_distance_au:.6f}")
    print(f"  delta   : {final_distance_au - initial_distance_au:+.6f}")
    print()
    print("Earth position (AU):")
    print(
        f"  initial : ({earth_initial.x / AU_IN_METERS:+.4f}, "
        f"{earth_initial.y / AU_IN_METERS:+.4f}, "
        f"{earth_initial.z / AU_IN_METERS:+.4f})"
    )
    print(
        f"  final   : ({earth_final.x / AU_IN_METERS:+.4f}, "
        f"{earth_final.y / AU_IN_METERS:+.4f}, "
        f"{earth_final.z / AU_IN_METERS:+.4f})"
    )
    print()
    print("Mars position (AU):")
    print(
        f"  initial : ({mars_initial.x / AU_IN_METERS:+.4f}, "
        f"{mars_initial.y / AU_IN_METERS:+.4f}, "
        f"{mars_initial.z / AU_IN_METERS:+.4f})"
    )
    print(
        f"  final   : ({mars_final.x / AU_IN_METERS:+.4f}, "
        f"{mars_final.y / AU_IN_METERS:+.4f}, "
        f"{mars_final.z / AU_IN_METERS:+.4f})"
    )

    finite = all(
        math.isfinite(c)
        for o in objects
        for c in (o.position_m.x, o.position_m.y, o.position_m.z)
    )
    print()
    print(f"all positions finite: {finite}")


if __name__ == "__main__":
    main()
