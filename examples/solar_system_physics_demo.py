"""Demo: advance the simplified solar system by 30 days and print the deltas."""

from __future__ import annotations

import math
from pathlib import Path

from cosmic_engine.core.registry import UniverseRegistry
from cosmic_engine.core.time import SimulationClock
from cosmic_engine.core.units import AU_IN_METERS
from cosmic_engine.physics.solar_system import create_solar_system_objects


_J2000 = 2_451_545.0
_OUT = Path(__file__).resolve().parent / "output_solar_system.json"


def _print_positions(label: str, objects):
    print(f"--- {label} ---")
    print(f"{'id':<10} {'x (AU)':>12} {'y (AU)':>12} {'z (AU)':>12} "
          f"{'r (AU)':>12}")
    for obj in objects:
        x = obj.position_m.x / AU_IN_METERS
        y = obj.position_m.y / AU_IN_METERS
        z = obj.position_m.z / AU_IN_METERS
        r = math.sqrt(x * x + y * y + z * z)
        print(f"{obj.id:<10} {x:>12.4f} {y:>12.4f} {z:>12.4f} {r:>12.4f}")
    print()


def main() -> None:
    clock = SimulationClock(current_julian_date=_J2000)
    print(f"clock starts at JD = {clock.get_julian_date()}")
    objects_t0 = create_solar_system_objects(clock.get_julian_date())
    _print_positions("J2000.0 snapshot", objects_t0)

    clock.tick(30.0 * 86_400.0)  # advance 30 days of wall time, time_scale=1
    print(f"clock advances to JD = {clock.get_julian_date()}")
    objects_t1 = create_solar_system_objects(clock.get_julian_date())
    _print_positions("+30 days snapshot", objects_t1)

    by_id_0 = {o.id: o for o in objects_t0}
    by_id_1 = {o.id: o for o in objects_t1}

    print("--- position deltas after 30 days ---")
    for planet_id in ("earth", "mars"):
        a = by_id_0[planet_id].position_m
        b = by_id_1[planet_id].position_m
        dx = (b.x - a.x) / AU_IN_METERS
        dy = (b.y - a.y) / AU_IN_METERS
        dz = (b.z - a.z) / AU_IN_METERS
        d = math.sqrt(dx * dx + dy * dy + dz * dz)
        print(
            f"  {planet_id:<7}  Δr = {d:.4f} AU   "
            f"Δ = ({dx:+.4f}, {dy:+.4f}, {dz:+.4f}) AU"
        )

    registry = UniverseRegistry()
    for obj in objects_t1:
        registry.add_object(obj)
    print()
    print(f"registered {len(registry.list_objects())} solar-system objects")

    serialized = {"objects": [o.to_dict() for o in registry.list_objects()]}
    _OUT.write_text(_format_json(serialized))
    print(f"snapshot written to {_OUT}")


def _format_json(data: dict) -> str:
    import json

    return json.dumps(data, indent=2)


if __name__ == "__main__":
    main()
