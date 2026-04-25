"""Scale-aware zone selection driven by observer distance."""

from __future__ import annotations

from typing import TYPE_CHECKING

from cosmic_engine.multiscale.representation import (
    get_representation_for_zone,
)
from cosmic_engine.multiscale.scale_zone import ScaleZone

if TYPE_CHECKING:
    from cosmic_engine.runtime.runtime import CosmicRuntime


# Reasonable defaults spanning ~37 orders of magnitude:
# 1 m → solar system → interstellar → intergalactic → cosmological.
DEFAULT_ZONES: tuple[ScaleZone, ...] = (
    ScaleZone(
        name="microscale",
        min_scale_m=0.0,
        max_scale_m=1.0e9,
        representation_type="density_field",
    ),
    ScaleZone(
        name="solar_system",
        min_scale_m=1.0e9,
        max_scale_m=1.0e16,
        representation_type="nbody",
    ),
    ScaleZone(
        name="interstellar",
        min_scale_m=1.0e16,
        max_scale_m=1.0e20,
        representation_type="star_field",
    ),
    ScaleZone(
        name="intergalactic",
        min_scale_m=1.0e20,
        max_scale_m=1.0e27,
        representation_type="galaxy_field",
    ),
)


class ScaleManager:
    """Pick the right :class:`ScaleZone` for an observer scale."""

    def __init__(self, zones: list[ScaleZone] | tuple[ScaleZone, ...]) -> None:
        if not zones:
            raise ValueError("ScaleManager needs at least one zone")
        sorted_zones = sorted(zones, key=lambda z: z.min_scale_m)
        # Validate that the sorted zones are non-overlapping (allowing
        # touching boundaries — adjacent zones should align exactly).
        for prev, nxt in zip(sorted_zones, sorted_zones[1:]):
            if nxt.min_scale_m < prev.max_scale_m:
                raise ValueError(
                    f"ScaleManager zones overlap between {prev.name!r} "
                    f"and {nxt.name!r}"
                )
        self.zones: tuple[ScaleZone, ...] = tuple(sorted_zones)

    def get_zone(self, distance_m: float) -> ScaleZone:
        """Return the zone covering ``distance_m`` (clamping at the ends)."""
        if distance_m < self.zones[0].min_scale_m:
            return self.zones[0]
        if distance_m >= self.zones[-1].max_scale_m:
            return self.zones[-1]
        for z in self.zones:
            if z.contains_scale(distance_m):
                return z
        # Should be unreachable given the half-open interval contract; fall
        # back to the smallest zone above the distance to be safe.
        for z in self.zones:
            if z.min_scale_m >= distance_m:
                return z
        return self.zones[-1]

    def get_neighbor_above(self, zone: ScaleZone) -> ScaleZone | None:
        """Return the zone immediately above ``zone`` in scale, if any."""
        for i, z in enumerate(self.zones):
            if z is zone or z.name == zone.name:
                if i + 1 < len(self.zones):
                    return self.zones[i + 1]
                return None
        return None

    def get_neighbor_below(self, zone: ScaleZone) -> ScaleZone | None:
        """Return the zone immediately below ``zone`` in scale, if any."""
        for i, z in enumerate(self.zones):
            if z is zone or z.name == zone.name:
                if i > 0:
                    return self.zones[i - 1]
                return None
        return None

    def get_representation(
        self,
        distance_m: float,
        runtime: "CosmicRuntime",
    ) -> dict:
        """Convenience: zone -> :func:`get_representation_for_zone`."""
        return get_representation_for_zone(self.get_zone(distance_m), runtime)
