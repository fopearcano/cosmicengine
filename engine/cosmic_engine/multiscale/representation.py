"""Map a :class:`ScaleZone` to a renderable bundle of registry objects.

The result is intentionally a plain ``dict`` with primitive fields so
the multi-scale system stays decoupled from any specific renderer or
backend. Downstream code (viewer, runtime) inspects ``type`` and
``objects`` and decides how to draw.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from cosmic_engine.core.object_types import CosmicObjectType
from cosmic_engine.multiscale.scale_zone import ScaleZone

if TYPE_CHECKING:
    from cosmic_engine.runtime.runtime import CosmicRuntime


REPRESENTATION_TYPES = (
    "galaxy_field",
    "star_field",
    "nbody",
    "density_field",
    "neural_field",
)


def get_representation_for_zone(
    zone: ScaleZone,
    runtime: "CosmicRuntime",
) -> dict[str, Any]:
    """Filter the runtime's registry into a representation for ``zone``.

    Returns a dict with at least::

        {
            "type":         <zone.representation_type>,
            "zone_name":    <zone.name>,
            "objects":      <list of UniverseObject>,
            "object_count": <int>,
        }
    """
    objects = runtime.registry.list_objects()
    rep_type = zone.representation_type

    if rep_type == "galaxy_field":
        filtered = [
            o for o in objects if o.object_type is CosmicObjectType.GALAXY
        ]
    elif rep_type == "star_field":
        filtered = [
            o for o in objects if o.object_type is CosmicObjectType.STAR
        ]
    elif rep_type == "nbody":
        filtered = [
            o
            for o in objects
            if o.mass_kg is not None
            and o.object_type
            in (
                CosmicObjectType.STAR,
                CosmicObjectType.PLANET,
                CosmicObjectType.MOON,
                CosmicObjectType.ASTEROID,
                CosmicObjectType.COMET,
                CosmicObjectType.BLACK_HOLE,
                CosmicObjectType.NEUTRON_STAR,
                CosmicObjectType.WHITE_DWARF,
            )
        ]
    elif rep_type == "density_field":
        # Density representations operate over the full registry (the
        # density grid handles its own filtering downstream).
        filtered = list(objects)
    elif rep_type == "neural_field":
        # Neural field consumes whatever Gaussian source the caller
        # decides to use; default to the full registry.
        filtered = list(objects)
    else:  # pragma: no cover - guarded by ScaleZone.__post_init__
        filtered = list(objects)

    return {
        "type": rep_type,
        "zone_name": zone.name,
        "objects": filtered,
        "object_count": len(filtered),
        "blended": False,
    }
