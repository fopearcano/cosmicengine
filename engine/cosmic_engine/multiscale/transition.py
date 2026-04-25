"""Smooth transitions between adjacent scale zones."""

from __future__ import annotations

import math
from typing import Any

from cosmic_engine.multiscale.scale_zone import ScaleZone


def compute_transition_alpha(
    distance_m: float,
    zone_a: ScaleZone,
    zone_b: ScaleZone,
    *,
    blend_width: float = 0.1,
) -> float:
    """Return blend weight ``alpha ∈ [0, 1]``.

    ``alpha == 0`` means fully ``zone_a``; ``alpha == 1`` means fully
    ``zone_b``. The boundary is taken to be ``zone_a.max_scale_m``;
    ``blend_width`` is the fraction of ``zone_a``'s range over which
    the ramp spans (centered on the boundary).
    """
    if blend_width <= 0.0:
        return 0.0 if distance_m < zone_a.max_scale_m else 1.0
    if blend_width > 1.0:
        blend_width = 1.0
    boundary = float(zone_a.max_scale_m)
    width = (zone_a.max_scale_m - zone_a.min_scale_m) * blend_width
    lower = boundary - 0.5 * width
    upper = boundary + 0.5 * width
    if distance_m <= lower:
        return 0.0
    if distance_m >= upper:
        return 1.0
    return float((distance_m - lower) / (upper - lower))


def blend_representations(
    rep_a: dict[str, Any],
    rep_b: dict[str, Any],
    alpha: float,
) -> dict[str, Any]:
    """Mix two representation dicts with weight ``alpha``.

    Object selection is deterministic: a ``(1 − alpha)`` fraction of
    ``rep_a``'s objects (taken from the start of the list) and an
    ``alpha`` fraction of ``rep_b``'s objects. Returns a new dict
    with ``type = "blended"`` and metadata recording both source
    types.
    """
    if alpha <= 0.0:
        return dict(rep_a)
    if alpha >= 1.0:
        return dict(rep_b)

    objects_a = list(rep_a.get("objects", []))
    objects_b = list(rep_b.get("objects", []))
    # Use ceil so a non-empty source still contributes at least one
    # object whenever its weight is > 0, instead of disappearing to
    # rounding (Python's round-half-to-even sends 0.5 to 0).
    take_a = max(0, math.ceil(len(objects_a) * (1.0 - alpha)))
    take_b = max(0, math.ceil(len(objects_b) * alpha))
    blended_objects = objects_a[:take_a] + objects_b[:take_b]

    return {
        "type": "blended",
        "primary_type": rep_a.get("type"),
        "secondary_type": rep_b.get("type"),
        "primary_zone": rep_a.get("zone_name"),
        "secondary_zone": rep_b.get("zone_name"),
        "alpha": float(alpha),
        "objects": blended_objects,
        "object_count": len(blended_objects),
        "blended": True,
    }
