"""Level-of-Detail selection driven by inverse-square distance weights.

Closer objects are more likely to be kept; far objects are
probabilistically dropped. The result is a bounded-size sample that
preserves the spatial distribution rather than just the nearest N
hits.

Selection is deterministic for a given ``(observer_position,
max_objects, seed)`` triple so frames are reproducible.
"""

from __future__ import annotations

import math

import numpy as np

from cosmic_engine.core.universe_object import UniverseObject
from cosmic_engine.core.vector import Vector3


_DEFAULT_EPSILON_M = 1.0e3  # softening so a coincident object isn't ∞


def compute_lod_weight(
    distance_m: float,
    epsilon_m: float = _DEFAULT_EPSILON_M,
) -> float:
    """Return ``1 / (distance² + ε²)``. Always strictly positive."""
    return 1.0 / (distance_m * distance_m + epsilon_m * epsilon_m)


def select_lod_objects(
    objects: list[UniverseObject],
    observer_position: Vector3,
    max_objects: int,
    *,
    seed: int = 42,
) -> list[UniverseObject]:
    """Probabilistically downsample ``objects`` to at most ``max_objects``.

    Sampling is without replacement, weighted by
    :func:`compute_lod_weight`. If the input is already short enough
    the original list is returned unchanged.
    """
    if max_objects <= 0:
        return []
    if not objects:
        return []
    if len(objects) <= max_objects:
        return list(objects)

    weights = np.empty(len(objects), dtype=np.float64)
    for i, obj in enumerate(objects):
        dx = obj.position_m.x - observer_position.x
        dy = obj.position_m.y - observer_position.y
        dz = obj.position_m.z - observer_position.z
        d = math.sqrt(dx * dx + dy * dy + dz * dz)
        weights[i] = compute_lod_weight(d)

    total = float(weights.sum())
    if not math.isfinite(total) or total <= 0.0:
        # All-zero weights (objects infinitely far): just take a deterministic
        # prefix so the function still returns something sane.
        return list(objects[:max_objects])

    probabilities = weights / total
    rng = np.random.default_rng(seed)
    chosen = rng.choice(
        len(objects),
        size=max_objects,
        replace=False,
        p=probabilities,
    )
    chosen.sort()  # stable ordering for downstream determinism
    return [objects[int(i)] for i in chosen]
