"""Apply perception transforms directly in Gaussian field space.

Phase 21 / 22 wired neural and deterministic warps into the photon
sample pipeline. Phase 25 fuses that with the Gaussian splatting
renderer: each :class:`GaussianPoint` is treated as a photon sample,
warped by the existing
:mod:`cosmic_engine.perception.transform` machinery, and then
reconstructed back into a :class:`GaussianPoint` with the warped
direction expressed as a position rotation around the observer.

Why warp positions instead of just colors? The splat renderer
projects each point through the camera basis from its world
position, so to actually move the apparent location of a galaxy we
move the underlying position along the warped direction.
"""

from __future__ import annotations

import math

import numpy as np

from ai_viewer.neural_field.gaussian import GaussianPoint
from cosmic_engine.ai.base import AIWarpModel
from cosmic_engine.core.vector import Vector3
from cosmic_engine.perception.observer import ObserverState
from cosmic_engine.perception.transform import transform_photon_sample
from cosmic_engine.rendering.photon_field import PhotonSample


_BRIGHTNESS_CEILING = 1.0e18
_INTENSITY_FLOOR = 0.0
_DEFAULT_OBJECT_TYPE = "galaxy"
_DEFAULT_TRUTH_LEVEL = "procedural_approximation"


def _color_to_int_tuple(color: np.ndarray) -> tuple[int, int, int]:
    return (
        max(0, min(255, int(round(float(color[0]))))),
        max(0, min(255, int(round(float(color[1]))))),
        max(0, min(255, int(round(float(color[2]))))),
    )


def warp_gaussian_point(
    point: GaussianPoint,
    observer: ObserverState,
    ai_model: AIWarpModel | None = None,
) -> GaussianPoint:
    """Warp a single :class:`GaussianPoint` through the perception pipeline.

    Returns a fresh :class:`GaussianPoint`; the input is never mutated.
    The new position lies along the warped unit direction at the same
    distance from the observer, so the splat renderer projects it to a
    new screen-space location consistent with the perception transform.
    """
    dx = float(point.position[0]) - float(observer.position_m.x)
    dy = float(point.position[1]) - float(observer.position_m.y)
    dz = float(point.position[2]) - float(observer.position_m.z)
    distance = math.sqrt(dx * dx + dy * dy + dz * dz)
    metadata_copy = dict(point.metadata)

    if distance == 0.0:
        # Coincident with observer; nothing meaningful to warp.
        metadata_copy.setdefault("warp_factor", float(observer.warp_factor))
        return GaussianPoint(
            position=point.position.copy(),
            color=point.color.copy(),
            intensity=point.intensity,
            sigma=point.sigma,
            object_id=point.object_id,
            truth_level=point.truth_level,
            metadata=metadata_copy,
        )

    direction = Vector3(dx / distance, dy / distance, dz / distance)
    color_int = _color_to_int_tuple(point.color)
    sample = PhotonSample(
        object_id=point.object_id or "",
        name=point.object_id or "",
        object_type=_DEFAULT_OBJECT_TYPE,
        direction=direction,
        distance_m=distance,
        apparent_brightness=max(_INTENSITY_FLOOR, point.intensity),
        color_rgb=color_int,
        truth_level=point.truth_level or _DEFAULT_TRUTH_LEVEL,
    )
    warped = transform_photon_sample(sample, observer, ai_model=ai_model)

    new_position = np.array(
        [
            float(observer.position_m.x) + warped.direction.x * distance,
            float(observer.position_m.y) + warped.direction.y * distance,
            float(observer.position_m.z) + warped.direction.z * distance,
        ],
        dtype=np.float64,
    )
    new_intensity = max(
        0.0,
        min(float(warped.apparent_brightness), _BRIGHTNESS_CEILING),
    )
    new_sigma = max(point.sigma, 0.0) * max(
        1.0, float(observer.warp_factor) ** 0.25
    )
    new_color = np.asarray(warped.color_rgb, dtype=np.float64)

    metadata_copy.setdefault(
        "original_position",
        [float(point.position[0]), float(point.position[1]), float(point.position[2])],
    )
    metadata_copy["warp_factor"] = float(observer.warp_factor)
    metadata_copy["warped"] = True

    return GaussianPoint(
        position=new_position,
        color=new_color,
        intensity=new_intensity,
        sigma=new_sigma,
        object_id=point.object_id,
        truth_level=point.truth_level,
        metadata=metadata_copy,
    )


def warp_gaussian_field(
    points: list[GaussianPoint],
    observer: ObserverState,
    ai_model: AIWarpModel | None = None,
    max_points: int | None = None,
) -> list[GaussianPoint]:
    """Warp every point in a field.

    ``max_points`` truncates deterministically (from the start of the
    list — callers wanting a different ordering should sort first).
    Empty input is returned unchanged.
    """
    if not points:
        return []
    if max_points is not None:
        if max_points <= 0:
            return []
        if len(points) > max_points:
            points = points[:max_points]
    return [warp_gaussian_point(p, observer, ai_model) for p in points]
