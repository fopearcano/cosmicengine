"""Stepwise geodesic ray marcher.

Wraps :func:`integrate_geodesic_step` in a fixed-iteration loop so
the GPU shader and the CPU path use the same control flow. Each
ray either survives ``max_steps`` iterations (returning the final
direction) or gets absorbed when its position falls inside the
black hole's event horizon.
"""

from __future__ import annotations

import numpy as np

from ai_viewer.neural_field.gaussian import GaussianPoint
from ai_viewer.neural_field.gr.black_hole import BlackHole
from ai_viewer.neural_field.gr.geodesic import integrate_geodesic_step


class GeodesicRayMarcher:
    """Ray marcher that advances photons along approximate geodesics."""

    def __init__(
        self,
        black_hole: BlackHole,
        step_size: float,
        max_steps: int,
    ) -> None:
        if step_size <= 0.0:
            raise ValueError("step_size must be positive")
        if max_steps <= 0:
            raise ValueError("max_steps must be positive")
        self.black_hole = black_hole
        self.step_size = float(step_size)
        self.max_steps = int(max_steps)

    def trace_ray(
        self,
        origin: np.ndarray,
        direction: np.ndarray,
    ) -> tuple[np.ndarray, bool]:
        """Trace a ray from ``origin`` in ``direction``.

        Returns ``(final_direction, absorbed)``. If the photon
        crosses the event horizon at any step, ``absorbed`` is
        ``True`` and the partially-bent direction at that step is
        returned. The mass position is removed before stepping and
        not re-added to the result.
        """
        d = np.asarray(direction, dtype=np.float64).reshape(3)
        norm = float(np.linalg.norm(d))
        if norm == 0.0:
            return d.copy(), False
        d = d / norm

        bh_offset = self.black_hole.position
        position = (
            np.asarray(origin, dtype=np.float64).reshape(3) - bh_offset
        )
        rs = self.black_hole.schwarzschild_radius()

        for _ in range(self.max_steps):
            r = float(np.linalg.norm(position))
            if r <= rs:
                return d, True
            position, d = integrate_geodesic_step(
                position, d, self.step_size, self.black_hole.mass_kg
            )
        # Re-normalize at the end so downstream `observer + d * distance`
        # preserves the distance exactly even after N accumulated cos/sin
        # updates.
        final_norm = float(np.linalg.norm(d))
        if final_norm > 0.0:
            d = d / final_norm
        return d, False


def trace_points_through_geodesic(
    points: list[GaussianPoint],
    marcher: GeodesicRayMarcher,
    observer_position: np.ndarray,
) -> list[GaussianPoint]:
    """Apply the ray marcher to every point's apparent direction.

    Each point's distance from the observer is preserved; only the
    direction is replaced by the marched final direction. Points
    whose rays fall into the event horizon are removed. Original
    points are never mutated.
    """
    if not points:
        return []
    observer = np.asarray(observer_position, dtype=np.float64).reshape(3)
    out: list[GaussianPoint] = []
    for p in points:
        offset = p.position - observer
        distance = float(np.linalg.norm(offset))
        if distance == 0.0:
            out.append(_clone(p, geodesic_traced=False, absorbed=False))
            continue
        direction = offset / distance
        new_dir, absorbed = marcher.trace_ray(observer, direction)
        if absorbed:
            continue
        new_position = observer + new_dir * distance
        out.append(
            _clone(
                p,
                position=new_position,
                geodesic_traced=True,
                absorbed=False,
            )
        )
    return out


def _clone(
    point: GaussianPoint,
    *,
    position: np.ndarray | None = None,
    geodesic_traced: bool,
    absorbed: bool,
) -> GaussianPoint:
    metadata = dict(point.metadata)
    metadata["geodesic_traced"] = geodesic_traced
    metadata["absorbed"] = absorbed
    return GaussianPoint(
        position=position if position is not None else point.position.copy(),
        color=point.color.copy(),
        intensity=point.intensity,
        sigma=point.sigma,
        object_id=point.object_id,
        truth_level=point.truth_level,
        metadata=metadata,
    )
