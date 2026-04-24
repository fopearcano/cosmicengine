"""Vectorized galaxy point-cloud projection.

The galaxy batch is the large-scale analogue of
:class:`PhotonFieldBatch`: same shape story, but with galaxy-specific
fields (redshift, redshift-derived color). Designed to feed both the
2D PPM renderer and the 3D density grid.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cosmic_engine.core.object_types import CosmicObjectType
from cosmic_engine.core.universe_object import UniverseObject
from cosmic_engine.rendering.simple_camera import SimpleCamera


@dataclass
class GalaxyFieldBatch:
    """All rendering-relevant data for a batch of galaxies."""

    object_ids: list[str]
    names: list[str]
    positions_m: np.ndarray       # shape (N, 3), relative to camera
    directions: np.ndarray        # shape (N, 3), unit vectors
    distances_m: np.ndarray       # shape (N,)
    brightness: np.ndarray        # shape (N,)
    redshifts_z: np.ndarray       # shape (N,), NaN when unknown
    colors_rgb: np.ndarray        # shape (N, 3), float in [0, 255]
    truth_levels: list[str]

    def __len__(self) -> int:
        return len(self.object_ids)


def _empty_batch() -> GalaxyFieldBatch:
    return GalaxyFieldBatch(
        object_ids=[],
        names=[],
        positions_m=np.zeros((0, 3), dtype=np.float64),
        directions=np.zeros((0, 3), dtype=np.float64),
        distances_m=np.zeros((0,), dtype=np.float64),
        brightness=np.zeros((0,), dtype=np.float64),
        redshifts_z=np.zeros((0,), dtype=np.float64),
        colors_rgb=np.zeros((0, 3), dtype=np.float64),
        truth_levels=[],
    )


def _colors_from_redshift(z: np.ndarray) -> np.ndarray:
    """Linear interpolation from white (z ≤ 0) to red-orange (z ≥ 1).

    NaN or negative z values are clamped to 0 (white).
    """
    safe = np.where(np.isfinite(z), z, 0.0)
    t = np.clip(safe, 0.0, 1.0)
    r = np.full_like(t, 255.0)
    g = 255.0 * (1.0 - 0.6 * t)
    b = 255.0 * (1.0 - 0.8 * t)
    return np.stack([r, g, b], axis=1)


def _brightness(
    apparent_magnitudes: np.ndarray,
    distances_m: np.ndarray,
) -> np.ndarray:
    """Magnitude-derived brightness, falling back to inverse-square of distance."""
    mag_mask = np.isfinite(apparent_magnitudes)
    out = np.empty_like(distances_m)
    out[mag_mask] = np.power(10.0, -0.4 * apparent_magnitudes[mag_mask])
    safe_d = np.maximum(distances_m[~mag_mask], 1.0)
    out[~mag_mask] = 1.0 / (safe_d * safe_d)
    return out


def build_galaxy_field_batch(
    objects: list[UniverseObject],
    camera: SimpleCamera,
) -> GalaxyFieldBatch:
    """Assemble a :class:`GalaxyFieldBatch` for every galaxy in ``objects``.

    Non-galaxy objects and galaxies coincident with the camera position
    are skipped.
    """
    cam = camera.position_m
    raw_positions: list[tuple[float, float, float]] = []
    ids: list[str] = []
    names: list[str] = []
    truths: list[str] = []
    magnitudes: list[float] = []
    redshifts: list[float] = []

    for obj in objects:
        if obj.object_type is not CosmicObjectType.GALAXY:
            continue
        dx = obj.position_m.x - cam.x
        dy = obj.position_m.y - cam.y
        dz = obj.position_m.z - cam.z
        if dx == 0.0 and dy == 0.0 and dz == 0.0:
            continue
        raw_positions.append((dx, dy, dz))
        ids.append(obj.id)
        names.append(obj.name)
        truths.append(obj.truth_level.value)
        mag = obj.metadata.get("apparent_magnitude") if obj.metadata else None
        magnitudes.append(float(mag) if mag is not None else float("nan"))
        z = obj.redshift_z
        redshifts.append(float(z) if z is not None else float("nan"))

    if not raw_positions:
        return _empty_batch()

    positions = np.asarray(raw_positions, dtype=np.float64)
    distances = np.linalg.norm(positions, axis=1)
    directions = positions / distances[:, None]

    mags = np.asarray(magnitudes, dtype=np.float64)
    zs = np.asarray(redshifts, dtype=np.float64)

    brightness = _brightness(mags, distances)
    colors = _colors_from_redshift(zs)

    return GalaxyFieldBatch(
        object_ids=ids,
        names=names,
        positions_m=positions,
        directions=directions,
        distances_m=distances,
        brightness=brightness,
        redshifts_z=zs,
        colors_rgb=colors,
        truth_levels=truths,
    )
