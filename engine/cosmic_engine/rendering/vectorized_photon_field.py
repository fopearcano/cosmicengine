"""Batched photon-field construction for large catalogs.

:class:`PhotonFieldBatch` is the vectorized sibling of
:class:`PhotonSample`: one dataclass holding the same information as a
list of samples, but with numeric fields stored as NumPy arrays so the
perception transform can operate on everything at once.

The scalar pipeline (``build_star_photon_field`` / :class:`PhotonSample`)
remains the reference implementation; this module produces identical
results (modulo float precision) but is orders of magnitude faster for
10k+ object catalogs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cosmic_engine.core.object_types import CosmicObjectType
from cosmic_engine.core.universe_object import UniverseObject
from cosmic_engine.core.vector import Vector3
from cosmic_engine.rendering.photon_field import (
    PhotonSample,
    _color_for_spectral_class,
)
from cosmic_engine.rendering.simple_camera import SimpleCamera


@dataclass
class PhotonFieldBatch:
    """Vectorized photon field. All numeric fields align by index."""

    object_ids: list[str]
    names: list[str]
    object_types: list[str]
    directions: np.ndarray       # shape (N, 3), float64, unit vectors
    distances_m: np.ndarray      # shape (N,),  float64
    brightness: np.ndarray       # shape (N,),  float64
    colors_rgb: np.ndarray       # shape (N, 3), float64, values in [0, 255]
    truth_levels: list[str]

    def __len__(self) -> int:
        return len(self.object_ids)


def _empty_batch() -> PhotonFieldBatch:
    return PhotonFieldBatch(
        object_ids=[],
        names=[],
        object_types=[],
        directions=np.zeros((0, 3), dtype=np.float64),
        distances_m=np.zeros((0,), dtype=np.float64),
        brightness=np.zeros((0,), dtype=np.float64),
        colors_rgb=np.zeros((0, 3), dtype=np.float64),
        truth_levels=[],
    )


def build_star_photon_field_batch(
    objects: list[UniverseObject],
    camera: SimpleCamera,
) -> PhotonFieldBatch:
    """Build a :class:`PhotonFieldBatch` for every star in ``objects``.

    Non-star objects and stars coincident with the camera are skipped,
    matching :func:`cosmic_engine.rendering.photon_field.build_star_photon_field`.
    """
    cam = camera.position_m
    positions: list[tuple[float, float, float]] = []
    ids: list[str] = []
    names: list[str] = []
    types: list[str] = []
    truths: list[str] = []
    magnitudes: list[float | None] = []
    colors: list[tuple[int, int, int]] = []

    for obj in objects:
        if obj.object_type is not CosmicObjectType.STAR:
            continue
        dx = obj.position_m.x - cam.x
        dy = obj.position_m.y - cam.y
        dz = obj.position_m.z - cam.z
        if dx == 0.0 and dy == 0.0 and dz == 0.0:
            continue
        positions.append((dx, dy, dz))
        ids.append(obj.id)
        names.append(obj.name)
        types.append(obj.object_type.value)
        truths.append(obj.truth_level.value)
        mag = obj.metadata.get("apparent_magnitude")
        magnitudes.append(float(mag) if mag is not None else None)
        colors.append(_color_for_spectral_class(obj.spectral_class))

    if not positions:
        return _empty_batch()

    pos_arr = np.asarray(positions, dtype=np.float64)
    distances = np.linalg.norm(pos_arr, axis=1)
    directions = pos_arr / distances[:, None]

    brightness = np.array(
        [10.0 ** (-0.4 * m) if m is not None else 1.0 for m in magnitudes],
        dtype=np.float64,
    )
    colors_arr = np.asarray(colors, dtype=np.float64)

    return PhotonFieldBatch(
        object_ids=ids,
        names=names,
        object_types=types,
        directions=directions,
        distances_m=distances,
        brightness=brightness,
        colors_rgb=colors_arr,
        truth_levels=truths,
    )


def photon_batch_to_samples(batch: PhotonFieldBatch) -> list[PhotonSample]:
    """Expand a :class:`PhotonFieldBatch` into a list of :class:`PhotonSample`.

    Colors are clamped to ``[0, 255]`` and integer-rounded so samples
    are byte-identical to the scalar pipeline's output shape.
    """
    samples: list[PhotonSample] = []
    clamped_colors = np.clip(batch.colors_rgb, 0.0, 255.0)
    for i in range(len(batch)):
        r, g, b = clamped_colors[i]
        samples.append(
            PhotonSample(
                object_id=batch.object_ids[i],
                name=batch.names[i],
                object_type=batch.object_types[i],
                direction=Vector3(
                    float(batch.directions[i, 0]),
                    float(batch.directions[i, 1]),
                    float(batch.directions[i, 2]),
                ),
                distance_m=float(batch.distances_m[i]),
                apparent_brightness=float(batch.brightness[i]),
                color_rgb=(
                    int(round(float(r))),
                    int(round(float(g))),
                    int(round(float(b))),
                ),
                truth_level=batch.truth_levels[i],
            )
        )
    return samples
