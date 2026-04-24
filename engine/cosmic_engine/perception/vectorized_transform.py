"""Vectorized counterpart of :mod:`cosmic_engine.perception.transform`.

Operates on a :class:`PhotonFieldBatch` in whole-array NumPy ops;
conceptually identical to the scalar deterministic transform, modulo
the order of floating-point operations.

AI integration deliberately lives only on the scalar path for now:
this module is pure math and stays dependency-light (NumPy only).
"""

from __future__ import annotations

import numpy as np

from cosmic_engine.perception.observer import ObserverState
from cosmic_engine.rendering.vectorized_photon_field import PhotonFieldBatch


_BRIGHTNESS_CEILING = 1.0e18
_FACTOR_CEILING = 1.0e12


def transform_photon_field_batch(
    batch: PhotonFieldBatch,
    observer: ObserverState,
) -> PhotonFieldBatch:
    """Apply the deterministic perception transform to every sample at once."""
    n = len(batch)
    if n == 0:
        return batch

    forward = np.array(
        [observer.forward.x, observer.forward.y, observer.forward.z],
        dtype=np.float64,
    )
    fnorm = np.linalg.norm(forward)
    if fnorm == 0.0:
        return batch
    forward /= fnorm

    beta = observer.beta()
    warp = float(observer.warp_factor)

    alignment = batch.directions @ forward  # (N,)

    # --- direction warp ---
    amount = beta * alignment * warp  # (N,)
    warped = batch.directions + forward[None, :] * amount[:, None]
    norms = np.linalg.norm(warped, axis=1, keepdims=True)
    safe = np.where(norms == 0.0, 1.0, norms)
    new_directions = np.where(norms == 0.0, warped, warped / safe)

    # --- brightness warp ---
    factor = np.maximum(1.0 + beta * alignment, 0.0)
    # np.power(0, 0) == 1 here, matching the scalar path (0**warp==0 for warp>0)
    with np.errstate(invalid="ignore", over="ignore"):
        factor = np.power(factor, warp)
    factor = np.where(np.isfinite(factor), factor, _FACTOR_CEILING)
    factor = np.minimum(factor, _FACTOR_CEILING)
    new_brightness = np.minimum(
        np.maximum(batch.brightness * factor, 0.0), _BRIGHTNESS_CEILING
    )

    # --- color warp ---
    shift = np.clip(beta * alignment * warp, -1.0, 1.0)  # (N,)
    r = batch.colors_rgb[:, 0]
    g = batch.colors_rgb[:, 1]
    b = batch.colors_rgb[:, 2]

    positive = shift > 0.0
    negative = shift < 0.0

    new_b_pos = b + (255.0 - b) * shift
    new_r_pos = r * (1.0 - shift)
    s = -shift
    new_r_neg = r + (255.0 - r) * s
    new_b_neg = b * (1.0 - s)

    new_r = np.where(positive, new_r_pos, np.where(negative, new_r_neg, r))
    new_b = np.where(positive, new_b_pos, np.where(negative, new_b_neg, b))
    new_g = g

    new_colors = np.clip(
        np.stack([new_r, new_g, new_b], axis=1), 0.0, 255.0
    )

    return PhotonFieldBatch(
        object_ids=batch.object_ids,
        names=batch.names,
        object_types=batch.object_types,
        directions=new_directions,
        distances_m=batch.distances_m,
        brightness=new_brightness,
        colors_rgb=new_colors,
        truth_levels=batch.truth_levels,
    )
