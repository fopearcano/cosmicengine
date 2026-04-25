"""Vectorized photon-space AI warp with deterministic fallback.

Wraps :class:`cosmic_engine.ai.onnx_photon_warp_batch.BatchONNXPhotonWarpModel`
in an integration layer that is fallback-safe: if the model raises or
its output is malformed, the call silently routes to the existing
deterministic vectorized transform
(:func:`cosmic_engine.perception.vectorized_transform.transform_photon_field_batch`).

Single-photon inference still lives in
:mod:`cosmic_engine.perception.transform`; this module is the
many-photons-at-once path.
"""

from __future__ import annotations

import numpy as np

from cosmic_engine.perception.observer import ObserverState
from cosmic_engine.perception.vectorized_transform import (
    transform_photon_field_batch,
)
from cosmic_engine.rendering.vectorized_photon_field import PhotonFieldBatch


def build_photon_warp_input(
    batch: PhotonFieldBatch,
    observer: ObserverState,
) -> np.ndarray:
    """Pack a :class:`PhotonFieldBatch` + :class:`ObserverState` into ``(N, 9)``.

    Column layout matches the schema documented on
    :class:`BatchONNXPhotonWarpModel`.
    """
    n = len(batch)
    if n == 0:
        return np.zeros((0, 9), dtype=np.float32)
    out = np.empty((n, 9), dtype=np.float32)
    out[:, 0:3] = batch.directions
    out[:, 3] = batch.brightness
    out[:, 4:7] = batch.colors_rgb
    out[:, 7] = float(observer.beta())
    out[:, 8] = float(observer.warp_factor)
    return out


def validate_warp_output(output: np.ndarray, expected_count: int) -> np.ndarray:
    """Raise :class:`ValueError` if ``output`` is not ``(expected_count, 7)``.

    Returns ``output`` unchanged on success so callers can chain.
    """
    if output.ndim != 2 or output.shape[1] != 7:
        raise ValueError(
            f"validate_warp_output expects shape (N, 7); got {output.shape}"
        )
    if output.shape[0] != expected_count:
        raise ValueError(
            f"validate_warp_output expected {expected_count} rows; "
            f"got {output.shape[0]}"
        )
    return output


def _replace_with_warp(
    batch: PhotonFieldBatch,
    new_directions: np.ndarray,
    new_brightness: np.ndarray,
    new_colors_rgb: np.ndarray,
) -> PhotonFieldBatch:
    """Return a new :class:`PhotonFieldBatch` with overridden warped fields."""
    return PhotonFieldBatch(
        object_ids=list(batch.object_ids),
        names=list(batch.names),
        object_types=list(batch.object_types),
        directions=np.asarray(new_directions, dtype=np.float64),
        distances_m=batch.distances_m.copy(),
        brightness=np.asarray(new_brightness, dtype=np.float64),
        colors_rgb=np.asarray(new_colors_rgb, dtype=np.float64),
        truth_levels=list(batch.truth_levels),
    )


def apply_batch_ai_warp(
    batch: PhotonFieldBatch,
    observer: ObserverState,
    model: "BatchONNXPhotonWarpModel | None",
) -> PhotonFieldBatch:
    """Apply a batch AI photon warp; fall back to deterministic on any failure.

    - ``model is None``: just runs the deterministic vectorized transform.
    - ``model`` raises during input prep, inference, or validation: also
      runs the deterministic transform.
    - Otherwise: replaces directions / brightness / colors with the model's
      output, leaving identity fields untouched.
    """
    if model is None:
        return transform_photon_field_batch(batch, observer)
    if len(batch) == 0:
        # Nothing to warp; preserve the empty batch verbatim.
        return _replace_with_warp(
            batch,
            np.zeros((0, 3), dtype=np.float64),
            np.zeros((0,), dtype=np.float64),
            np.zeros((0, 3), dtype=np.float64),
        )

    try:
        feed = build_photon_warp_input(batch, observer)
        raw = model.predict_batch(feed)
        validate_warp_output(raw, len(batch))
    except Exception:
        return transform_photon_field_batch(batch, observer)

    return _replace_with_warp(
        batch,
        raw[:, 0:3],
        raw[:, 3],
        raw[:, 4:7],
    )


# Late import to avoid a circular reference at module-load time.
from cosmic_engine.ai.onnx_photon_warp_batch import (  # noqa: E402
    BatchONNXPhotonWarpModel,
)
