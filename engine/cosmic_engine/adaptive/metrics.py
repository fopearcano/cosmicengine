"""Normalized error metrics for adaptive feedback."""

from __future__ import annotations

import numpy as np


# Floor used by every metric so an analytical value of 0 doesn't
# produce a divide-by-zero when normalising.
_EPS = 1.0e-30


def _to_array(x) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    return arr


def compute_acceleration_error(
    analytical: np.ndarray,
    predicted: np.ndarray,
) -> float:
    """Return ``||predicted − analytical|| / max(||analytical||, eps)``.

    Vector-valued. Both inputs must be the same shape; raises
    :class:`ValueError` otherwise.
    """
    a = _to_array(analytical)
    p = _to_array(predicted)
    if a.shape != p.shape:
        raise ValueError(
            f"shape mismatch: analytical={a.shape}, predicted={p.shape}"
        )
    diff_norm = float(np.linalg.norm(p - a))
    base_norm = float(np.linalg.norm(a))
    return diff_norm / max(base_norm, _EPS)


def compute_direction_error(
    analytical: np.ndarray,
    predicted: np.ndarray,
) -> float:
    """Return ``1 − cos(θ)`` between the two unit-direction inputs.

    Inputs are normalised internally so the caller doesn't have to
    pre-normalise. Result is in ``[0, 2]``: ``0`` for identical
    directions, ``1`` for orthogonal, ``2`` for antiparallel.
    """
    a = _to_array(analytical)
    p = _to_array(predicted)
    if a.shape != p.shape:
        raise ValueError(
            f"shape mismatch: analytical={a.shape}, predicted={p.shape}"
        )
    a_norm = float(np.linalg.norm(a))
    p_norm = float(np.linalg.norm(p))
    if a_norm < _EPS or p_norm < _EPS:
        # Either input is degenerate; report maximum error rather than
        # NaN so callers don't have to filter.
        return 2.0
    cos_theta = float(np.dot(a, p) / (a_norm * p_norm))
    cos_theta = max(-1.0, min(1.0, cos_theta))
    return 1.0 - cos_theta


def compute_brightness_error(
    analytical: float,
    predicted: float,
) -> float:
    """Return the relative brightness error.

    ``|predicted − analytical| / max(|analytical|, eps)``. Always
    non-negative.
    """
    a = float(analytical)
    p = float(predicted)
    return abs(p - a) / max(abs(a), _EPS)
