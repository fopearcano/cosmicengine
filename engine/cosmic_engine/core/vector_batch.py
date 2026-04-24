"""NumPy helpers for converting between Python :class:`Vector3` lists
and batched ``(N, 3)`` arrays.

These are the connective tissue between the scalar core and the
vectorized rendering / perception paths. Every helper either produces
or validates a shape-``(N, 3)`` array.
"""

from __future__ import annotations

import numpy as np

from cosmic_engine.core.vector import Vector3


def _ensure_shape_n3(array: np.ndarray) -> None:
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError(
            f"expected shape (N, 3); got shape {array.shape}"
        )


def vectors_to_array(vectors: list[Vector3]) -> np.ndarray:
    """Pack a list of :class:`Vector3` into a ``(N, 3)`` float64 array."""
    if not vectors:
        return np.zeros((0, 3), dtype=np.float64)
    arr = np.empty((len(vectors), 3), dtype=np.float64)
    for i, v in enumerate(vectors):
        arr[i, 0] = v.x
        arr[i, 1] = v.y
        arr[i, 2] = v.z
    return arr


def array_to_vectors(array: np.ndarray) -> list[Vector3]:
    """Unpack a ``(N, 3)`` array into a list of :class:`Vector3`."""
    _ensure_shape_n3(array)
    return [Vector3(float(r[0]), float(r[1]), float(r[2])) for r in array]


def normalize_vectors(array: np.ndarray) -> np.ndarray:
    """Return a ``(N, 3)`` array whose rows are unit vectors.

    Zero-length rows are returned unchanged (still zero) rather than
    producing NaNs.
    """
    _ensure_shape_n3(array)
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    safe = np.where(norms == 0.0, 1.0, norms)
    return np.where(norms == 0.0, array, array / safe)


def distances_from_origin(array: np.ndarray) -> np.ndarray:
    """Return a ``(N,)`` array of per-row Euclidean norms."""
    _ensure_shape_n3(array)
    return np.linalg.norm(array, axis=1)


def clamp_array(
    array: np.ndarray,
    min_value: float,
    max_value: float,
) -> np.ndarray:
    """Return ``array`` clipped element-wise to ``[min_value, max_value]``."""
    return np.clip(array, min_value, max_value)
