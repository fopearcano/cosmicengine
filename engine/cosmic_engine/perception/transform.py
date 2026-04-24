"""Per-photon perception transforms driven by an :class:`ObserverState`.

The deterministic path is always available. An optional
:class:`~cosmic_engine.ai.base.AIWarpModel` can replace any of the three
per-sample transforms; passing ``ai_model=None`` falls back to the
deterministic logic.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from cosmic_engine.core.vector import Vector3
from cosmic_engine.perception.observer import ObserverState
from cosmic_engine.rendering.photon_field import PhotonSample

if TYPE_CHECKING:
    from cosmic_engine.ai.base import AIWarpModel


# Ceilings that keep extreme warp_factor values from producing infinities.
_BRIGHTNESS_CEILING = 1.0e18
_FACTOR_CEILING = 1.0e12


def _normalize(v: Vector3) -> Vector3:
    n = math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z)
    if n == 0.0:
        return v
    return Vector3(v.x / n, v.y / n, v.z / n)


def _dot(a: Vector3, b: Vector3) -> float:
    return a.x * b.x + a.y * b.y + a.z * b.z


def apply_direction_warp(
    direction: Vector3,
    observer: ObserverState,
) -> Vector3:
    """Return a unit direction pulled toward the observer's forward axis.

    Uses the observer's forward as a compression axis; the pull is
    proportional to ``beta * dot(direction, forward) * warp_factor``.
    Aberration-like in spirit, not a full Lorentz boost.
    """
    forward = _normalize(observer.forward)
    alignment = _dot(direction, forward)
    amount = observer.beta() * alignment * observer.warp_factor
    warped = Vector3(
        direction.x + forward.x * amount,
        direction.y + forward.y * amount,
        direction.z + forward.z * amount,
    )
    return _normalize(warped)


def apply_brightness_warp(
    brightness: float,
    direction: Vector3,
    observer: ObserverState,
) -> float:
    """Return a beaming-adjusted brightness (forward-aligned brighter)."""
    forward = _normalize(observer.forward)
    alignment = _dot(direction, forward)
    factor = max(0.0, 1.0 + observer.beta() * alignment)
    try:
        factor = factor ** observer.warp_factor
    except OverflowError:
        factor = _FACTOR_CEILING
    factor = min(factor, _FACTOR_CEILING)
    return min(max(brightness * factor, 0.0), _BRIGHTNESS_CEILING)


def apply_color_warp(
    color_rgb: tuple[int, int, int],
    direction: Vector3,
    observer: ObserverState,
) -> tuple[int, int, int]:
    """Return an RGB tint shifted blue (forward) or red (backward).

    The shift magnitude is ``beta * alignment * warp_factor`` clamped
    into ``[-1, 1]``. Intentionally approximate; not wavelength-correct.
    """
    forward = _normalize(observer.forward)
    alignment = _dot(direction, forward)
    shift = observer.beta() * alignment * observer.warp_factor
    shift = max(-1.0, min(1.0, shift))

    r, g, b = color_rgb
    if shift > 0.0:
        b = int(round(b + (255 - b) * shift))
        r = int(round(r * (1.0 - shift)))
    elif shift < 0.0:
        s = -shift
        r = int(round(r + (255 - r) * s))
        b = int(round(b * (1.0 - s)))

    r = max(0, min(255, r))
    g = max(0, min(255, g))
    b = max(0, min(255, b))
    return r, g, b


def transform_photon_sample(
    sample: PhotonSample,
    observer: ObserverState,
    ai_model: "AIWarpModel | None" = None,
) -> PhotonSample:
    """Return a new :class:`PhotonSample` with perception effects applied.

    If ``ai_model`` is ``None`` the deterministic transforms are used.
    Otherwise the model's ``predict_*`` methods replace each per-sample
    step. The original sample direction is passed to every model call
    so model predictions are not coupled to each other's outputs.
    """
    if ai_model is None:
        new_direction = apply_direction_warp(sample.direction, observer)
        new_brightness = apply_brightness_warp(
            sample.apparent_brightness, sample.direction, observer
        )
        new_color = apply_color_warp(
            sample.color_rgb, sample.direction, observer
        )
    else:
        # Each AI call is guarded so a partial model failure still produces
        # a valid sample via the deterministic fallback for that field.
        try:
            new_direction = ai_model.predict_direction(
                sample.direction, observer
            )
        except Exception:
            new_direction = apply_direction_warp(sample.direction, observer)
        try:
            new_brightness = ai_model.predict_brightness(
                sample.apparent_brightness, sample.direction, observer
            )
        except Exception:
            new_brightness = apply_brightness_warp(
                sample.apparent_brightness, sample.direction, observer
            )
        try:
            new_color = ai_model.predict_color(
                sample.color_rgb, sample.direction, observer
            )
        except Exception:
            new_color = apply_color_warp(
                sample.color_rgb, sample.direction, observer
            )
    return PhotonSample(
        object_id=sample.object_id,
        name=sample.name,
        object_type=sample.object_type,
        direction=new_direction,
        distance_m=sample.distance_m,
        apparent_brightness=new_brightness,
        color_rgb=new_color,
        truth_level=sample.truth_level,
    )


def transform_photon_field(
    samples: list[PhotonSample],
    observer: ObserverState,
    ai_model: "AIWarpModel | None" = None,
) -> list[PhotonSample]:
    """Apply :func:`transform_photon_sample` to every sample in the list."""
    return [transform_photon_sample(s, observer, ai_model) for s in samples]
