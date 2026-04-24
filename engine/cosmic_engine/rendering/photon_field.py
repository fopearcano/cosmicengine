"""Turn catalog stars into per-sample photon contributions.

The "photon field" here is an intentional oversimplification: one sample
per visible star, carrying direction, distance, a magnitude-derived
brightness, and a spectral-class-derived color. Anything real
(wavelengths, flux integration, PSFs, relativistic effects) is deferred
to later phases.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from cosmic_engine.core.object_types import CosmicObjectType
from cosmic_engine.core.universe_object import UniverseObject
from cosmic_engine.core.vector import Vector3
from cosmic_engine.rendering.simple_camera import SimpleCamera


# First-letter spectral-class → approximate RGB tint.
_SPECTRAL_COLORS: dict[str, tuple[int, int, int]] = {
    "O": (150, 180, 255),
    "B": (170, 200, 255),
    "A": (240, 240, 255),
    "F": (255, 250, 220),
    "G": (255, 240, 180),
    "K": (255, 200, 140),
    "M": (255, 140, 100),
}
_DEFAULT_COLOR: tuple[int, int, int] = (255, 255, 255)


@dataclass
class PhotonSample:
    """A single star's contribution to the photon field."""

    object_id: str
    name: str
    object_type: str
    direction: Vector3
    distance_m: float
    apparent_brightness: float
    color_rgb: tuple[int, int, int]
    truth_level: str


def _color_for_spectral_class(spectral_class: str | None) -> tuple[int, int, int]:
    """Pick an RGB tint from the first character of a spectral-class string."""
    if not spectral_class:
        return _DEFAULT_COLOR
    return _SPECTRAL_COLORS.get(spectral_class[0].upper(), _DEFAULT_COLOR)


def build_star_photon_field(
    objects: list[UniverseObject],
    camera: SimpleCamera,
) -> list[PhotonSample]:
    """Build photon samples for every star visible to ``camera``.

    Non-star objects are ignored. Stars coincident with the camera
    (zero-length direction vector) are skipped.
    """
    samples: list[PhotonSample] = []
    for obj in objects:
        if obj.object_type is not CosmicObjectType.STAR:
            continue

        dx = obj.position_m.x - camera.position_m.x
        dy = obj.position_m.y - camera.position_m.y
        dz = obj.position_m.z - camera.position_m.z
        distance = math.sqrt(dx * dx + dy * dy + dz * dz)
        if distance == 0.0:
            continue
        direction = Vector3(dx / distance, dy / distance, dz / distance)

        magnitude = obj.metadata.get("apparent_magnitude")
        brightness = (
            10.0 ** (-0.4 * float(magnitude)) if magnitude is not None else 1.0
        )

        samples.append(
            PhotonSample(
                object_id=obj.id,
                name=obj.name,
                object_type=obj.object_type.value,
                direction=direction,
                distance_m=distance,
                apparent_brightness=brightness,
                color_rgb=_color_for_spectral_class(obj.spectral_class),
                truth_level=obj.truth_level.value,
            )
        )
    return samples
