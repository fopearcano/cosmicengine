"""Coordinate conversions between spherical (RA/Dec) and Cartesian.

Conventions
-----------
- Right Ascension (RA) is in degrees, measured eastward from the vernal
  equinox, normalized to the half-open interval ``[0, 360)``.
- Declination (Dec) is in degrees in the closed interval ``[-90, +90]``.
- Cartesian axes are equatorial:
    x points toward RA=0, Dec=0
    y points toward RA=90°, Dec=0
    z points toward Dec=+90° (the north celestial pole)
- Distance is in meters.

Conversions are pure rotations of a radial vector; no relativistic or
parallax corrections are applied.
"""

from __future__ import annotations

import math

from cosmic_engine.core.vector import Vector3


def distance(a: Vector3, b: Vector3) -> float:
    """Return the Euclidean distance between two Cartesian points (meters)."""
    dx = a.x - b.x
    dy = a.y - b.y
    dz = a.z - b.z
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def ra_dec_distance_to_cartesian(
    ra_deg: float,
    dec_deg: float,
    distance_m: float,
) -> Vector3:
    """Convert equatorial spherical coordinates to a Cartesian vector.

    ``ra_deg`` and ``dec_deg`` are in degrees, ``distance_m`` in meters.
    """
    ra_rad = math.radians(ra_deg)
    dec_rad = math.radians(dec_deg)
    cos_dec = math.cos(dec_rad)
    return Vector3(
        distance_m * cos_dec * math.cos(ra_rad),
        distance_m * cos_dec * math.sin(ra_rad),
        distance_m * math.sin(dec_rad),
    )


def cartesian_to_ra_dec_distance(vec: Vector3) -> tuple[float, float, float]:
    """Convert a Cartesian vector to ``(ra_deg, dec_deg, distance_m)``.

    The returned RA is normalized to ``[0, 360)``. For a zero vector the
    RA and Dec are both zero and distance is zero.
    """
    d = math.sqrt(vec.x * vec.x + vec.y * vec.y + vec.z * vec.z)
    if d == 0.0:
        return 0.0, 0.0, 0.0
    dec_deg = math.degrees(math.asin(vec.z / d))
    ra_deg = math.degrees(math.atan2(vec.y, vec.x)) % 360.0
    return ra_deg, dec_deg, d
