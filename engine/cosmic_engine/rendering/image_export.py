"""Write photon samples to a plain (P3) PPM image.

The projection is a pinhole: the camera's ``forward`` / ``up`` are
orthonormalized, a right axis is derived, and each sample direction is
split into forward / right / up components. Samples behind the camera
are dropped; the remaining ones are scaled by ``tan(fov/2)`` to pixel
coordinates.

Brightness is normalized against the brightest sample in the batch so
the image always has a usable dynamic range, however dim the scene.
"""

from __future__ import annotations

import math

from cosmic_engine.core.vector import Vector3
from cosmic_engine.rendering.photon_field import PhotonSample
from cosmic_engine.rendering.simple_camera import SimpleCamera


def _normalize(v: Vector3) -> Vector3:
    n = math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z)
    if n == 0.0:
        return v
    return Vector3(v.x / n, v.y / n, v.z / n)


def _dot(a: Vector3, b: Vector3) -> float:
    return a.x * b.x + a.y * b.y + a.z * b.z


def _cross(a: Vector3, b: Vector3) -> Vector3:
    return Vector3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x,
    )


def _camera_basis(camera: SimpleCamera) -> tuple[Vector3, Vector3, Vector3]:
    """Return (forward, right, up) orthonormal axes."""
    forward = _normalize(camera.forward)
    right = _normalize(_cross(forward, camera.up))
    up = _cross(right, forward)
    return forward, right, up


def _project(
    direction: Vector3,
    camera: SimpleCamera,
    forward: Vector3,
    right: Vector3,
    up: Vector3,
) -> tuple[int, int] | None:
    """Project a unit direction to pixel coordinates, or ``None`` if off-screen."""
    f = _dot(direction, forward)
    if f <= 0.0:
        return None
    r = _dot(direction, right)
    u = _dot(direction, up)

    half_fov = math.radians(camera.fov_degrees) / 2.0
    scale = math.tan(half_fov)
    aspect = camera.image_height / camera.image_width

    nx = (r / f) / scale
    ny = (u / f) / (scale * aspect)
    if abs(nx) > 1.0 or abs(ny) > 1.0:
        return None

    px = int((nx + 1.0) * 0.5 * camera.image_width)
    py = int((1.0 - ny) * 0.5 * camera.image_height)
    px = max(0, min(camera.image_width - 1, px))
    py = max(0, min(camera.image_height - 1, py))
    return px, py


def _plot(
    pixels: list[list[tuple[int, int, int]]],
    px: int,
    py: int,
    color: tuple[int, int, int],
    width: int,
    height: int,
    bright: bool,
) -> None:
    pixels[py][px] = color
    if not bright:
        return
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            x = px + dx
            y = py + dy
            if 0 <= x < width and 0 <= y < height:
                pixels[y][x] = color


def render_photon_field_to_ppm(
    samples: list[PhotonSample],
    camera: SimpleCamera,
    output_path: str,
) -> None:
    """Rasterize ``samples`` into a plain (P3) PPM at ``output_path``.

    An empty sample list still produces a valid black image.
    """
    camera.validate()
    width = camera.image_width
    height = camera.image_height
    pixels: list[list[tuple[int, int, int]]] = [
        [(0, 0, 0) for _ in range(width)] for _ in range(height)
    ]

    if samples:
        max_brightness = max(s.apparent_brightness for s in samples)
        if max_brightness <= 0.0:
            max_brightness = 1.0
        forward, right, up = _camera_basis(camera)
        for sample in samples:
            coords = _project(sample.direction, camera, forward, right, up)
            if coords is None:
                continue
            intensity = sample.apparent_brightness / max_brightness
            intensity = max(0.0, min(1.0, intensity))
            r, g, b = sample.color_rgb
            scaled = (
                int(r * intensity),
                int(g * intensity),
                int(b * intensity),
            )
            _plot(
                pixels,
                coords[0],
                coords[1],
                scaled,
                width,
                height,
                bright=intensity > 0.5,
            )

    with open(output_path, "w", encoding="ascii") as fh:
        fh.write("P3\n")
        fh.write(f"{width} {height}\n")
        fh.write("255\n")
        for row in pixels:
            fh.write(" ".join(f"{r} {g} {b}" for r, g, b in row))
            fh.write("\n")
