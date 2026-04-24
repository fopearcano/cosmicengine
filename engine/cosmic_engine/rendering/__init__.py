"""Minimal photon-field rendering layer.

Phase 3 contribution: a deliberately tiny pipeline that turns a list of
:class:`UniverseObject` stars into a 2D image.

- :class:`SimpleCamera` — pinhole-style camera pose + FOV.
- :class:`PhotonSample` — one direction/brightness/color contribution.
- :func:`build_star_photon_field` — gather star samples for a camera.
- :func:`render_photon_field_to_ppm` — rasterize to a plain PPM file.

No GPU, no AI, no relativistic effects, no third-party libraries.
"""

from cosmic_engine.rendering.image_export import render_photon_field_to_ppm
from cosmic_engine.rendering.photon_field import (
    PhotonSample,
    build_star_photon_field,
)
from cosmic_engine.rendering.simple_camera import SimpleCamera

__all__ = [
    "PhotonSample",
    "SimpleCamera",
    "build_star_photon_field",
    "render_photon_field_to_ppm",
]
