"""Minimal photon-field rendering layer.

Phase 3 contribution: a deliberately tiny pipeline that turns a list of
:class:`UniverseObject` stars into a 2D image.

- :class:`SimpleCamera` — pinhole-style camera pose + FOV.
- :class:`PhotonSample` — one direction/brightness/color contribution.
- :func:`build_star_photon_field` — gather star samples for a camera.
- :func:`render_photon_field_to_ppm` — rasterize to a plain PPM file.

No GPU, no AI, no relativistic effects, no third-party libraries.
"""

from cosmic_engine.rendering.density_field import galaxy_batch_to_density_grid
from cosmic_engine.rendering.image_export import (
    render_galaxy_batch_to_ppm,
    render_photon_batch_to_ppm,
    render_photon_field_to_ppm,
)
from cosmic_engine.rendering.photon_field import (
    PhotonSample,
    build_star_photon_field,
)
from cosmic_engine.rendering.simple_camera import SimpleCamera
from cosmic_engine.rendering.vectorized_galaxy_field import (
    GalaxyFieldBatch,
    build_galaxy_field_batch,
)
from cosmic_engine.rendering.vectorized_photon_field import (
    PhotonFieldBatch,
    build_star_photon_field_batch,
    photon_batch_to_samples,
)

__all__ = [
    "GalaxyFieldBatch",
    "PhotonFieldBatch",
    "PhotonSample",
    "SimpleCamera",
    "build_galaxy_field_batch",
    "build_star_photon_field",
    "build_star_photon_field_batch",
    "galaxy_batch_to_density_grid",
    "photon_batch_to_samples",
    "render_galaxy_batch_to_ppm",
    "render_photon_batch_to_ppm",
    "render_photon_field_to_ppm",
]
