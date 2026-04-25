"""Headless frame pipeline.

A single function that runs the post-physics rendering preparation
chain — active-object selection, photon-field assembly, perception
transform, and optional PPM export. Designed to be called once per
frame in tests, demos, and offline batch jobs.
"""

from __future__ import annotations

from cosmic_engine.core.object_types import CosmicObjectType
from cosmic_engine.core.vector import Vector3
from cosmic_engine.perception.observer import ObserverState
from cosmic_engine.perception.transform import transform_photon_field
from cosmic_engine.rendering.image_export import render_photon_field_to_ppm
from cosmic_engine.rendering.photon_field import (
    PhotonSample,
    build_star_photon_field,
)
from cosmic_engine.rendering.simple_camera import SimpleCamera
from cosmic_engine.rendering.vectorized_galaxy_field import (
    GalaxyFieldBatch,
    build_galaxy_field_batch,
)
from cosmic_engine.runtime.runtime import CosmicRuntime
from cosmic_engine.runtime.scene_state import SceneState


_RENDERABLE_TYPES = {CosmicObjectType.STAR, CosmicObjectType.GALAXY}


def _galaxy_batch_to_samples(batch: GalaxyFieldBatch) -> list[PhotonSample]:
    samples: list[PhotonSample] = []
    for i in range(len(batch)):
        r, g, b = batch.colors_rgb[i]
        samples.append(
            PhotonSample(
                object_id=batch.object_ids[i],
                name=batch.names[i],
                object_type="galaxy",
                direction=Vector3(
                    float(batch.directions[i, 0]),
                    float(batch.directions[i, 1]),
                    float(batch.directions[i, 2]),
                ),
                distance_m=float(batch.distances_m[i]),
                apparent_brightness=float(batch.brightness[i]),
                color_rgb=(
                    max(0, min(255, int(round(float(r))))),
                    max(0, min(255, int(round(float(g))))),
                    max(0, min(255, int(round(float(b))))),
                ),
                truth_level=batch.truth_levels[i],
            )
        )
    return samples


def run_headless_frame(
    runtime: CosmicRuntime,
    camera: SimpleCamera,
    observer: ObserverState,
    output_path: str | None = None,
) -> SceneState:
    """Build a frame from the current runtime state and optionally render it.

    Always returns a :class:`SceneState` — the function never raises on
    an empty scene or on object types it cannot draw.
    """
    notes: list[str] = []

    active = runtime.select_active_objects(observer.position_m)

    stars = [o for o in active if o.object_type is CosmicObjectType.STAR]
    galaxies = [o for o in active if o.object_type is CosmicObjectType.GALAXY]
    skipped = [o for o in active if o.object_type not in _RENDERABLE_TYPES]
    if skipped:
        notes.append(
            f"renderer skipped {len(skipped)} non-renderable object(s)"
        )

    star_samples = build_star_photon_field(stars, camera) if stars else []

    galaxy_samples: list[PhotonSample] = []
    if galaxies:
        galaxy_batch = build_galaxy_field_batch(galaxies, camera)
        galaxy_samples = _galaxy_batch_to_samples(galaxy_batch)

    if runtime.config.enable_perception and star_samples:
        star_samples = transform_photon_field(star_samples, observer)
        notes.append(f"perception applied to {len(star_samples)} star samples")

    all_samples = star_samples + galaxy_samples

    if output_path is not None:
        render_photon_field_to_ppm(all_samples, camera, output_path)
        notes.append(
            f"rendered {len(all_samples)} samples to {output_path}"
        )
    else:
        notes.append("no output path; render skipped")

    state = runtime.build_scene_state()
    state.notes = list(state.notes) + [
        f"active: {len(active)}",
        f"stars: {len(stars)}",
        f"galaxies: {len(galaxies)}",
    ] + notes
    runtime.last_scene_state = state
    return state
