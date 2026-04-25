"""Photon-space neural warp viewer.

Drives a frame end-to-end on the local side: select active objects,
build the photon field, apply the (optional) AI warp model, and
render the resulting samples to a PPM. Unlike :class:`AIViewer`,
this one does not consume the runtime server — it owns the runtime
directly so the warp can happen *before* projection.

If ``warp_model`` is ``None``, the deterministic perception transform
runs unchanged.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING

from cosmic_engine.ai.base import AIWarpModel
from cosmic_engine.ai.onnx_photon_warp_batch import BatchONNXPhotonWarpModel
from cosmic_engine.core.object_types import CosmicObjectType
from cosmic_engine.core.vector import Vector3
from cosmic_engine.perception.observer import ObserverState
from cosmic_engine.perception.transform import transform_photon_field
from cosmic_engine.perception.vectorized_ai_transform import apply_batch_ai_warp
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
from cosmic_engine.rendering.vectorized_photon_field import (
    build_star_photon_field_batch,
    photon_batch_to_samples,
)

if TYPE_CHECKING:
    from cosmic_engine.runtime.runtime import CosmicRuntime


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


class NeuralWarpViewer:
    """Drive frames locally with optional photon-space AI warp."""

    def __init__(
        self,
        runtime: "CosmicRuntime",
        camera: SimpleCamera,
        observer: ObserverState,
        warp_model: AIWarpModel | None = None,
        batch_warp_model: BatchONNXPhotonWarpModel | None = None,
    ) -> None:
        self.runtime = runtime
        self.camera = camera
        self.observer = observer
        self.warp_model = warp_model
        self.batch_warp_model = batch_warp_model
        self.frames_rendered: int = 0
        self.last_frame_path: str | None = None
        self.last_frame_time_seconds: float = 0.0
        self.last_batch_inference_seconds: float = 0.0
        self.last_photon_count: int = 0
        self.last_used_fallback: bool = False

    @property
    def mode(self) -> str:
        if self.batch_warp_model is not None:
            return "batch_neural"
        if self.warp_model is not None:
            return "neural"
        return "deterministic"

    def confidence(self) -> float:
        if self.batch_warp_model is not None:
            return float(self.batch_warp_model.confidence())
        if self.warp_model is None:
            return 1.0
        return float(self.warp_model.confidence())

    def _gather_samples(self) -> list[PhotonSample]:
        active = self.runtime.select_active_objects(self.observer.position_m)
        stars = [o for o in active if o.object_type is CosmicObjectType.STAR]
        galaxies = [o for o in active if o.object_type is CosmicObjectType.GALAXY]
        star_samples = (
            build_star_photon_field(stars, self.camera) if stars else []
        )
        galaxy_samples: list[PhotonSample] = []
        if galaxies:
            galaxy_batch = build_galaxy_field_batch(galaxies, self.camera)
            galaxy_samples = _galaxy_batch_to_samples(galaxy_batch)
        return star_samples + galaxy_samples

    def run_once(
        self,
        output_path: str | None = None,
    ) -> list[PhotonSample]:
        """Build, warp, and (optionally) render one frame."""
        if self.batch_warp_model is not None:
            return self._run_once_batch(output_path)
        return self._run_once_scalar(output_path)

    def _run_once_scalar(
        self,
        output_path: str | None,
    ) -> list[PhotonSample]:
        all_samples = self._gather_samples()
        start = time.perf_counter()
        warped = transform_photon_field(
            all_samples, self.observer, ai_model=self.warp_model
        )
        self.last_frame_time_seconds = time.perf_counter() - start
        self.last_photon_count = len(warped)
        self.last_batch_inference_seconds = 0.0
        self.last_used_fallback = False

        if output_path is not None:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            render_photon_field_to_ppm(warped, self.camera, output_path)
            self.last_frame_path = output_path
        self.frames_rendered += 1

        print(
            f"warp mode: {self.mode:<13}  "
            f"confidence: {self.confidence():.2f}  "
            f"photons: {len(warped):>5}  "
            f"frame_time: {self.last_frame_time_seconds * 1000:7.2f} ms"
        )
        return warped

    def _run_once_batch(
        self,
        output_path: str | None,
    ) -> list[PhotonSample]:
        active = self.runtime.select_active_objects(self.observer.position_m)
        stars = [o for o in active if o.object_type is CosmicObjectType.STAR]
        galaxies = [o for o in active if o.object_type is CosmicObjectType.GALAXY]

        # AI batch path: stars only (vectorized). Galaxies stay deterministic.
        star_batch = build_star_photon_field_batch(stars, self.camera) if stars else None

        warped_star_samples: list[PhotonSample] = []
        batch_seconds = 0.0
        if star_batch is not None and len(star_batch) > 0:
            error_before = self.batch_warp_model.last_error
            t0 = time.perf_counter()
            warped_batch = apply_batch_ai_warp(
                star_batch, self.observer, self.batch_warp_model
            )
            batch_seconds = time.perf_counter() - t0
            warped_star_samples = photon_batch_to_samples(warped_batch)
            # apply_batch_ai_warp swallows model exceptions; if last_error
            # got newly set during this call, the path fell back.
            self.last_used_fallback = (
                self.batch_warp_model.last_error is not None
                and self.batch_warp_model.last_error != error_before
            )
        else:
            self.last_used_fallback = False

        # Galaxies: existing deterministic scalar path (keeps Phase 21 behavior).
        galaxy_samples: list[PhotonSample] = []
        if galaxies:
            galaxy_batch = build_galaxy_field_batch(galaxies, self.camera)
            galaxy_samples = _galaxy_batch_to_samples(galaxy_batch)
            galaxy_samples = transform_photon_field(
                galaxy_samples, self.observer, ai_model=None
            )

        all_samples = warped_star_samples + galaxy_samples
        self.last_batch_inference_seconds = batch_seconds
        self.last_frame_time_seconds = batch_seconds
        self.last_photon_count = len(all_samples)

        if output_path is not None:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            render_photon_field_to_ppm(all_samples, self.camera, output_path)
            self.last_frame_path = output_path
        self.frames_rendered += 1

        print(
            f"warp mode: {self.mode:<13}  "
            f"confidence: {self.confidence():.2f}  "
            f"photons: {len(all_samples):>6}  "
            f"batch_time: {batch_seconds * 1000:8.2f} ms  "
            f"fallback: {'yes' if self.last_used_fallback else 'no'}"
        )
        return all_samples

    def run_loop(
        self,
        max_frames: int | None = None,
        output_pattern: str | None = None,
    ) -> int:
        """Run ``max_frames`` iterations. ``output_pattern`` may contain ``{n}``."""
        rendered = 0
        while max_frames is None or rendered < max_frames:
            output = (
                output_pattern.format(n=rendered)
                if output_pattern is not None
                else None
            )
            self.run_once(output)
            rendered += 1
        return rendered
