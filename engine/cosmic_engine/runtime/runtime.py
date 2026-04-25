"""Central orchestrator for the headless engine."""

from __future__ import annotations

import logging
from pathlib import Path

from cosmic_engine.core.registry import UniverseRegistry
from cosmic_engine.core.time import SimulationClock
from cosmic_engine.core.universe_object import UniverseObject
from cosmic_engine.core.vector import Vector3
from cosmic_engine.runtime.config import RuntimeConfig
from cosmic_engine.runtime.scene_state import SceneState

_LOG = logging.getLogger(__name__)

_J2000_JD = 2_451_545.0


class CosmicRuntime:
    """Owns a :class:`UniverseRegistry`, a :class:`SimulationClock`, and a config.

    The runtime is deliberately thin: it knows how to schedule physics,
    counting, and active-object selection but delegates every actual
    transform to the existing modules.
    """

    def __init__(
        self,
        registry: UniverseRegistry | None = None,
        config: RuntimeConfig | None = None,
        clock: SimulationClock | None = None,
    ) -> None:
        self.registry = registry if registry is not None else UniverseRegistry()
        self.config = config if config is not None else RuntimeConfig()
        self.config.validate()
        self.clock = (
            clock
            if clock is not None
            else SimulationClock(current_julian_date=_J2000_JD)
        )
        self.last_scene_state: SceneState | None = None

    # --- ingestion --------------------------------------------------------

    def add_objects(self, objects: list[UniverseObject]) -> None:
        """Add objects to the registry, skipping IDs that already exist."""
        for obj in objects:
            if self.registry.get_object(obj.id) is None:
                self.registry.add_object(obj)

    def load_sample_data(self) -> None:
        """Best-effort load of the bundled sample CSVs and JPL placeholder."""
        # Resolve repo root from this file's path:
        #   .../engine/cosmic_engine/runtime/runtime.py
        repo_root = Path(__file__).resolve().parents[3]
        data_dir = repo_root / "data"

        # Lazy imports keep module-level surface area minimal.
        from cosmic_engine.data.desi_catalog import load_desi_catalog
        from cosmic_engine.data.gaia_catalog import load_gaia_like_catalog
        from cosmic_engine.data.jpl_ephemeris import (
            load_jpl_ephemeris_placeholder,
        )
        from cosmic_engine.data.sdss_catalog import load_sdss_like_catalog

        sources: list[tuple[str, Path, callable]] = [
            ("gaia", data_dir / "sample_gaia_like.csv", load_gaia_like_catalog),
            ("sdss", data_dir / "sample_sdss_like.csv", load_sdss_like_catalog),
            ("desi", data_dir / "sample_desi.csv", load_desi_catalog),
        ]

        for label, path, loader in sources:
            if not path.is_file():
                _LOG.info("load_sample_data: %s not found at %s", label, path)
                continue
            try:
                self.add_objects(loader(str(path)))
            except Exception as e:  # pragma: no cover - defensive
                _LOG.warning("load_sample_data: %s failed: %s", label, e)

        # JPL placeholder needs no file.
        try:
            self.add_objects(
                load_jpl_ephemeris_placeholder(self.clock.get_julian_date())
            )
        except Exception as e:  # pragma: no cover - defensive
            _LOG.warning("load_sample_data: jpl placeholder failed: %s", e)

    # --- selection --------------------------------------------------------

    def select_active_objects(
        self,
        observer_position: Vector3,
    ) -> list[UniverseObject]:
        """Return a deterministic, capped slice of registry objects.

        Sorted by ``id`` so successive frames see the same ordering.
        ``active_radius_m`` filters by distance from ``observer_position``;
        ``max_active_objects`` truncates the result.
        """
        objects = sorted(self.registry.list_objects(), key=lambda o: o.id)
        if self.config.active_radius_m is not None:
            r2 = self.config.active_radius_m ** 2
            ox = observer_position.x
            oy = observer_position.y
            oz = observer_position.z
            objects = [
                o
                for o in objects
                if (o.position_m.x - ox) ** 2
                + (o.position_m.y - oy) ** 2
                + (o.position_m.z - oz) ** 2
                <= r2
            ]
        return objects[: self.config.max_active_objects]

    # --- stepping ---------------------------------------------------------

    def step(self, delta_seconds: float) -> SceneState:
        """Advance the clock and (if configured) one physics step."""
        self.clock.tick(delta_seconds)
        notes: list[str] = [
            f"clock advanced by {delta_seconds} s",
        ]

        if self.config.enable_physics and self.config.physics_backend != "none":
            notes.extend(self._run_physics_step(delta_seconds))
        else:
            notes.append("physics: disabled")

        state = self.build_scene_state()
        state.notes = notes
        self.last_scene_state = state
        return state

    def _run_physics_step(self, delta_seconds: float) -> list[str]:
        """Run one physics step on all massive registry objects."""
        from cosmic_engine.physics.nbody import (
            NBodySimulator,
            apply_nbody_state_to_objects,
            objects_to_nbody_state,
        )

        massive = [
            o for o in self.registry.list_objects() if o.mass_kg is not None
        ]
        if len(massive) < 2:
            return ["physics: skipped (need >= 2 massive bodies)"]

        try:
            state = objects_to_nbody_state(massive)
        except ValueError as e:
            return [f"physics: skipped ({e})"]

        backend = self.config.physics_backend
        integrator = "leapfrog" if backend == "exact_nbody" else "barnes_hut"
        try:
            sim = NBodySimulator(state, integrator=integrator)
            sim.step(delta_seconds)
            apply_nbody_state_to_objects(massive, sim.get_state())
        except Exception as e:  # pragma: no cover - defensive
            return [f"physics: failed ({e})"]
        return [f"physics: {backend} stepped {len(massive)} bodies"]

    # --- introspection ----------------------------------------------------

    def build_scene_state(self) -> SceneState:
        """Build a :class:`SceneState` from the current registry."""
        objects = self.registry.list_objects()
        type_counts: dict[str, int] = {}
        truth_counts: dict[str, int] = {}
        source_counts: dict[str, int] = {}
        for obj in objects:
            t = obj.object_type.value
            type_counts[t] = type_counts.get(t, 0) + 1
            tl = obj.truth_level.value
            truth_counts[tl] = truth_counts.get(tl, 0) + 1
            src = obj.source if obj.source is not None else "unknown"
            source_counts[src] = source_counts.get(src, 0) + 1

        return SceneState(
            julian_date=self.clock.get_julian_date(),
            total_objects=len(objects),
            active_objects=min(len(objects), self.config.max_active_objects),
            object_type_counts=type_counts,
            truth_level_counts=truth_counts,
            source_counts=source_counts,
            physics_backend=self.config.physics_backend,
            perception_enabled=self.config.enable_perception,
            ai_warp_enabled=self.config.enable_ai_warp,
            notes=[],
        )
