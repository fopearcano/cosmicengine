"""Central orchestrator for the headless engine."""

from __future__ import annotations

import logging
import math
from pathlib import Path

from cosmic_engine.core.registry import UniverseRegistry
from cosmic_engine.core.time import SimulationClock
from cosmic_engine.core.universe_object import UniverseObject
from cosmic_engine.core.vector import Vector3
from cosmic_engine.observer.observer_manager import ObserverManager
from cosmic_engine.runtime.config import RuntimeConfig
from cosmic_engine.runtime.scene_state import SceneState
from cosmic_engine.time.event_store import EventStore
from cosmic_engine.streaming.lod import select_lod_objects
from cosmic_engine.streaming.spatial_index import SpatialIndex
from cosmic_engine.streaming.tile_store import (
    TileStore,
    cell_index_for_position,
    partition_objects_into_tiles,
    tile_id_for_cell,
)

_LOG = logging.getLogger(__name__)

_J2000_JD = 2_451_545.0
_DEFAULT_TILE_CACHE_SIZE = 16


class _TileCache:
    """Tiny LRU cache for tile contents (FIFO-ish on equal access)."""

    def __init__(self, max_size: int = _DEFAULT_TILE_CACHE_SIZE) -> None:
        if max_size <= 0:
            raise ValueError("max_size must be positive")
        self.max_size = max_size
        self._store: dict[str, list[UniverseObject]] = {}
        self._order: list[str] = []

    def __len__(self) -> int:
        return len(self._store)

    def get(self, tile_id: str) -> list[UniverseObject] | None:
        if tile_id not in self._store:
            return None
        self._order.remove(tile_id)
        self._order.append(tile_id)
        return self._store[tile_id]

    def put(self, tile_id: str, objects: list[UniverseObject]) -> None:
        if tile_id in self._store:
            self._order.remove(tile_id)
        elif len(self._store) >= self.max_size:
            evicted = self._order.pop(0)
            self._store.pop(evicted, None)
        self._store[tile_id] = objects
        self._order.append(tile_id)

    def clear(self) -> None:
        self._store.clear()
        self._order.clear()


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
        # Streaming / LOD state (Phase 23). All optional; ``None`` means
        # the runtime behaves as before.
        self.spatial_index: SpatialIndex | None = None
        self.tile_store: TileStore | None = None
        self._tile_cell_size_m: float | None = None
        self._tile_cache: _TileCache | None = None
        self.last_loaded_tiles: list[str] = []
        # Multi-scale state (Phase 32). All optional; leave ``None`` to
        # keep the runtime behaving exactly as before.
        self.scale_manager = None
        self.multiscale_blend_width: float = 0.1
        # Observer registry (Phase 33). Empty by default — single-observer
        # callers never need to touch it.
        self.observer_manager: ObserverManager = ObserverManager()
        # Event log + global coordinate time (Phase 34). The event store
        # is shared truth; each observer filters it through its own past
        # light cone in get_visible_events().
        self.event_store: EventStore = EventStore()
        self.coordinate_time_t: float = 0.0
        # Reality rule engine (Phase 35). ``None`` means "scientific
        # default" — no rules applied, behavior identical to before.
        self.reality_rule_engine = None

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

        With streaming or a spatial index attached, candidates come from
        the relevant tiles / cells around ``observer_position`` instead
        of the entire registry. The :func:`select_lod_objects`
        probabilistic downsampler caps the result at
        ``config.max_active_objects`` while preserving the spatial
        distribution.
        """
        candidates = self._gather_candidates(observer_position)
        candidates = sorted(candidates, key=lambda o: o.id)
        return select_lod_objects(
            candidates,
            observer_position,
            self.config.max_active_objects,
        )

    def _gather_candidates(
        self,
        observer_position: Vector3,
    ) -> list[UniverseObject]:
        """Choose between in-memory / spatial-index / tile-store sources."""
        radius = self.config.active_radius_m

        if self.spatial_index is not None and radius is not None:
            return self.spatial_index.query_radius(observer_position, radius)

        if (
            self.tile_store is not None
            and self._tile_cell_size_m is not None
            and radius is not None
        ):
            return self._gather_tiles_around(observer_position, radius)

        # Fallback: filter the in-memory registry.
        objects = self.registry.list_objects()
        if radius is None:
            return objects
        r2 = radius * radius
        ox = observer_position.x
        oy = observer_position.y
        oz = observer_position.z
        return [
            o
            for o in objects
            if (o.position_m.x - ox) ** 2
            + (o.position_m.y - oy) ** 2
            + (o.position_m.z - oz) ** 2
            <= r2
        ]

    def _gather_tiles_around(
        self,
        observer_position: Vector3,
        radius: float,
    ) -> list[UniverseObject]:
        """Load tiles around the observer through the LRU cache and filter."""
        if self.tile_store is None or self._tile_cell_size_m is None:
            return []
        cache = self._tile_cache
        cell_size = self._tile_cell_size_m
        cx, cy, cz = cell_index_for_position(
            observer_position.x,
            observer_position.y,
            observer_position.z,
            cell_size,
        )
        cell_radius = int(math.ceil(radius / cell_size))
        loaded_tiles: list[str] = []
        candidates: list[UniverseObject] = []
        r2 = radius * radius
        ox, oy, oz = observer_position.x, observer_position.y, observer_position.z
        for dx in range(-cell_radius, cell_radius + 1):
            for dy in range(-cell_radius, cell_radius + 1):
                for dz in range(-cell_radius, cell_radius + 1):
                    tile_id = tile_id_for_cell(cx + dx, cy + dy, cz + dz)
                    objects = cache.get(tile_id) if cache is not None else None
                    if objects is None:
                        if not self.tile_store.has_tile(tile_id):
                            continue
                        objects = self.tile_store.load_tile(tile_id)
                        if cache is not None:
                            cache.put(tile_id, objects)
                    loaded_tiles.append(tile_id)
                    for obj in objects:
                        dxp = obj.position_m.x - ox
                        dyp = obj.position_m.y - oy
                        dzp = obj.position_m.z - oz
                        if dxp * dxp + dyp * dyp + dzp * dzp <= r2:
                            candidates.append(obj)
        self.last_loaded_tiles = loaded_tiles
        return candidates

    # --- multi-scale -----------------------------------------------------

    def enable_multiscale(self, scale_manager, blend_width: float = 0.1) -> None:
        """Attach a :class:`ScaleManager` and configure the blend width."""
        if blend_width < 0.0:
            raise ValueError("blend_width must be non-negative")
        self.scale_manager = scale_manager
        self.multiscale_blend_width = float(blend_width)

    def get_multiscale_scene(self, observer_position: Vector3) -> dict:
        """Return the right zone's representation for ``observer_position``.

        The "scale" is the distance from the observer to the nearest
        active object — i.e. what scale of structure is closest. When
        that distance lands near the upper or lower edge of the chosen
        zone, the result is blended with the adjacent zone using
        :func:`cosmic_engine.multiscale.transition.blend_representations`.
        """
        if self.scale_manager is None:
            raise RuntimeError(
                "CosmicRuntime.get_multiscale_scene requires "
                "enable_multiscale() to be called first"
            )
        # Lazy import to keep runtime free of multiscale unless used.
        from cosmic_engine.multiscale.representation import (
            get_representation_for_zone,
        )
        from cosmic_engine.multiscale.transition import (
            blend_representations,
            compute_transition_alpha,
        )

        active = self.select_active_objects(observer_position)
        if not active:
            zone = self.scale_manager.zones[0]
            return get_representation_for_zone(zone, self)

        ox, oy, oz = (
            observer_position.x,
            observer_position.y,
            observer_position.z,
        )
        nearest_distance = min(
            math.sqrt(
                (o.position_m.x - ox) ** 2
                + (o.position_m.y - oy) ** 2
                + (o.position_m.z - oz) ** 2
            )
            for o in active
        )

        zone = self.scale_manager.get_zone(nearest_distance)
        primary = get_representation_for_zone(zone, self)

        # Try to blend with the neighbor on whichever side the observer
        # is closer to. This keeps the two-zone blend naturally one-sided.
        neighbor_above = self.scale_manager.get_neighbor_above(zone)
        neighbor_below = self.scale_manager.get_neighbor_below(zone)
        midpoint = zone.min_scale_m + 0.5 * (
            zone.max_scale_m - zone.min_scale_m
        )
        if neighbor_above is not None and nearest_distance > midpoint:
            alpha = compute_transition_alpha(
                nearest_distance,
                zone,
                neighbor_above,
                blend_width=self.multiscale_blend_width,
            )
            if alpha > 0.0:
                secondary = get_representation_for_zone(neighbor_above, self)
                return blend_representations(primary, secondary, alpha)
        if neighbor_below is not None and nearest_distance < midpoint:
            # Treat the lower neighbor as "zone_a" so the alpha ramp
            # measures how far we still are inside ``zone``.
            alpha = 1.0 - compute_transition_alpha(
                nearest_distance,
                neighbor_below,
                zone,
                blend_width=self.multiscale_blend_width,
            )
            if alpha > 0.0:
                secondary = get_representation_for_zone(neighbor_below, self)
                return blend_representations(primary, secondary, alpha)

        return primary

    # --- multi-observer ---------------------------------------------------

    def render_for_observer(
        self,
        observer_id: str,
        camera,
    ) -> "RealityView":
        """Build a :class:`RealityView` for a single observer.

        Steps:

        1. Resolve the observer.
        2. Select active objects from the observer's position (re-uses
           streaming + LOD).
        3. If a :class:`ScaleManager` is attached, pick a multiscale
           representation; otherwise return the active objects flat.
        4. Apply this observer's spacetime / AI warp models if set
           (best-effort; failures fall back to the deterministic path
           and are reported in the metadata).
        5. Render a PPM-style frame using the photon path when stars
           are present; otherwise return ``frame_data=None``.
        6. Return the assembled :class:`RealityView`.
        """
        from cosmic_engine.observer.reality_view import RealityView
        from cosmic_engine.perception.observer import ObserverState
        from cosmic_engine.perception.transform import transform_photon_field
        from cosmic_engine.rendering import (
            build_star_photon_field,
            render_photon_field_to_ppm,
        )

        observer = self.observer_manager.get_observer(observer_id)
        observer.validate()

        active = self.select_active_objects(observer.position_m)

        if self.scale_manager is not None:
            rep = self.get_multiscale_scene(observer.position_m)
            rep_objects = rep.get("objects", []) or []
            rep_type = rep.get("type", "flat")
            zone_name = rep.get("zone_name") or rep.get("primary_zone")
        else:
            rep_objects = active
            rep_type = "flat"
            zone_name = None

        # Phase 35: evaluate the reality rule engine into a fresh
        # context. The result is used for this render only — the
        # observer's stored warp_factor / state is never mutated.
        rule_context = self._build_rule_context(observer, zone_name)
        rule_context = self._evaluate_reality_rules(rule_context)
        effective_warp = float(rule_context.warp_factor)

        notes: list[str] = []
        spacetime_label = "none"
        if observer.spacetime_model is not None:
            try:
                _ = observer.spacetime_model.confidence()
                spacetime_label = type(observer.spacetime_model).__name__
            except Exception as e:  # pragma: no cover - defensive
                notes.append(
                    f"spacetime_model fallback: {type(e).__name__}: {e}"
                )
                spacetime_label = "fallback"

        ai_warp_label = "none"
        if observer.ai_warp_model is not None:
            try:
                _ = observer.ai_warp_model.confidence()
                ai_warp_label = type(observer.ai_warp_model).__name__
            except Exception as e:  # pragma: no cover - defensive
                notes.append(
                    f"ai_warp_model fallback: {type(e).__name__}: {e}"
                )
                ai_warp_label = "fallback"

        frame_data = None
        out_path = observer.config.get("output_ppm_path")
        try:
            samples = build_star_photon_field(rep_objects, camera)
        except Exception as e:  # pragma: no cover - defensive
            samples = []
            notes.append(f"photon_field error: {type(e).__name__}: {e}")
        # Apply this observer's perception (relativistic aberration +
        # warp_factor + optional AI warp) so identical scene data
        # produces a different reality per observer.
        if samples:
            obs_state = ObserverState(
                position_m=observer.position_m,
                velocity_m_s=observer.velocity_m_s,
                forward=observer.forward,
                up=observer.up,
                warp_factor=effective_warp,
            )
            try:
                samples = transform_photon_field(
                    samples,
                    obs_state,
                    ai_model=observer.ai_warp_model,
                )
            except Exception as e:  # pragma: no cover - defensive
                notes.append(f"perception error: {type(e).__name__}: {e}")
        if samples and out_path:
            try:
                render_photon_field_to_ppm(samples, camera, str(out_path))
                frame_data = self._load_ppm_pixels(out_path)
            except Exception as e:  # pragma: no cover - defensive
                notes.append(f"render error: {type(e).__name__}: {e}")

        scene_state = self.build_scene_state()
        if notes:
            scene_state.notes = list(scene_state.notes) + notes

        visible_events = self.get_visible_events(observer)

        return RealityView(
            observer_id=observer.id,
            scene_state=scene_state,
            representation_type=rep_type,
            frame_data=frame_data,
            metadata={
                "warp_factor": observer.warp_factor,
                "effective_warp_factor": effective_warp,
                "beta": observer.beta(),
                "spacetime_model": spacetime_label,
                "ai_warp_model": ai_warp_label,
                "object_count": len(rep_objects),
                "active_count": len(active),
                "sample_count": len(samples),
                "zone_name": zone_name,
                "output_ppm_path": str(out_path) if out_path else None,
                "visible_event_ids": [e.id for e in visible_events],
            },
            proper_time_tau=float(observer.proper_time_tau),
            coordinate_time_t=float(observer.coordinate_time_t),
            visible_event_count=len(visible_events),
            active_rule_ids=list(rule_context.active_rule_ids),
            reality_metadata=dict(rule_context.metadata),
        )

    def step_all_observers(
        self,
        delta_seconds: float,
    ) -> list["RealityView"]:
        """Advance the clock once, then render a view per observer.

        Camera is taken from the observer's pose. If a SimpleCamera is
        attached via ``observer.config['camera']`` it's used as-is;
        otherwise a default 90° camera is built from the observer's
        position / forward / up.
        """
        from cosmic_engine.rendering import SimpleCamera

        self.step(delta_seconds)
        views: list["RealityView"] = []
        for observer in self.observer_manager.list_observers():
            cam = observer.config.get("camera")
            if cam is None:
                cam = SimpleCamera(
                    position_m=observer.position_m,
                    forward=observer.forward,
                    up=observer.up,
                    fov_degrees=float(observer.config.get("fov_degrees", 90.0)),
                    image_width=int(observer.config.get("image_width", 256)),
                    image_height=int(observer.config.get("image_height", 256)),
                )
            views.append(self.render_for_observer(observer.id, cam))
        return views

    def _build_rule_context(self, observer, scale_zone_name: str | None):
        """Construct a fresh :class:`RuleContext` for this render."""
        from cosmic_engine.reality.rule_context import RuleContext

        truth_counts: dict[str, int] = {}
        for obj in self.registry.list_objects():
            tl = obj.truth_level.value
            truth_counts[tl] = truth_counts.get(tl, 0) + 1
        return RuleContext(
            observer_id=observer.id,
            observer_position_m=observer.position_m,
            observer_velocity_m_s=observer.velocity_m_s,
            warp_factor=float(observer.warp_factor),
            coordinate_time_t=float(observer.coordinate_time_t),
            proper_time_tau=float(observer.proper_time_tau),
            scale_zone=scale_zone_name,
            truth_level_counts=truth_counts,
        )

    def _evaluate_reality_rules(self, context):
        """Run ``self.reality_rule_engine`` if attached; passthrough otherwise."""
        if self.reality_rule_engine is None:
            return context.clone()
        try:
            return self.reality_rule_engine.evaluate(context)
        except Exception as e:  # pragma: no cover - defensive
            # Never let a bad rule break a render. Stash the error
            # marker in metadata so it's visible from the RealityView.
            fallback = context.clone()
            fallback.metadata["reality_rule_error"] = (
                f"{type(e).__name__}: {e}"
            )
            return fallback

    @staticmethod
    def _load_ppm_pixels(path) -> "np.ndarray | None":
        """Read a PPM file written by :func:`render_photon_field_to_ppm`."""
        from pathlib import Path as _P

        import numpy as _np

        from PIL import Image as _Image

        p = _P(path)
        if not p.is_file():
            return None
        try:
            with _Image.open(p) as im:
                if im.mode != "RGB":
                    im = im.convert("RGB")
                return _np.asarray(im, dtype=_np.uint8).copy()
        except Exception:  # pragma: no cover - defensive
            return None

    # --- streaming bring-up ----------------------------------------------

    def build_spatial_index(self, cell_size_m: float) -> SpatialIndex:
        """Build a :class:`SpatialIndex` from the current registry contents."""
        index = SpatialIndex(cell_size_m)
        index.build(self.registry.list_objects())
        self.spatial_index = index
        return index

    def enable_streaming(
        self,
        cell_size_m: float,
        data_directory: str,
        clear_registry: bool = True,
        cache_size: int = _DEFAULT_TILE_CACHE_SIZE,
    ) -> TileStore:
        """Partition the current registry into tiles on disk.

        With ``clear_registry=True`` (the default) the in-memory
        registry is emptied so memory tracks the cache rather than
        the full dataset.
        """
        if cell_size_m <= 0.0:
            raise ValueError("cell_size_m must be positive")
        store = TileStore(data_directory)
        partitions = partition_objects_into_tiles(
            self.registry.list_objects(), cell_size_m
        )
        for tile_id, objects in partitions.items():
            store.save_tile(tile_id, objects)
        self.tile_store = store
        self._tile_cell_size_m = cell_size_m
        self._tile_cache = _TileCache(max_size=cache_size)
        if clear_registry:
            self.registry = UniverseRegistry()
        return store

    # --- stepping ---------------------------------------------------------

    def step(self, delta_seconds: float) -> SceneState:
        """Advance the clock and (if configured) one physics step.

        Also advances ``self.coordinate_time_t`` and every registered
        observer's ``proper_time_tau`` / ``coordinate_time_t`` (using
        the local gravitational potential from any massive registry
        objects when available).
        """
        if delta_seconds < 0.0:
            raise ValueError("delta_seconds must be non-negative")
        self.clock.tick(delta_seconds)
        self.coordinate_time_t += float(delta_seconds)
        notes: list[str] = [
            f"clock advanced by {delta_seconds} s",
        ]

        if self.config.enable_physics and self.config.physics_backend != "none":
            notes.extend(self._run_physics_step(delta_seconds))
        else:
            notes.append("physics: disabled")

        # Phase 34: advance each observer's proper time. We re-use the
        # registry's massive bodies as the gravitational sources for the
        # weak-field correction, capping at the most-massive 16 to keep
        # the per-observer cost bounded for large registries.
        if len(self.observer_manager) > 0:
            masses = self._collect_massive_for_potential(limit=16)
            for observer in self.observer_manager.list_observers():
                try:
                    observer.advance_time(
                        delta_seconds,
                        masses_for_potential=masses if masses else None,
                    )
                except Exception as e:  # pragma: no cover - defensive
                    notes.append(
                        f"observer {observer.id} advance_time failed: "
                        f"{type(e).__name__}: {e}"
                    )

        state = self.build_scene_state()
        state.notes = notes
        self.last_scene_state = state
        return state

    def _collect_massive_for_potential(
        self,
        limit: int = 16,
    ) -> list[tuple]:
        """Return up to ``limit`` ``(position_xyz, mass_kg)`` pairs.

        Picks the heaviest objects so the dominant potential is
        captured; the long tail is dropped to keep per-observer
        gravitational-potential evaluation O(1) per step.
        """
        import numpy as _np

        massive = [
            o for o in self.registry.list_objects()
            if o.mass_kg is not None and o.mass_kg > 0.0
        ]
        if not massive:
            return []
        massive.sort(key=lambda o: float(o.mass_kg), reverse=True)
        out: list[tuple] = []
        for obj in massive[:limit]:
            out.append((
                _np.array(
                    [obj.position_m.x, obj.position_m.y, obj.position_m.z],
                    dtype=_np.float64,
                ),
                float(obj.mass_kg),
            ))
        return out

    def get_visible_events(self, observer) -> list:
        """Return the events in ``observer``'s past light cone.

        Filters :attr:`event_store` against
        :func:`cosmic_engine.time.is_event_visible` using the
        observer's current position and ``coordinate_time_t``.
        """
        import numpy as _np

        from cosmic_engine.time.causality import is_event_visible

        position = _np.array(
            [observer.position_m.x, observer.position_m.y, observer.position_m.z],
            dtype=_np.float64,
        )
        return [
            e
            for e in self.event_store.list_events()
            if is_event_visible(e, position, observer.coordinate_time_t)
        ]

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
