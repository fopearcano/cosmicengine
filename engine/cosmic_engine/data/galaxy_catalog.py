"""Galaxy catalog ingestion and synthetic DESI-like generator.

The CSV loader is the real-data entry point: columns ``id``, ``name``,
``x_m``, ``y_m``, ``z_m`` are required; ``redshift_z``,
``apparent_magnitude``, and ``morphology`` are optional per row (empty
cells become ``None``).

The synthetic generator is a **structural placeholder** — filamentary
positions, a linear redshift proxy, and magnitudes that fall off with
distance. It exists so the rest of the pipeline can be exercised on
~50k-scale point clouds before real DESI ingestion is wired up.
"""

from __future__ import annotations

import logging

import numpy as np

from cosmic_engine.cosmos.galaxy import GalaxyProperties, create_galaxy_object
from cosmic_engine.core.registry import UniverseRegistry
from cosmic_engine.core.universe_object import UniverseObject
from cosmic_engine.core.vector import Vector3
from cosmic_engine.data.catalog_loader import load_csv

_LOG = logging.getLogger(__name__)

_REQUIRED_COLUMNS = (
    "id",
    "name",
    "x_m",
    "y_m",
    "z_m",
    "redshift_z",
    "apparent_magnitude",
    "morphology",
)


def _parse_optional_float(value: str) -> float | None:
    value = value.strip()
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _parse_row(row: dict[str, str]) -> UniverseObject | None:
    try:
        obj_id = row["id"]
        name = row["name"]
        if not obj_id or not name:
            return None
        x = float(row["x_m"])
        y = float(row["y_m"])
        z = float(row["z_m"])
    except (KeyError, ValueError):
        return None

    morphology = row.get("morphology", "").strip() or None
    redshift = _parse_optional_float(row.get("redshift_z", ""))
    apparent_magnitude = _parse_optional_float(row.get("apparent_magnitude", ""))

    properties = GalaxyProperties(
        morphology=morphology,
        redshift_z=redshift,
        apparent_magnitude=apparent_magnitude,
    )
    return create_galaxy_object(
        id=obj_id,
        name=name,
        position_m=Vector3(x, y, z),
        properties=properties,
        source="CSV",
    )


def load_galaxy_catalog(file_path: str) -> list[UniverseObject]:
    """Load a galaxy CSV into a list of :class:`UniverseObject`.

    Missing headers raise :class:`ValueError`; rows with invalid
    ``id`` / ``name`` / numeric positions are skipped and counted via
    the module logger.
    """
    rows = load_csv(file_path)
    if rows:
        missing = [c for c in _REQUIRED_COLUMNS if c not in rows[0]]
        if missing:
            raise ValueError(
                f"galaxy catalog {file_path} missing columns: {missing}"
            )

    objects: list[UniverseObject] = []
    skipped = 0
    for row in rows:
        obj = _parse_row(row)
        if obj is None:
            skipped += 1
            continue
        objects.append(obj)
    if skipped:
        _LOG.warning(
            "load_galaxy_catalog(%s): skipped %d invalid row(s)",
            file_path,
            skipped,
        )
    return objects


def load_galaxy_catalog_into_registry(
    file_path: str,
    registry: UniverseRegistry,
) -> None:
    """Load a galaxy catalog and add each object to ``registry``.

    Galaxies whose IDs already exist in the registry are skipped; the
    load is idempotent over repeated calls.
    """
    duplicates = 0
    for obj in load_galaxy_catalog(file_path):
        if registry.get_object(obj.id) is not None:
            duplicates += 1
            continue
        registry.add_object(obj)
    if duplicates:
        _LOG.warning(
            "load_galaxy_catalog_into_registry(%s): skipped %d duplicate id(s)",
            file_path,
            duplicates,
        )


def generate_synthetic_galaxy_catalog(
    count: int,
    radius_m: float,
    seed: int = 42,
) -> list[UniverseObject]:
    """Generate ``count`` galaxies on a filament-like distribution.

    Structural placeholder for real DESI ingestion. Galaxies are
    clustered around a handful of random axes through the origin with
    perpendicular Gaussian jitter; redshift and magnitude are loose
    monotonic functions of distance. Same ``seed`` ⇒ byte-identical
    output.
    """
    if count <= 0:
        return []
    if radius_m <= 0:
        raise ValueError("radius_m must be positive")

    rng = np.random.default_rng(seed)

    n_filaments = 6
    axes = rng.standard_normal((n_filaments, 3))
    axes /= np.linalg.norm(axes, axis=1, keepdims=True)

    filament_index = rng.integers(0, n_filaments, count)
    t = rng.uniform(-1.0, 1.0, count) * radius_m
    jitter = rng.standard_normal((count, 3)) * (radius_m * 0.04)
    positions = axes[filament_index] * t[:, None] + jitter

    # Clip extreme outliers so every galaxy is comfortably inside the
    # requested radius for downstream gridding.
    distances = np.linalg.norm(positions, axis=1)
    overshoot = distances > radius_m
    if overshoot.any():
        positions[overshoot] *= (radius_m / distances[overshoot])[:, None]
        distances[overshoot] = radius_m

    redshifts = distances / radius_m * 1.5  # linear proxy; not cosmology
    magnitude_noise = rng.normal(0.0, 0.7, count)
    magnitudes = 15.0 + 5.0 * (distances / radius_m) + magnitude_noise

    galaxies: list[UniverseObject] = []
    for i in range(count):
        properties = GalaxyProperties(
            morphology=None,
            redshift_z=float(redshifts[i]),
            apparent_magnitude=float(magnitudes[i]),
        )
        galaxies.append(
            create_galaxy_object(
                id=f"synthgal_{i}",
                name=f"SynthGal {i}",
                position_m=Vector3(
                    float(positions[i, 0]),
                    float(positions[i, 1]),
                    float(positions[i, 2]),
                ),
                properties=properties,
                source="synthetic_desi_like",
            )
        )
    return galaxies
