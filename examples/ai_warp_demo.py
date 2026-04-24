"""Demo: compare deterministic perception vs. the AI warp placeholder."""

from __future__ import annotations

from pathlib import Path

from cosmic_engine.ai import SimpleNeuralWarp
from cosmic_engine.core.registry import UniverseRegistry
from cosmic_engine.core.units import SPEED_OF_LIGHT_M_S
from cosmic_engine.core.vector import Vector3
from cosmic_engine.data.star_catalog import load_star_catalog_into_registry
from cosmic_engine.perception import ObserverState, transform_photon_field
from cosmic_engine.rendering import (
    SimpleCamera,
    build_star_photon_field,
    render_photon_field_to_ppm,
)

_REPO_ROOT = Path(__file__).resolve().parent.parent
_CATALOG = _REPO_ROOT / "data" / "sample_stars.csv"
_OUTDIR = Path(__file__).resolve().parent


def main() -> None:
    registry = UniverseRegistry()
    load_star_catalog_into_registry(str(_CATALOG), registry)
    objects = registry.list_objects()

    camera = SimpleCamera(
        position_m=Vector3.zero(),
        forward=Vector3(0.0, 1.0, 0.0),
        up=Vector3(0.0, 0.0, 1.0),
        fov_degrees=140.0,
        image_width=512,
        image_height=512,
    )
    samples = build_star_photon_field(objects, camera)

    speed = 0.5 * SPEED_OF_LIGHT_M_S
    model = SimpleNeuralWarp()
    scenarios = [
        ("ai_off", 1.0, None),
        ("ai_on", 1.0, model),
        ("ai_extreme", 50.0, model),
    ]

    for label, warp_factor, ai in scenarios:
        observer = ObserverState(
            position_m=Vector3.zero(),
            velocity_m_s=Vector3(0.0, speed, 0.0),
            forward=Vector3(0.0, 1.0, 0.0),
            up=Vector3(0.0, 0.0, 1.0),
            warp_factor=warp_factor,
        )
        observer.validate()
        warped = transform_photon_field(samples, observer, ai)
        out_path = _OUTDIR / f"output_{label}.ppm"
        render_photon_field_to_ppm(warped, camera, str(out_path))

        active = "yes" if ai is not None else "no"
        conf = f"{ai.confidence():.2f}" if ai is not None else "n/a"
        print(
            f"scenario={label:<10}  ai_active={active:<3}  "
            f"warp_factor={warp_factor:>5}  confidence={conf}  "
            f"samples={len(warped)}  -> {out_path.name}"
        )


if __name__ == "__main__":
    main()
