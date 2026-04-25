"""Demo: audit one observer's renders under three reality presets."""

from __future__ import annotations

from pathlib import Path

from cosmic_engine.ai.base import AIWarpModel
from cosmic_engine.core.units import LIGHTYEAR_IN_METERS, SPEED_OF_LIGHT_M_S
from cosmic_engine.core.vector import Vector3
from cosmic_engine.data.galaxy_catalog import generate_synthetic_galaxy_catalog
from cosmic_engine.observer import Observer
from cosmic_engine.provenance import audit_reality_view
from cosmic_engine.reality import (
    RealityRuleEngine,
    create_blackhole_perception_preset,
    create_hypertravel_reality_preset,
    create_scientific_reality_preset,
)
from cosmic_engine.rendering import SimpleCamera
from cosmic_engine.runtime import CosmicRuntime, RuntimeConfig


_REPO_ROOT = Path(__file__).resolve().parent.parent
_OUT_DIR = _REPO_ROOT / "outputs" / "viewer"


class _IdentityAIWarp(AIWarpModel):
    """Identity AI warp model — emits an ``ai_warp`` provenance label
    without changing the perceived photon field. Lets the demo
    exercise the AI-mixing audit path on observed (Gaia) data."""

    def predict_direction(self, direction, observer):  # noqa: ARG002
        return direction

    def predict_brightness(self, brightness, direction, observer):  # noqa: ARG002
        return brightness

    def predict_color(self, color_rgb, direction, observer):  # noqa: ARG002
        return color_rgb

    def confidence(self) -> float:
        return 1.0


def _print_audit(label: str, view) -> None:
    audit = audit_reality_view(view)
    summary = view.provenance_summary or {}
    print()
    print(f"--- {label} ---")
    print(
        f"observer            : {audit['observer_id']}"
    )
    print(
        f"truth_distribution  : "
        f"{summary.get('truth_level_counts', {})}"
    )
    print(
        f"source_counts       : "
        f"{summary.get('source_counts', {})}"
    )
    print(
        f"transformations     : {audit['transformations']}"
    )
    print(
        f"active_rules        : {audit['active_rules']}"
    )
    print(
        f"audit_warnings      : "
        f"{view.audit_warnings if view.audit_warnings else '<none>'}"
    )


def main() -> None:
    width, height = 192, 192
    runtime = CosmicRuntime(
        config=RuntimeConfig(
            enable_physics=False,
            enable_perception=False,
            active_radius_m=1.0e30,
            max_active_objects=10_000,
        )
    )
    # Mixed dataset: bundled (Gaia=observed, SDSS, DESI, JPL) + synthetic.
    runtime.load_sample_data()
    runtime.add_objects(
        generate_synthetic_galaxy_catalog(200, radius_m=5.0e22, seed=42)
    )
    n_objects = len(runtime.registry.list_objects())
    print(f"loaded objects     : {n_objects}")
    print(f"provenance records : {len(runtime.truth_tracker)}")
    _OUT_DIR.mkdir(parents=True, exist_ok=True)

    observer = Observer(
        id="explorer",
        position_m=Vector3(0.0, -5.0 * LIGHTYEAR_IN_METERS, 0.0),
        velocity_m_s=Vector3(0.05 * SPEED_OF_LIGHT_M_S, 0.0, 0.0),
        forward=Vector3(0.0, 1.0, 0.0),
        up=Vector3(0.0, 0.0, 1.0),
        warp_factor=1.0,
        ai_warp_model=_IdentityAIWarp(),
    )
    runtime.observer_manager.add_observer(observer)
    camera = SimpleCamera(
        position_m=observer.position_m,
        forward=observer.forward,
        up=observer.up,
        fov_degrees=120.0,
        image_width=width,
        image_height=height,
    )

    presets = [
        ("scientific", create_scientific_reality_preset()),
        ("hypertravel", create_hypertravel_reality_preset()),
        ("blackhole", create_blackhole_perception_preset()),
    ]
    for label, rules in presets:
        runtime.reality_rule_engine = RealityRuleEngine(rules)
        observer.config["output_ppm_path"] = _OUT_DIR / (
            f"output_audit_{label}.ppm"
        )
        view = runtime.render_for_observer("explorer", camera)
        _print_audit(label, view)

    print()
    # Final tracker statistics
    print(f"final provenance records : {len(runtime.truth_tracker)}")
    transformed = sum(
        1 for r in runtime.truth_tracker.list_records() if r.transformations
    )
    print(f"records with transforms  : {transformed}")
    sample = runtime.truth_tracker.list_records()[:1]
    if sample:
        rec = sample[0]
        print(
            f"example record           : {rec.entity_id} "
            f"source={rec.source} truth_level={rec.truth_level} "
            f"transformations={rec.transformations}"
        )


if __name__ == "__main__":
    main()
