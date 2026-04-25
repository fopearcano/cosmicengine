"""Demo: render the same observer under three reality presets."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from cosmic_engine.core.units import LIGHTYEAR_IN_METERS, SPEED_OF_LIGHT_M_S
from cosmic_engine.core.vector import Vector3
from cosmic_engine.observer import Observer
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


def _make_runtime() -> CosmicRuntime:
    runtime = CosmicRuntime(
        config=RuntimeConfig(
            enable_physics=False,
            enable_perception=False,
            active_radius_m=1.0e30,
            max_active_objects=10_000,
        )
    )
    runtime.load_sample_data()
    return runtime


def _take_snapshot(runtime: CosmicRuntime) -> dict:
    """Return a small fingerprint of the universe state for tamper-checking."""
    objects = runtime.registry.list_objects()
    if not objects:
        return {
            "n": 0,
            "ids_hash": 0,
            "warps": (),
        }
    ids_hash = sum(hash(o.id) for o in objects)
    truth_counts: dict[str, int] = {}
    for o in objects:
        tl = o.truth_level.value
        truth_counts[tl] = truth_counts.get(tl, 0) + 1
    return {
        "n": len(objects),
        "ids_hash": ids_hash,
        "truth_counts": truth_counts,
    }


def main() -> None:
    width, height = 192, 192
    runtime = _make_runtime()
    print(f"loaded objects     : {len(runtime.registry.list_objects())}")
    _OUT_DIR.mkdir(parents=True, exist_ok=True)

    # One observer, parked 5 ly along +y with a bit of motion so the
    # warp_factor amplification has something to amplify (the
    # perception transform is a function of beta * warp_factor).
    observer = Observer(
        id="explorer",
        position_m=Vector3(0.0, -5.0 * LIGHTYEAR_IN_METERS, 0.0),
        velocity_m_s=Vector3(0.05 * SPEED_OF_LIGHT_M_S, 0.0, 0.0),
        forward=Vector3(0.0, 1.0, 0.0),
        up=Vector3(0.0, 0.0, 1.0),
        warp_factor=1.0,
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

    pre_snapshot = _take_snapshot(runtime)
    pre_observer_warp = observer.warp_factor

    presets = [
        ("scientific", create_scientific_reality_preset()),
        ("hypertravel", create_hypertravel_reality_preset()),
        ("blackhole", create_blackhole_perception_preset()),
    ]

    print()
    print(
        f"{'preset':<14} {'rules':<5} {'effective_warp':>15} "
        f"{'active_rule_ids':<60}"
    )
    print("-" * 100)

    rendered = []
    for label, rules in presets:
        runtime.reality_rule_engine = RealityRuleEngine(rules)
        observer.config["output_ppm_path"] = _OUT_DIR / (
            f"output_reality_{label}.ppm"
        )
        view = runtime.render_for_observer("explorer", camera)
        rendered.append((label, view))
        active = view.active_rule_ids or []
        active_str = ",".join(active) if active else "-"
        eff_warp = view.metadata.get("effective_warp_factor")
        print(
            f"{label:<14} {len(active):<5} {eff_warp:>15.2f} {active_str:<60}"
        )

    print()
    print("reality_metadata per preset:")
    for label, view in rendered:
        meta = view.reality_metadata or {}
        if not meta:
            print(f"  {label:<14} (none)")
        else:
            for k in sorted(meta):
                print(f"  {label:<14} {k} = {meta[k]}")

    # --- tamper check: nothing in the underlying state should have moved
    post_snapshot = _take_snapshot(runtime)
    post_observer_warp = observer.warp_factor
    print()
    print("tamper check:")
    print(f"  registry_n unchanged     : {pre_snapshot['n'] == post_snapshot['n']}")
    print(
        f"  ids_hash unchanged       : "
        f"{pre_snapshot['ids_hash'] == post_snapshot['ids_hash']}"
    )
    print(
        f"  truth_counts unchanged   : "
        f"{pre_snapshot['truth_counts'] == post_snapshot['truth_counts']}"
    )
    print(
        f"  observer warp_factor     : pre={pre_observer_warp} "
        f"post={post_observer_warp} same={pre_observer_warp == post_observer_warp}"
    )

    # --- frame difference report
    print()
    sci = next(v for label, v in rendered if label == "scientific")
    if sci.frame_data is not None:
        print("mean abs pixel delta vs scientific:")
        for label, view in rendered:
            if label == "scientific":
                continue
            if view.frame_data is None:
                print(f"  {label:<14} <no frame>")
                continue
            d = float(
                np.mean(
                    np.abs(
                        view.frame_data.astype(np.int32)
                        - sci.frame_data.astype(np.int32)
                    )
                )
            )
            print(f"  {label:<14} {d:.3f}")
    print()
    print(f"frames in            : {_OUT_DIR}")


if __name__ == "__main__":
    main()
