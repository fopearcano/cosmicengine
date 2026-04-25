"""Demo: generate three synthetic universes from declarative specs."""

from __future__ import annotations

from pathlib import Path

from cosmic_engine.core.vector import Vector3
from cosmic_engine.observer import Observer
from cosmic_engine.rendering import SimpleCamera
from cosmic_engine.synthesis import (
    create_hyperwarp_universe,
    create_runtime_from_spec,
    create_standard_physics_universe,
    create_symbolic_universe,
)


_REPO_ROOT = Path(__file__).resolve().parent.parent
_OUT_DIR = _REPO_ROOT / "outputs" / "viewer"


def _summary(runtime) -> dict:
    spec = runtime.synthesis_spec
    state = runtime.synthesis_state
    objects = runtime.registry.list_objects()
    truth_counts: dict[str, int] = {}
    for o in objects:
        tl = o.truth_level.value
        truth_counts[tl] = truth_counts.get(tl, 0) + 1
    rules = (
        [r.id for r in runtime.reality_rule_engine.list_rules()]
        if runtime.reality_rule_engine is not None
        else []
    )
    return {
        "spec_id": spec.id,
        "seed": spec.seed,
        "object_count": len(objects),
        "scale_limits": spec.scale_limits,
        "physics_model": spec.physics_model,
        "spacetime_model": spec.spacetime_model,
        "rule_ids": rules,
        "constraints": dict(spec.constraints or {}),
        "constraint_notes": list(state.get("constraint_notes", [])),
        "truth_counts": truth_counts,
    }


def _render(runtime, label: str) -> tuple[int, str]:
    """Render one frame from a fixed observer pose; returns (samples, path)."""
    out_path = _OUT_DIR / f"output_universe_{label}.ppm"
    # Camera looks at the origin from a position outside the spec's
    # scale_hi so all objects fall in front.
    spec = runtime.synthesis_spec
    distance = float(spec.scale_limits[1]) * 1.5
    observer = Observer(
        id="cam",
        position_m=Vector3(0.0, -distance, 0.0),
        velocity_m_s=Vector3.zero(),
        forward=Vector3(0.0, 1.0, 0.0),
        up=Vector3(0.0, 0.0, 1.0),
        warp_factor=1.0,
        config={
            "output_ppm_path": out_path,
            "image_width": 192,
            "image_height": 192,
        },
    )
    runtime.observer_manager.add_observer(observer)
    camera = SimpleCamera(
        position_m=observer.position_m,
        forward=observer.forward,
        up=observer.up,
        fov_degrees=110.0,
        image_width=192,
        image_height=192,
    )
    view = runtime.render_for_observer("cam", camera)
    return view.metadata.get("sample_count", 0), str(out_path)


def main() -> None:
    seed = 42
    _OUT_DIR.mkdir(parents=True, exist_ok=True)

    presets = [
        ("standard", create_standard_physics_universe(seed)),
        ("hyperwarp", create_hyperwarp_universe(seed)),
        ("symbolic", create_symbolic_universe(seed)),
    ]

    print(
        f"{'label':<10} {'spec_id':<24} {'seed':>5} "
        f"{'objects':>7} {'rules':<35} {'samples':>7}"
    )
    print("-" * 110)

    runtimes = []
    for label, spec in presets:
        runtime = create_runtime_from_spec(spec)
        sample_count, out_path = _render(runtime, label)
        runtimes.append((label, spec, runtime))
        info = _summary(runtime)
        rules_str = ",".join(info["rule_ids"]) if info["rule_ids"] else "-"
        print(
            f"{label:<10} {info['spec_id']:<24} {info['seed']:>5} "
            f"{info['object_count']:>7} {rules_str:<35} {sample_count:>7}"
        )

    print()
    print("--- per-spec details ---")
    for label, spec, runtime in runtimes:
        info = _summary(runtime)
        print()
        print(f"  [{label}]")
        print(f"    description     : {spec.description}")
        print(f"    scale_limits    : {info['scale_limits']}")
        print(f"    physics_model   : {info['physics_model']}")
        print(f"    spacetime_model : {info['spacetime_model']}")
        print(f"    constraints     : {info['constraints']}")
        print(f"    truth_counts    : {info['truth_counts']}")
        print(f"    constraint_notes: {len(info['constraint_notes'])} adjustment(s)")

    # Determinism: re-running with the same seed must produce
    # identical ids + positions.
    print()
    print("--- determinism check ---")
    spec0 = create_standard_physics_universe(seed=7)
    a = create_runtime_from_spec(spec0)
    b = create_runtime_from_spec(spec0)
    ids_a = sorted(o.id for o in a.registry.list_objects())
    ids_b = sorted(o.id for o in b.registry.list_objects())
    pos_a = sorted(
        (o.id, o.position_m.x, o.position_m.y, o.position_m.z)
        for o in a.registry.list_objects()
    )
    pos_b = sorted(
        (o.id, o.position_m.x, o.position_m.y, o.position_m.z)
        for o in b.registry.list_objects()
    )
    print(f"  ids identical       : {ids_a == ids_b}")
    print(f"  positions identical : {pos_a == pos_b}")

    # Provenance: every synthesized object should be tagged
    # SYNTHETIC_GENERATED, never observed/catalog.
    print()
    print("--- provenance check ---")
    for label, spec, runtime in runtimes:
        levels = {
            o.truth_level.value
            for o in runtime.registry.list_objects()
        }
        sources = {o.source for o in runtime.registry.list_objects()}
        print(
            f"  {label:<10} truth_levels={levels} sources={sources}"
        )

    print()
    print(f"frames in            : {_OUT_DIR}")


if __name__ == "__main__":
    main()
