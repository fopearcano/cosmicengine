"""Demo: detect drifting neural model + suggest retraining (no mutation)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from cosmic_engine.adaptive import AdaptiveEngine, AdaptivePolicy
from cosmic_engine.ai.spacetime_field import SpacetimeFieldModel
from cosmic_engine.core.units import LIGHTYEAR_IN_METERS
from cosmic_engine.core.vector import Vector3
from cosmic_engine.observer import Observer
from cosmic_engine.rendering import SimpleCamera
from cosmic_engine.runtime import CosmicRuntime, RuntimeConfig


_REPO_ROOT = Path(__file__).resolve().parent.parent
_OUT_DIR = _REPO_ROOT / "outputs" / "viewer"


class _DriftingSpacetime(SpacetimeFieldModel):
    """Analytical Schwarzschild scaled by ``bias`` to simulate drift.

    ``bias = 1.0`` matches analytical exactly (no drift). The demo
    bumps it slightly above 1 so the adaptive engine sees a sustained
    discrepancy rather than random noise.
    """

    def __init__(self, *, mass_kg: float, bias: float) -> None:
        self.mass_kg = float(mass_kg)
        self.bias = float(bias)

    def query_acceleration(self, position, direction=None):  # noqa: ARG002
        position = np.asarray(position, dtype=np.float64)
        r = float(np.linalg.norm(position))
        if r == 0.0:
            return np.zeros(3, dtype=np.float64)
        analytical = (
            -2.0 * 6.67430e-11 * self.mass_kg / (r ** 3)
        ) * position
        return analytical * self.bias

    def confidence(self) -> float:
        return 0.85


def main() -> None:
    width, height = 128, 128
    runtime = CosmicRuntime(
        config=RuntimeConfig(
            enable_physics=False,
            enable_perception=False,
            active_radius_m=1.0e30,
            max_active_objects=1_000,
        )
    )
    runtime.load_sample_data()
    print(f"loaded objects     : {len(runtime.registry.list_objects())}")
    _OUT_DIR.mkdir(parents=True, exist_ok=True)

    # The "drift" — neural model returns 30 % too much acceleration.
    # That's a deviation of 0.30 every probe, well above the 0.10
    # default threshold.
    reference_mass = 1.989e30
    spacetime = _DriftingSpacetime(mass_kg=reference_mass, bias=1.30)

    observer = Observer(
        id="explorer",
        position_m=Vector3(0.0, -1.0 * LIGHTYEAR_IN_METERS, 0.0),
        velocity_m_s=Vector3.zero(),
        forward=Vector3(0.0, 1.0, 0.0),
        up=Vector3(0.0, 0.0, 1.0),
        warp_factor=1.0,
        spacetime_model=spacetime,
        config={
            "output_ppm_path": _OUT_DIR / "output_adaptive.ppm",
            "adaptive_reference_mass_kg": reference_mass,
            "adaptive_probe_radius_m": 1.0e9,
        },
    )
    runtime.observer_manager.add_observer(observer)
    camera = SimpleCamera(
        position_m=observer.position_m,
        forward=observer.forward,
        up=observer.up,
        fov_degrees=90.0,
        image_width=width,
        image_height=height,
    )

    # Tight policy so a small number of consistent probes triggers a
    # suggestion and the demo runs in seconds.
    policy = AdaptivePolicy(
        error_threshold=0.10,
        window_size=5,
        sustained_fraction=0.6,
        max_updates_per_run=4,
    )
    runtime.adaptive_engine = AdaptiveEngine(policy=policy)

    n_steps = 8
    print()
    print(
        f"{'step':<5} {'records':>8} {'max_dev':>8} "
        f"{'mean_dev':>9} {'suggestions':<14}"
    )
    print("-" * 60)

    last_view = None
    for step in range(n_steps):
        view = runtime.render_for_observer("explorer", camera)
        summary = view.feedback_summary
        suggestions = view.adaptive_suggestions
        sugg_actions = ",".join(s["action"] for s in suggestions) or "-"
        print(
            f"{step:<5} {summary.get('record_count', 0):>8} "
            f"{summary.get('max_deviation', 0.0):>8.4f} "
            f"{summary.get('mean_deviation', 0.0):>9.4f} "
            f"{sugg_actions:<14}"
        )
        last_view = view

    print()
    print("--- final feedback summary ---")
    if last_view is not None:
        for k, v in sorted(last_view.feedback_summary.items()):
            print(f"  {k:<14} = {v}")

        print()
        print("--- final adaptive suggestions ---")
        if last_view.adaptive_suggestions:
            for s in last_view.adaptive_suggestions:
                print(f"  action      : {s['action']}")
                print(f"  source      : {s['source']}")
                print(f"  reason      : {s['reason']}")
                print(f"  feedback_ids: {s['feedback_ids']}")
        else:
            print("  (none)")

    # No-mutation report: the observer's model and the runtime's
    # rule engine are still the original instances; nothing has been
    # silently retrained or swapped.
    print()
    print("--- no-mutation check ---")
    print(
        f"  observer.spacetime_model is original : "
        f"{observer.spacetime_model is spacetime}"
    )
    print(f"  spacetime.bias unchanged             : {spacetime.bias == 1.30}")
    print(
        f"  runtime.reality_rule_engine          : "
        f"{runtime.reality_rule_engine}"
    )
    print(f"  observer.warp_factor                 : {observer.warp_factor}")


if __name__ == "__main__":
    main()
