"""Demo: per-observer proper time + light-cone event filtering."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from cosmic_engine.core.units import SPEED_OF_LIGHT_M_S
from cosmic_engine.core.vector import Vector3
from cosmic_engine.observer import Observer
from cosmic_engine.runtime import CosmicRuntime, RuntimeConfig
from cosmic_engine.time import Event


_REPO_ROOT = Path(__file__).resolve().parent.parent
_OUT_DIR = _REPO_ROOT / "outputs" / "viewer"


def main() -> None:
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

    # --- 3 observers at different points and velocities --------------
    # 1 light-second ≈ 3e8 m. Position observers at light-second scales
    # so visibility transitions land between sim steps. Keep them well
    # off the Sun (origin) so the weak-field potential doesn't dominate.
    one_ls = SPEED_OF_LIGHT_M_S
    near = Observer(
        id="near",
        position_m=Vector3(one_ls, 0.0, 0.0),  # 1 ls from origin
        velocity_m_s=Vector3.zero(),
        forward=Vector3(0.0, 1.0, 0.0),
        up=Vector3(0.0, 0.0, 1.0),
    )
    far = Observer(
        id="far",
        position_m=Vector3(10.0 * one_ls, 0.0, 0.0),  # 10 ls
        velocity_m_s=Vector3.zero(),
        forward=Vector3(0.0, 1.0, 0.0),
        up=Vector3(0.0, 0.0, 1.0),
    )
    moving = Observer(
        id="moving",
        position_m=Vector3(one_ls, 0.0, 0.0),  # 1 ls, but moving
        velocity_m_s=Vector3(0.8 * SPEED_OF_LIGHT_M_S, 0.0, 0.0),
        forward=Vector3(0.0, 1.0, 0.0),
        up=Vector3(0.0, 0.0, 1.0),
    )
    for obs in (near, far, moving):
        runtime.observer_manager.add_observer(obs)

    # --- inject 3 events ---------------------------------------------
    # Origin flash at t=0 — light reaches "near"/"moving" after 1 s,
    # "far" after 10 s.
    runtime.event_store.add_event(
        Event(
            id="origin_flash",
            position_m=np.zeros(3),
            time_t=0.0,
            payload={"label": "origin flash"},
        )
    )
    # Flash at far observer's location, also at t=0. Light arrives at
    # "far" immediately, at "near"/"moving" after 9 s.
    runtime.event_store.add_event(
        Event(
            id="far_flash",
            position_m=np.array([10.0 * one_ls, 0.0, 0.0]),
            time_t=0.0,
            payload={"label": "flash at far observer"},
        )
    )
    # An event scheduled for t = 6.0 s at the origin. No observer sees
    # it on step 0 (coordinate time only at 2 s); "near"/"moving" see
    # it once t > 7 s; "far" once t > 16 s.
    runtime.event_store.add_event(
        Event(
            id="future_flash",
            position_m=np.zeros(3),
            time_t=6.0,
            payload={"label": "future event"},
        )
    )
    print(f"events in store    : {len(runtime.event_store)}")
    print()

    # --- step a few times, dump per-observer subjective state --------
    n_steps = 6
    delta_t = 2.0  # 2 s per step → 12 s total; light walks ~6 ls
    print(
        f"{'step':<5} {'observer':<8} {'t':>8} {'tau':>8} "
        f"{'beta':>5} {'visible_events':<32}"
    )
    print("-" * 80)
    for step_idx in range(n_steps):
        runtime.step(delta_t)
        for obs in runtime.observer_manager.list_observers():
            visible = runtime.get_visible_events(obs)
            ev_ids = ",".join(e.id for e in visible) or "-"
            print(
                f"{step_idx:<5} {obs.id:<8} "
                f"{obs.coordinate_time_t:>8.2f} "
                f"{obs.proper_time_tau:>8.2f} "
                f"{obs.beta():>5.2f} "
                f"{ev_ids:<32}"
            )

    # --- subjective time gap report ----------------------------------
    print()
    print("subjective time vs coordinate time at end of run:")
    for obs in runtime.observer_manager.list_observers():
        diff = obs.coordinate_time_t - obs.proper_time_tau
        print(
            f"  {obs.id:<8} t={obs.coordinate_time_t:.2f} "
            f"τ={obs.proper_time_tau:.2f}  Δ={diff:.2f} s"
        )

    # --- visible event totals ---------------------------------------
    print()
    print("visible event counts at end of run:")
    for obs in runtime.observer_manager.list_observers():
        n = len(runtime.get_visible_events(obs))
        print(f"  {obs.id:<8} visible_events={n}")


if __name__ == "__main__":
    main()
