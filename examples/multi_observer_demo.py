"""Demo: three observers perceive the same universe differently."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from cosmic_engine.ai.spacetime_field import SpacetimeFieldModel
from cosmic_engine.core.units import SPEED_OF_LIGHT_M_S
from cosmic_engine.core.vector import Vector3
from cosmic_engine.observer import Observer, RealityView
from cosmic_engine.rendering import SimpleCamera
from cosmic_engine.runtime import CosmicRuntime, RuntimeConfig


_REPO_ROOT = Path(__file__).resolve().parent.parent
_OUT_DIR = _REPO_ROOT / "outputs" / "viewer"


class _MockSpacetimeField(SpacetimeFieldModel):
    """Tiny no-op spacetime model so the observer's slot is non-trivial."""

    def query_acceleration(
        self,
        position: np.ndarray,
        direction: np.ndarray | None = None,
    ) -> np.ndarray:
        return np.zeros(3, dtype=np.float64)

    def confidence(self) -> float:
        return 0.5


def _make_camera(observer: Observer, width: int, height: int) -> SimpleCamera:
    return SimpleCamera(
        position_m=observer.position_m,
        forward=observer.forward,
        up=observer.up,
        fov_degrees=120.0,
        image_width=width,
        image_height=height,
    )


def _frame_summary(view: RealityView) -> str:
    if view.frame_data is None:
        return "frame=<none>"
    pixels = view.frame_data
    return (
        f"frame={pixels.shape[1]}x{pixels.shape[0]} "
        f"mean={pixels.mean():.2f} max={pixels.max()}"
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
    runtime.load_sample_data()
    print(f"loaded objects     : {len(runtime.registry.list_objects())}")

    _OUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- 3 observers, deliberately different setups ------------------
    baseline = Observer(
        id="baseline",
        position_m=Vector3(0.0, -5.0e17, 0.0),
        velocity_m_s=Vector3.zero(),
        forward=Vector3(0.0, 1.0, 0.0),
        up=Vector3(0.0, 0.0, 1.0),
        warp_factor=1.0,
        config={
            "output_ppm_path": _OUT_DIR / "output_obs_baseline.ppm",
            "image_width": width,
            "image_height": height,
        },
    )
    relativistic = Observer(
        id="relativistic",
        position_m=Vector3(0.0, -5.0e17, 0.0),
        velocity_m_s=Vector3(0.6 * SPEED_OF_LIGHT_M_S, 0.0, 0.0),
        forward=Vector3(0.0, 1.0, 0.0),
        up=Vector3(0.0, 0.0, 1.0),
        warp_factor=1.0,
        config={
            "output_ppm_path": _OUT_DIR / "output_obs_relativistic.ppm",
            "image_width": width,
            "image_height": height,
        },
    )
    neural = Observer(
        id="neural",
        position_m=Vector3(0.0, -5.0e17, 0.0),
        velocity_m_s=Vector3(0.1 * SPEED_OF_LIGHT_M_S, 0.0, 0.0),
        forward=Vector3(0.0, 1.0, 0.0),
        up=Vector3(0.0, 0.0, 1.0),
        warp_factor=8.0,
        spacetime_model=_MockSpacetimeField(),
        config={
            "output_ppm_path": _OUT_DIR / "output_obs_neural.ppm",
            "image_width": width,
            "image_height": height,
        },
    )
    for obs in (baseline, relativistic, neural):
        runtime.observer_manager.add_observer(obs)

    print(f"observers          : {len(runtime.observer_manager)}")
    print()
    print(
        f"{'observer':<14} {'beta':>5} {'warp':>5} {'spacetime':<22} "
        f"{'rep':<10} {'objects':>7} {'frame':<28}"
    )
    print("-" * 100)

    # --- two frames so we can show step_all_observers ----------------
    n_frames = 2
    start = time.perf_counter()
    last_views: list[RealityView] = []
    for frame_idx in range(n_frames):
        # We don't want the bundled output_ppm_path to be overwritten
        # silently between frames — append a frame index.
        for obs in runtime.observer_manager.list_observers():
            obs.config["output_ppm_path"] = _OUT_DIR / (
                f"output_obs_{obs.id}_frame_{frame_idx}.ppm"
            )
        views = []
        for obs in runtime.observer_manager.list_observers():
            cam = _make_camera(obs, width, height)
            views.append(runtime.render_for_observer(obs.id, cam))
        # Advance the clock once after rendering so each frame is a
        # "snapshot in time" — equivalent to step_all_observers but
        # with the per-observer cameras we want for this demo.
        runtime.step(1.0)
        last_views = views

    elapsed = time.perf_counter() - start

    for view in last_views:
        meta = view.metadata
        print(
            f"{view.observer_id:<14} {meta['beta']:>5.2f} "
            f"{meta['warp_factor']:>5.1f} "
            f"{meta['spacetime_model']:<22} "
            f"{view.representation_type:<10} "
            f"{meta['object_count']:>7} {_frame_summary(view):<28}"
        )

    print()
    # Quick frame-difference report (last frame).
    base = next(v for v in last_views if v.observer_id == "baseline")
    diffs = []
    for view in last_views:
        if view.observer_id == "baseline":
            continue
        if view.frame_data is None or base.frame_data is None:
            diffs.append((view.observer_id, None))
            continue
        d = float(
            np.mean(
                np.abs(
                    view.frame_data.astype(np.int32)
                    - base.frame_data.astype(np.int32)
                )
            )
        )
        diffs.append((view.observer_id, d))
    print("frame difference vs baseline (mean abs pixel delta):")
    for oid, d in diffs:
        print(f"  {oid:<14} {('nan' if d is None else f'{d:.3f}')}")

    print()
    print(f"frames per observer  : {n_frames}")
    print(f"observers x frames   : {len(runtime.observer_manager) * n_frames}")
    print(f"total render time    : {elapsed * 1_000:.1f} ms")
    print(f"frames in            : {_OUT_DIR}")


if __name__ == "__main__":
    main()
