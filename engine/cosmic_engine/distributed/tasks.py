"""Ray remote tasks (with non-Ray fallbacks).

Each task obeys two rules that make tests + air-gapped runs easy:

1. The CosmicEngine submodule that does the actual work is imported
   *inside* the function, so importing :mod:`tasks` doesn't require
   the full engine to be ready.
2. Every task returns a JSON-serializable ``dict`` and never raises:
   on failure the dict carries an ``error`` field describing the
   failure. Callers (whether plain Python or Ray) get the same
   contract.

The :func:`as_remote` helper wraps a task with ``ray.remote`` *only*
when Ray is available; otherwise the function is returned unchanged.
That keeps the call site identical in both modes::

    submit = as_remote(process_catalog_chunk)
    result = submit(payload)            # local: returns dict
    handle = submit.remote(payload)     # ray: returns ObjectRef
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from cosmic_engine.distributed.ray_runtime import is_ray_available


_LOG = logging.getLogger(__name__)


def as_remote(func: Callable) -> Callable:
    """Wrap ``func`` with ``ray.remote`` if Ray is importable, else passthrough."""
    if not is_ray_available():
        return func
    try:
        import ray

        return ray.remote(func)
    except Exception:  # pragma: no cover - defensive
        return func


def _err(message: str, **extra: Any) -> dict[str, Any]:
    """Build a uniform failure dict."""
    out: dict[str, Any] = {"ok": False, "error": message}
    out.update(extra)
    return out


def _ok(**fields: Any) -> dict[str, Any]:
    """Build a uniform success dict."""
    out: dict[str, Any] = {"ok": True}
    out.update(fields)
    return out


# --- 1. catalog ingestion ---------------------------------------------


def process_catalog_chunk(chunk_path: str) -> dict[str, Any]:
    """Load one CSV catalog chunk and return basic stats.

    Doesn't mutate anything — designed for distributed warm-up of
    catalogs onto worker nodes.
    """
    from pathlib import Path

    if not chunk_path:
        return _err("chunk_path is empty")
    p = Path(chunk_path)
    if not p.is_file():
        return _err(f"file not found: {chunk_path}")
    try:
        # Lazy import: keeps the task importable on barebones nodes.
        from cosmic_engine.data.gaia_catalog import load_gaia_like_catalog

        objects = load_gaia_like_catalog(str(p))
        return _ok(
            chunk_path=str(p),
            object_count=len(objects),
        )
    except Exception as e:  # pragma: no cover - defensive
        return _err(f"{type(e).__name__}: {e}", chunk_path=str(p))


# --- 2. physics step --------------------------------------------------


def run_physics_step(serialized_state: dict) -> dict[str, Any]:
    """Step a small N-body state forward by ``dt``.

    Inputs::

        {
            "positions": [[x, y, z], ...],
            "velocities": [[vx, vy, vz], ...],
            "masses": [m, ...],
            "dt": float,
            "integrator": "leapfrog" | "barnes_hut" (optional),
        }

    Returns the new positions / velocities (or an ``error`` field).
    """
    try:
        import numpy as _np

        from cosmic_engine.physics.nbody import (
            NBodySimulator,
            NBodyState,
        )

        positions = _np.asarray(serialized_state["positions"], dtype=_np.float64)
        velocities = _np.asarray(serialized_state["velocities"], dtype=_np.float64)
        masses = _np.asarray(serialized_state["masses"], dtype=_np.float64)
        dt = float(serialized_state["dt"])
        integrator = serialized_state.get("integrator", "leapfrog")

        state = NBodyState(
            positions=positions, velocities=velocities, masses=masses
        )
        sim = NBodySimulator(state, integrator=integrator)
        sim.step(dt)
        out = sim.get_state()
        return _ok(
            positions=out.positions.tolist(),
            velocities=out.velocities.tolist(),
            integrator=integrator,
            dt=dt,
        )
    except Exception as e:  # pragma: no cover - defensive
        return _err(f"{type(e).__name__}: {e}")


# --- 3. AI photon warp batch -----------------------------------------


def run_ai_warp_batch(payload: dict) -> dict[str, Any]:
    """Run a batch of photon warps through an ONNX model on the worker.

    Inputs::

        {
            "model_path": str,
            "directions": [[dx, dy, dz], ...],
            "brightnesses": [b, ...],
            "observer_velocity": [vx, vy, vz],
        }

    Returns predicted directions / brightnesses or an ``error``.
    """
    try:
        import numpy as _np

        from cosmic_engine.ai.onnx_photon_warp_batch import (
            ONNXBatchPhotonWarp,
        )

        model_path = payload.get("model_path")
        if not model_path:
            return _err("model_path is required")
        warp = ONNXBatchPhotonWarp(model_path=model_path)
        directions = _np.asarray(payload["directions"], dtype=_np.float32)
        brightnesses = _np.asarray(payload["brightnesses"], dtype=_np.float32)
        velocity = _np.asarray(payload["observer_velocity"], dtype=_np.float32)
        new_dirs, new_bright = warp.run(directions, brightnesses, velocity)
        return _ok(
            directions=new_dirs.tolist(),
            brightnesses=new_bright.tolist(),
            count=int(directions.shape[0]),
        )
    except Exception as e:  # pragma: no cover - defensive
        return _err(f"{type(e).__name__}: {e}")


# --- 4. density-field reconstruction ---------------------------------


def run_density_reconstruction(payload: dict) -> dict[str, Any]:
    """Run an ONNX density model on a galaxy field batch.

    Inputs::

        {
            "model_path": str,
            "positions": [[x, y, z], ...],
            "luminosities": [L, ...],
        }
    """
    try:
        import numpy as _np

        from cosmic_engine.ai.density_onnx import ONNXDensityModel

        model_path = payload.get("model_path")
        if not model_path:
            return _err("model_path is required")
        model = ONNXDensityModel(model_path=model_path)
        positions = _np.asarray(payload["positions"], dtype=_np.float32)
        luminosities = _np.asarray(payload["luminosities"], dtype=_np.float32)
        density = model.run(positions, luminosities)
        return _ok(
            density_shape=list(density.shape),
            density_min=float(density.min()),
            density_max=float(density.max()),
            density_sum=float(density.sum()),
        )
    except Exception as e:  # pragma: no cover - defensive
        return _err(f"{type(e).__name__}: {e}")


# --- 5. Gaussian splat frame -----------------------------------------


def render_gaussian_frame(payload: dict) -> dict[str, Any]:
    """Render a single Gaussian-splat frame and return its shape.

    Inputs::

        {
            "points": [{"position": [...], "color": [...], "sigma": float}, ...],
            "image_width": int,
            "image_height": int,
            "camera": {
                "position": [x, y, z], "forward": [...], "up": [...],
                "fov_degrees": float,
            },
        }

    Returns the rendered frame's shape and mean intensity rather than
    the pixels themselves so the dict stays small over the wire.
    """
    try:
        # Lazy import: the apps tree may not be available everywhere.
        from ai_viewer.neural_field import GaussianSplatRenderer
        from ai_viewer.neural_field.types import GaussianPoint

        from cosmic_engine.core.vector import Vector3
        from cosmic_engine.rendering import SimpleCamera

        cam_payload = payload["camera"]
        camera = SimpleCamera(
            position_m=Vector3(*cam_payload["position"]),
            forward=Vector3(*cam_payload["forward"]),
            up=Vector3(*cam_payload["up"]),
            fov_degrees=float(cam_payload.get("fov_degrees", 90.0)),
            image_width=int(payload["image_width"]),
            image_height=int(payload["image_height"]),
        )
        points = [
            GaussianPoint(
                position=tuple(p["position"]),
                color=tuple(p.get("color", (1.0, 1.0, 1.0))),
                sigma=float(p.get("sigma", 1.0)),
            )
            for p in payload.get("points", [])
        ]
        renderer = GaussianSplatRenderer(
            payload["image_width"], payload["image_height"], camera
        )
        image = renderer.render(points)
        return _ok(
            shape=list(image.shape),
            dtype=str(image.dtype),
            mean=float(image.mean()),
            point_count=len(points),
        )
    except Exception as e:  # pragma: no cover - defensive
        return _err(f"{type(e).__name__}: {e}")
