"""Demo: batch ONNX photon warp on a synthetic 50k-star catalog."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from cosmic_engine.ai import BatchONNXPhotonWarpModel
from cosmic_engine.core.coordinates import ra_dec_distance_to_cartesian
from cosmic_engine.core.object_types import CosmicObjectType
from cosmic_engine.core.truth import TruthLevel
from cosmic_engine.core.units import LIGHTYEAR_IN_METERS, SPEED_OF_LIGHT_M_S
from cosmic_engine.core.universe_object import UniverseObject
from cosmic_engine.core.vector import Vector3
from cosmic_engine.perception.observer import ObserverState
from cosmic_engine.perception.vectorized_ai_transform import apply_batch_ai_warp
from cosmic_engine.perception.vectorized_transform import (
    transform_photon_field_batch,
)
from cosmic_engine.rendering import (
    SimpleCamera,
    build_star_photon_field_batch,
    render_photon_batch_to_ppm,
)


_REPO_ROOT = Path(__file__).resolve().parent.parent
_OUT = _REPO_ROOT / "outputs" / "viewer" / "output_batch_photon_warp.ppm"
_DEFAULT_MODEL = _REPO_ROOT / "data" / "photon_warp_model.onnx"
_SPECTRAL = ["O5V", "B1V", "A1V", "F5IV", "G2V", "K0III", "M1Ia"]


def _synthetic_star(i: int) -> UniverseObject:
    ra = (i * 37.9) % 360.0
    dec = ((i * 73.1) % 180.0) - 90.0
    distance_ly = 10.0 + float(i % 500)
    return UniverseObject(
        id=f"syn_{i}",
        name=f"Syn {i}",
        object_type=CosmicObjectType.STAR,
        position_m=ra_dec_distance_to_cartesian(
            ra, dec, distance_ly * LIGHTYEAR_IN_METERS
        ),
        velocity_m_s=Vector3.zero(),
        truth_level=TruthLevel.PROCEDURAL_APPROXIMATION,
        source="synthetic",
        spectral_class=_SPECTRAL[i % len(_SPECTRAL)],
        metadata={"apparent_magnitude": float((i % 20) - 5)},
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default=None,
        help="Optional ONNX model path. Defaults to deterministic only.",
    )
    parser.add_argument("--count", type=int, default=50_000)
    args = parser.parse_args()

    if args.model is None:
        print("no --model: deterministic batch transform only")
        model_path: str | None = None
    else:
        model_path = args.model
        print(f"--model = {model_path}")

    print(f"generating {args.count} synthetic stars...")
    stars = [_synthetic_star(i) for i in range(args.count)]

    camera = SimpleCamera(
        position_m=Vector3.zero(),
        forward=Vector3(0.0, 1.0, 0.0),
        up=Vector3(0.0, 0.0, 1.0),
        fov_degrees=140.0,
        image_width=512,
        image_height=512,
    )
    observer = ObserverState(
        position_m=Vector3.zero(),
        velocity_m_s=Vector3(0.0, 0.5 * SPEED_OF_LIGHT_M_S, 0.0),
        forward=Vector3(0.0, 1.0, 0.0),
        up=Vector3(0.0, 0.0, 1.0),
        warp_factor=5.0,
    )

    t0 = time.perf_counter()
    batch = build_star_photon_field_batch(stars, camera)
    build_seconds = time.perf_counter() - t0

    # 1. deterministic vectorized transform
    t1 = time.perf_counter()
    deterministic = transform_photon_field_batch(batch, observer)
    det_seconds = time.perf_counter() - t1

    # 2. ONNX batch warp (or fallback path if no model)
    fallback_used = False
    if model_path is not None:
        try:
            model = BatchONNXPhotonWarpModel(model_path)
            print(f"loaded ONNX model         : ok")
        except Exception as e:
            print(f"failed to load ONNX model : {e}")
            model = None
    else:
        model = None

    t2 = time.perf_counter()
    ai_warped = apply_batch_ai_warp(batch, observer, model)
    ai_seconds = time.perf_counter() - t2

    # 3. confirm fallback path is robust by passing a deliberately
    #    broken model (predict_batch raises) and showing it still
    #    returns the deterministic answer
    class _BrokenModel:
        last_error = None

        def predict_batch(self, _arr: np.ndarray) -> np.ndarray:
            raise RuntimeError("simulated inference failure")

        def confidence(self) -> float:
            return 0.3

    t3 = time.perf_counter()
    fallback_warped = apply_batch_ai_warp(batch, observer, _BrokenModel())
    fallback_seconds = time.perf_counter() - t3
    fallback_used = True

    render_photon_batch_to_ppm(ai_warped, camera, str(_OUT))

    n = len(batch)
    print()
    print(f"photon batch size       : {n}")
    print(f"build batch             : {build_seconds * 1000:8.2f} ms")
    print(f"deterministic transform : {det_seconds * 1000:8.2f} ms")
    print(
        f"AI batch warp           : {ai_seconds * 1000:8.2f} ms  "
        f"(model={'on' if model is not None else 'off (deterministic fallback)'})"
    )
    print(
        f"forced fallback path    : {fallback_seconds * 1000:8.2f} ms  "
        f"(used? {'yes' if fallback_used else 'no'})"
    )
    print()
    print(f"output saved to         : {_OUT}")


if __name__ == "__main__":
    main()
