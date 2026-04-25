"""Demo: compare exact O(N²) vs Barnes–Hut O(N log N) accelerations."""

from __future__ import annotations

import time

import numpy as np

from cosmic_engine.physics.barnes_hut import compute_accelerations_bh
from cosmic_engine.physics.nbody import compute_accelerations


def _synthetic_cluster(n: int, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic 3D cluster of bodies: Gaussian core + uniform halo."""
    rng = np.random.default_rng(seed)
    core_n = int(0.6 * n)
    halo_n = n - core_n
    core = rng.standard_normal((core_n, 3)) * 1.0e10
    halo = rng.uniform(-5.0e10, 5.0e10, size=(halo_n, 3))
    positions = np.concatenate([core, halo], axis=0)
    masses = rng.uniform(1.0e22, 1.0e24, size=n)
    return positions, masses


def _avg_relative_error(a: np.ndarray, b: np.ndarray) -> float:
    """Mean relative error between two acceleration arrays, ignoring zeros."""
    norm_a = np.linalg.norm(a, axis=1)
    diff = np.linalg.norm(a - b, axis=1)
    mask = norm_a > 0.0
    if not mask.any():
        return 0.0
    return float((diff[mask] / norm_a[mask]).mean())


def _bench(n: int) -> None:
    positions, masses = _synthetic_cluster(n)
    softening = 1.0e8

    if n <= 2000:
        t0 = time.perf_counter()
        exact = compute_accelerations(positions, masses, softening_m=softening)
        t_exact = time.perf_counter() - t0
    else:
        exact = None
        t_exact = float("nan")

    print(f"--- N = {n} bodies ---")
    if exact is not None:
        print(f"  exact O(N^2)        : {t_exact * 1000:8.1f} ms")
    else:
        print(f"  exact O(N^2)        :  (skipped, too large)")

    for theta in (0.3, 0.5, 0.8):
        t0 = time.perf_counter()
        bh = compute_accelerations_bh(
            positions, masses, theta=theta, softening_m=softening
        )
        t_bh = time.perf_counter() - t0
        if exact is not None and t_bh > 0:
            speedup = f"{t_exact / t_bh:>5.2f}x"
            rel_err_str = f"{_avg_relative_error(exact, bh):.2e}"
        else:
            speedup = "  n/a"
            rel_err_str = "n/a"
        print(
            f"  BH theta={theta:<3} : {t_bh * 1000:8.1f} ms  "
            f"speedup={speedup}  avg_rel_err={rel_err_str}"
        )
    print()


def main() -> None:
    print("Barnes-Hut vs exact direct-summation N-body acceleration\n")
    for n in (200, 1000, 2000, 5000):
        _bench(n)


if __name__ == "__main__":
    main()
