"""Newtonian N-body integrator for local gravitational systems.

What this is:
- A vectorized direct-summation Newtonian gravity solver: O(N²) per step.
- Two integrators — Euler (testing only) and a kick-drift-kick leapfrog
  (preferred for orbital stability).
- Optional Plummer-style softening to suppress close-encounter singularities.

What this is **not**:
- A galaxy-scale simulator (no tree code / FMM / mesh).
- Relativistic.
- GPU-accelerated.

The Keplerian orbital module in :mod:`cosmic_engine.physics.orbital`
remains the analytic reference for two-body cases; this module is for
many-body experiments and time-stepped integration.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cosmic_engine.core.universe_object import UniverseObject
from cosmic_engine.core.vector import Vector3


# Gravitational constant in SI: m³ kg⁻¹ s⁻².
GRAVITATIONAL_CONSTANT: float = 6.67430e-11


@dataclass
class NBodyState:
    """All state needed for a Newtonian N-body step.

    Arrays must align by index with ``object_ids``.
    """

    object_ids: list[str]
    positions_m: np.ndarray
    velocities_m_s: np.ndarray
    masses_kg: np.ndarray

    def __len__(self) -> int:
        return len(self.object_ids)

    def validate(self) -> None:
        """Raise :class:`ValueError` if the arrays are inconsistent or invalid."""
        n = len(self.object_ids)
        if n < 1:
            raise ValueError("NBodyState must contain at least one body")
        if self.positions_m.shape != (n, 3):
            raise ValueError(
                f"positions_m must have shape ({n}, 3); "
                f"got {self.positions_m.shape}"
            )
        if self.velocities_m_s.shape != (n, 3):
            raise ValueError(
                f"velocities_m_s must have shape ({n}, 3); "
                f"got {self.velocities_m_s.shape}"
            )
        if self.masses_kg.shape != (n,):
            raise ValueError(
                f"masses_kg must have shape ({n},); got {self.masses_kg.shape}"
            )
        if (self.masses_kg < 0.0).any():
            raise ValueError("masses_kg must be non-negative")


def objects_to_nbody_state(
    objects: list[UniverseObject],
) -> NBodyState:
    """Build an :class:`NBodyState` from massive :class:`UniverseObject` records.

    Objects with ``mass_kg is None`` are dropped. Raises :class:`ValueError`
    if fewer than two massive bodies remain.
    """
    valid = [o for o in objects if o.mass_kg is not None]
    if len(valid) < 2:
        raise ValueError(
            "objects_to_nbody_state requires at least 2 massive bodies"
        )
    ids = [o.id for o in valid]
    positions = np.array(
        [[o.position_m.x, o.position_m.y, o.position_m.z] for o in valid],
        dtype=np.float64,
    )
    velocities = np.array(
        [[o.velocity_m_s.x, o.velocity_m_s.y, o.velocity_m_s.z] for o in valid],
        dtype=np.float64,
    )
    masses = np.array([o.mass_kg for o in valid], dtype=np.float64)
    state = NBodyState(
        object_ids=ids,
        positions_m=positions,
        velocities_m_s=velocities,
        masses_kg=masses,
    )
    state.validate()
    return state


def apply_nbody_state_to_objects(
    objects: list[UniverseObject],
    state: NBodyState,
) -> list[UniverseObject]:
    """Mutate ``objects`` in place with positions / velocities from ``state``.

    Matching is by ``UniverseObject.id``; objects whose ID is not in
    ``state.object_ids`` are left untouched. Returns the same list for
    chaining.
    """
    by_id: dict[str, tuple[np.ndarray, np.ndarray]] = {
        state.object_ids[i]: (state.positions_m[i], state.velocities_m_s[i])
        for i in range(len(state))
    }
    for obj in objects:
        entry = by_id.get(obj.id)
        if entry is None:
            continue
        position, velocity = entry
        obj.position_m = Vector3(
            float(position[0]), float(position[1]), float(position[2])
        )
        obj.velocity_m_s = Vector3(
            float(velocity[0]), float(velocity[1]), float(velocity[2])
        )
    return objects


def compute_accelerations(
    positions_m: np.ndarray,
    masses_kg: np.ndarray,
    softening_m: float = 0.0,
) -> np.ndarray:
    """Newtonian acceleration on each body due to all others.

    Returns an ``(N, 3)`` array. The diagonal self-interaction is
    suppressed by setting the corresponding pairwise distance to
    infinity, so no NaNs arise even with ``softening_m == 0``.
    """
    n = len(masses_kg)
    if positions_m.shape != (n, 3):
        raise ValueError(
            f"positions_m must have shape ({n}, 3); got {positions_m.shape}"
        )
    if softening_m < 0.0:
        raise ValueError(f"softening_m must be non-negative; got {softening_m}")

    # Pairwise displacement r_j − r_i with shape (N, N, 3).
    diff = positions_m[None, :, :] - positions_m[:, None, :]
    dist_sq = np.sum(diff * diff, axis=-1) + softening_m * softening_m

    # Suppress self-interaction so 1/r³ on the diagonal stays at 0.
    np.fill_diagonal(dist_sq, np.inf)
    inv_r3 = dist_sq ** -1.5

    weights = (masses_kg[None, :] * inv_r3)[:, :, None]  # (N, N, 1)
    accel = (weights * diff).sum(axis=1) * GRAVITATIONAL_CONSTANT
    return accel


def _validate_dt(dt_seconds: float) -> None:
    if dt_seconds <= 0.0:
        raise ValueError(f"dt_seconds must be positive; got {dt_seconds}")


def euler_step(
    state: NBodyState,
    dt_seconds: float,
    softening_m: float = 0.0,
) -> NBodyState:
    """Forward-Euler step. Use this for testing only — not symplectic."""
    _validate_dt(dt_seconds)
    accel = compute_accelerations(
        state.positions_m, state.masses_kg, softening_m
    )
    new_positions = state.positions_m + state.velocities_m_s * dt_seconds
    new_velocities = state.velocities_m_s + accel * dt_seconds
    return NBodyState(
        object_ids=list(state.object_ids),
        positions_m=new_positions,
        velocities_m_s=new_velocities,
        masses_kg=state.masses_kg.copy(),
    )


def leapfrog_step(
    state: NBodyState,
    dt_seconds: float,
    softening_m: float = 0.0,
) -> NBodyState:
    """Velocity-Verlet (kick-drift-kick) leapfrog step. Symplectic."""
    _validate_dt(dt_seconds)
    accel = compute_accelerations(
        state.positions_m, state.masses_kg, softening_m
    )
    v_half = state.velocities_m_s + 0.5 * dt_seconds * accel
    new_positions = state.positions_m + dt_seconds * v_half
    accel_new = compute_accelerations(
        new_positions, state.masses_kg, softening_m
    )
    new_velocities = v_half + 0.5 * dt_seconds * accel_new
    return NBodyState(
        object_ids=list(state.object_ids),
        positions_m=new_positions,
        velocities_m_s=new_velocities,
        masses_kg=state.masses_kg.copy(),
    )


_INTEGRATORS = {
    "euler": euler_step,
    "leapfrog": leapfrog_step,
}


class NBodySimulator:
    """Stateful driver around the integrator functions."""

    def __init__(
        self,
        state: NBodyState,
        softening_m: float = 0.0,
        integrator: str = "leapfrog",
    ) -> None:
        if integrator not in _INTEGRATORS:
            raise ValueError(
                f"unknown integrator {integrator!r}; "
                f"expected one of {sorted(_INTEGRATORS)}"
            )
        if softening_m < 0.0:
            raise ValueError(
                f"softening_m must be non-negative; got {softening_m}"
            )
        state.validate()
        self.state = state
        self.softening_m = softening_m
        self.integrator = integrator

    def step(self, dt_seconds: float) -> NBodyState:
        """Advance the system by one step and return the new state."""
        self.state = _INTEGRATORS[self.integrator](
            self.state, dt_seconds, self.softening_m
        )
        return self.state

    def run(self, steps: int, dt_seconds: float) -> NBodyState:
        """Advance the system by ``steps`` steps of length ``dt_seconds``."""
        if steps < 0:
            raise ValueError(f"steps must be non-negative; got {steps}")
        for _ in range(steps):
            self.step(dt_seconds)
        return self.state

    def get_state(self) -> NBodyState:
        """Return the current :class:`NBodyState`."""
        return self.state
