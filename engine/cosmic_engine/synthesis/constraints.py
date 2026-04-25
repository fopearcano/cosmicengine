"""Bounded, auditable constraint system for synthesized universes."""

from __future__ import annotations

from typing import Any

from cosmic_engine.core.units import SPEED_OF_LIGHT_M_S


class Constraint:
    """Base class. Subclasses implement :meth:`check` and :meth:`enforce`.

    ``check`` returns ``True`` if the state already satisfies the
    constraint (no work needed). ``enforce`` returns the (possibly
    clamped) state and a list of human-readable adjustment notes —
    nothing is silently rewritten without a note.
    """

    name: str = "constraint"

    def check(self, state: dict[str, Any]) -> bool:
        raise NotImplementedError

    def enforce(self, state: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
        """Default: no-op. Subclasses clamp + log."""
        return state, []


class MaxMassConstraint(Constraint):
    """Cap each object's ``mass_kg`` at ``limit_kg``."""

    def __init__(self, limit_kg: float) -> None:
        if limit_kg <= 0.0:
            raise ValueError("MaxMassConstraint.limit_kg must be positive")
        self.limit_kg = float(limit_kg)
        self.name = "max_mass_kg"

    def check(self, state: dict[str, Any]) -> bool:
        for obj in state.get("objects", []):
            m = obj.get("mass_kg")
            if m is not None and m > self.limit_kg:
                return False
        return True

    def enforce(self, state: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
        notes: list[str] = []
        for obj in state.get("objects", []):
            m = obj.get("mass_kg")
            if m is not None and m > self.limit_kg:
                notes.append(
                    f"clamped {obj.get('id', '?')}.mass_kg from {m:.3e} "
                    f"to {self.limit_kg:.3e} (max_mass_kg)"
                )
                obj["mass_kg"] = self.limit_kg
        return state, notes


class MaxVelocityConstraint(Constraint):
    """Cap each object's velocity magnitude at ``limit_m_s``.

    Defaults to a fraction of the speed of light so generated
    universes stay subluminal (rejecting `>= c` happens in
    :class:`UniverseObject.validate` already, but the synth path
    enforces a softer cap up front).
    """

    def __init__(self, limit_m_s: float) -> None:
        if limit_m_s <= 0.0:
            raise ValueError("MaxVelocityConstraint.limit_m_s must be positive")
        if limit_m_s >= SPEED_OF_LIGHT_M_S:
            raise ValueError(
                "MaxVelocityConstraint.limit_m_s must be subluminal"
            )
        self.limit_m_s = float(limit_m_s)
        self.name = "max_velocity_m_s"

    def check(self, state: dict[str, Any]) -> bool:
        for obj in state.get("objects", []):
            v = obj.get("velocity_m_s") or [0.0, 0.0, 0.0]
            speed = (v[0] ** 2 + v[1] ** 2 + v[2] ** 2) ** 0.5
            if speed > self.limit_m_s:
                return False
        return True

    def enforce(self, state: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
        notes: list[str] = []
        for obj in state.get("objects", []):
            v = obj.get("velocity_m_s") or [0.0, 0.0, 0.0]
            speed = (v[0] ** 2 + v[1] ** 2 + v[2] ** 2) ** 0.5
            if speed > self.limit_m_s:
                scale = self.limit_m_s / speed
                obj["velocity_m_s"] = [v[0] * scale, v[1] * scale, v[2] * scale]
                notes.append(
                    f"clamped {obj.get('id', '?')}.velocity from "
                    f"{speed:.3e} m/s to {self.limit_m_s:.3e} m/s"
                )
        return state, notes


def apply_constraints(
    state: dict[str, Any],
    constraints: list[Constraint],
) -> dict[str, Any]:
    """Run every ``constraint.enforce`` in order; aggregate notes.

    Mutates ``state["objects"]`` in place (the generator owns these
    dicts before they're handed to :class:`UniverseObject`) but also
    returns ``state`` for chained-call convenience. Every adjustment
    is appended to ``state["constraint_notes"]`` so the caller can
    surface the audit trail.
    """
    notes_log: list[str] = list(state.get("constraint_notes", []))
    for constraint in constraints:
        state, notes = constraint.enforce(state)
        notes_log.extend(notes)
    state["constraint_notes"] = notes_log
    return state
