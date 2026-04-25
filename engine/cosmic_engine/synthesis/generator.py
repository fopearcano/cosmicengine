"""Deterministic universe generator from a :class:`UniverseSpec`."""

from __future__ import annotations

from typing import Any

import numpy as np

from cosmic_engine.core.object_types import CosmicObjectType
from cosmic_engine.core.truth import TruthLevel
from cosmic_engine.core.universe_object import UniverseObject
from cosmic_engine.core.vector import Vector3
from cosmic_engine.synthesis.constraints import (
    Constraint,
    MaxMassConstraint,
    MaxVelocityConstraint,
    apply_constraints,
)
from cosmic_engine.synthesis.universe_spec import UniverseSpec


_DEFAULT_OBJECT_COUNT = 64
_DEFAULT_MASS_RANGE = (1.0e22, 1.0e30)
_DEFAULT_VELOCITY_DISP = 1.0e3


class UniverseGenerator:
    """Build deterministic initial state + a ready :class:`CosmicRuntime`."""

    def __init__(self, spec: UniverseSpec) -> None:
        spec.validate()
        self.spec: UniverseSpec = spec

    # --- initial state ----------------------------------------------------

    def generate_initial_state(self) -> dict[str, Any]:
        """Return a dict of plain values describing the seeded universe.

        Output shape::

            {
                "spec_id": str,
                "seed": int,
                "objects": [{
                    "id": str, "mass_kg": float,
                    "position_m": [x, y, z], "velocity_m_s": [vx, vy, vz],
                    "object_type": "star" | "galaxy" | ...,
                }, ...],
                "constraint_notes": [str, ...],   # filled by apply_constraints
                "rule_ids": [str, ...],
            }

        The dict is intentionally JSON-serializable so a synth state
        can be saved / replayed without any engine imports.
        """
        rng = np.random.default_rng(int(self.spec.seed))
        ic = self.spec.initial_conditions

        n = int(ic.get("object_count", _DEFAULT_OBJECT_COUNT))
        if n < 0:
            raise ValueError("object_count must be >= 0")
        mass_range = tuple(ic.get("mass_range_kg", _DEFAULT_MASS_RANGE))
        v_disp = float(ic.get("velocity_dispersion_m_s", _DEFAULT_VELOCITY_DISP))
        radius_lo, radius_hi = self.spec.scale_limits
        include_central = bool(ic.get("include_central_mass", True))
        central_mass = float(ic.get("central_mass_kg", 1.989e30))
        object_type_label = ic.get("object_type", "star")

        try:
            object_type = CosmicObjectType(object_type_label)
        except ValueError:
            object_type = CosmicObjectType.STAR

        objects: list[dict[str, Any]] = []

        if include_central:
            objects.append({
                "id": f"{self.spec.id}__center",
                "mass_kg": central_mass,
                "position_m": [0.0, 0.0, 0.0],
                "velocity_m_s": [0.0, 0.0, 0.0],
                "object_type": object_type.value,
            })

        # Uniform-in-log radii on a 3D ball + isotropic Gaussian
        # velocity. Vectorized with a single rng.standard_normal /
        # rng.uniform pair so the seed maps deterministically to
        # arrays.
        if n > 0:
            log_lo = np.log(radius_lo)
            log_hi = np.log(radius_hi)
            radii = np.exp(rng.uniform(log_lo, log_hi, size=n))
            directions = rng.standard_normal(size=(n, 3))
            dir_norms = np.linalg.norm(directions, axis=1, keepdims=True)
            dir_norms = np.where(dir_norms == 0.0, 1.0, dir_norms)
            unit_dirs = directions / dir_norms
            positions = unit_dirs * radii[:, None]
            velocities = rng.standard_normal(size=(n, 3)) * v_disp
            masses = rng.uniform(mass_range[0], mass_range[1], size=n)

            for i in range(n):
                objects.append({
                    "id": f"{self.spec.id}__obj_{i:05d}",
                    "mass_kg": float(masses[i]),
                    "position_m": positions[i].tolist(),
                    "velocity_m_s": velocities[i].tolist(),
                    "object_type": object_type.value,
                })

        state: dict[str, Any] = {
            "spec_id": self.spec.id,
            "seed": int(self.spec.seed),
            "objects": objects,
            "constraint_notes": [],
            "rule_ids": list(self.spec.rule_ids),
        }
        # Apply spec-driven constraints up front so post-conditions
        # hold before we hand state to the registry.
        constraints = self._build_constraints()
        if constraints:
            apply_constraints(state, constraints)
        return state

    # --- runtime ---------------------------------------------------------

    def generate_runtime(self):
        """Build a :class:`CosmicRuntime`, populate it, and attach rules.

        The synthetic objects are tagged
        ``TruthLevel.SYNTHETIC_GENERATED`` and ``source = spec.id`` so
        the existing provenance / audit pipeline picks them up
        automatically. Rules referenced by ``spec.rule_ids`` are
        resolved against
        :func:`cosmic_engine.synthesis.presets._resolve_rule` and
        attached via a fresh :class:`RealityRuleEngine`.
        """
        # Imported here (not at module top) so importing the synthesis
        # package never drags in the rest of the engine.
        from cosmic_engine.reality import RealityRuleEngine
        from cosmic_engine.runtime import CosmicRuntime, RuntimeConfig
        from cosmic_engine.synthesis.presets import _resolve_rule

        state = self.generate_initial_state()

        runtime = CosmicRuntime(
            config=RuntimeConfig(
                enable_physics=False,
                enable_perception=False,
                active_radius_m=max(self.spec.scale_limits) * 100.0,
                max_active_objects=max(len(state["objects"]), 1),
            )
        )

        objects: list[UniverseObject] = []
        for entry in state["objects"]:
            try:
                ot = CosmicObjectType(entry["object_type"])
            except ValueError:
                ot = CosmicObjectType.STAR
            position = entry["position_m"]
            velocity = entry["velocity_m_s"]
            obj = UniverseObject(
                id=entry["id"],
                name=entry["id"],
                object_type=ot,
                position_m=Vector3(
                    float(position[0]), float(position[1]), float(position[2])
                ),
                velocity_m_s=Vector3(
                    float(velocity[0]), float(velocity[1]), float(velocity[2])
                ),
                truth_level=TruthLevel.SYNTHETIC_GENERATED,
                source=self.spec.id,
                mass_kg=float(entry["mass_kg"]),
                metadata={
                    "synthetic": True,
                    "synth_seed": int(self.spec.seed),
                    "synth_spec_id": self.spec.id,
                },
            )
            objects.append(obj)
        runtime.add_objects(objects)

        # Stamp every record with the spec id so the audit trail can
        # filter "this came from synthesis" with one transformation
        # label.
        runtime.truth_tracker.add_transformation_to_all(
            f"synthesized:{self.spec.id}"
        )

        # Wire up reality rules referenced by the spec.
        if self.spec.rule_ids:
            rules = []
            for rid in self.spec.rule_ids:
                resolved = _resolve_rule(rid)
                if resolved is not None:
                    rules.append(resolved)
            if rules:
                runtime.reality_rule_engine = RealityRuleEngine(rules)

        # Stash the generated state on the runtime so the demo /
        # tests can inspect it without re-running the generator.
        runtime.synthesis_state = state  # dynamic attribute
        runtime.synthesis_spec = self.spec  # dynamic attribute
        return runtime

    # --- helpers ---------------------------------------------------------

    def _build_constraints(self) -> list[Constraint]:
        out: list[Constraint] = []
        c = self.spec.constraints
        if "max_mass_kg" in c:
            out.append(MaxMassConstraint(float(c["max_mass_kg"])))
        if "max_velocity_m_s" in c:
            out.append(MaxVelocityConstraint(float(c["max_velocity_m_s"])))
        return out


def create_runtime_from_spec(spec: UniverseSpec):
    """Convenience: ``UniverseGenerator(spec).generate_runtime()``."""
    return UniverseGenerator(spec).generate_runtime()
