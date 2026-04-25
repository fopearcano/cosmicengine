"""Per-entity provenance record."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ProvenanceRecord:
    """A traceable history of one entity's origin and transformations.

    ``entity_id`` is whatever upstream id the engine uses (a
    UniverseObject id, a photon sample id, a Gaussian point id, an
    Event id …). ``source`` is the data origin (``"gaia_dr3"``,
    ``"synthetic"``, ``"jpl_placeholder"``, …) and ``truth_level``
    mirrors the ``TruthLevel`` enum value (``"observed"``,
    ``"derived"``, ``"simulated"``, ``"synthetic"``, …) — kept as a
    plain string so this module never has to import the rest of the
    engine.

    ``transformations`` accumulates labels in application order
    (``"physics_nbody"``, ``"perception_warp"``, ``"ai_warp"``,
    ``"reality_rule:warp_amplification"``, …) so the audit layer can
    reconstruct the full pipeline that produced the entity's last
    state. ``observer_id`` records who the transformation was applied
    *for* — ``None`` for shared (registry-level) transformations.
    """

    entity_id: str
    source: str
    truth_level: str
    transformations: list[str] = field(default_factory=list)
    timestamp: float = 0.0
    observer_id: str | None = None

    def add_transformation(self, name: str) -> None:
        """Append ``name`` to the transformation list (preserves order)."""
        if not name:
            raise ValueError("transformation name must be a non-empty string")
        self.transformations.append(str(name))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly dict copy."""
        return {
            "entity_id": self.entity_id,
            "source": self.source,
            "truth_level": self.truth_level,
            "transformations": list(self.transformations),
            "timestamp": float(self.timestamp),
            "observer_id": self.observer_id,
        }
