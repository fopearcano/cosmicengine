"""Truth / provenance levels for universe objects.

Every generated or imported object in CosmicEngine MUST declare where its
data comes from. AI-generated or procedurally-approximated data must never
be indistinguishable from observed or ephemeris-derived data.

The level answers a simple question: how much do we trust this?
"""

from enum import Enum


class TruthLevel(str, Enum):
    """Provenance level of a universe object's data.

    Values are ordered roughly from "most grounded in reality" to
    "least grounded" but the ordering carries no numeric meaning — it
    is only a classification. Every :class:`UniverseObject` must carry
    one of these so downstream systems can gate behavior on truth.
    """

    OBSERVED = "observed"
    CATALOG_IMPORTED = "catalog_imported"
    EPHEMERIS_REAL = "ephemeris_real"
    PHYSICS_SIMULATED = "physics_simulated"
    COSMOLOGICAL_MODEL = "cosmological_model"
    PROCEDURAL_APPROXIMATION = "procedural_approximation"
    AI_SURROGATE = "ai_surrogate"
    FICTIONAL_PLACEHOLDER = "fictional_placeholder"
    # Phase 38: an entire universe synthesized from a UniverseSpec.
    # Distinct from PROCEDURAL_APPROXIMATION (one-off filler galaxies)
    # so audit consumers can flag whole-universe synthetic content.
    SYNTHETIC_GENERATED = "synthetic_generated"
