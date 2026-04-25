"""Provenance and truth-integrity audit layer.

Phase 36 contribution: track *how* every rendered result was produced
so consumers can tell observed catalog data, simulated physics,
AI-generated approximations, and symbolic / experimental
transformations apart at a glance.

Lightweight by design — all storage is in-memory dictionaries; no
external dependencies; no heavyweight logging.
"""

from cosmic_engine.provenance.audit import (
    audit_reality_view,
    detect_truth_mixing,
)
from cosmic_engine.provenance.provenance_record import ProvenanceRecord
from cosmic_engine.provenance.truth_tracker import TruthTracker

__all__ = [
    "ProvenanceRecord",
    "TruthTracker",
    "audit_reality_view",
    "detect_truth_mixing",
]
