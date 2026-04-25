"""Audit a :class:`RealityView` against its provenance history."""

from __future__ import annotations

from typing import Any


# Per-rule label prefix the runtime emits when a reality rule fires.
_RULE_PREFIX = "reality_rule:"

# Symbolic / experimental rule ids — appearance of any of these in the
# transformations list flags a "scientific output mixed with symbolic
# overlay" condition. Kept here (not in reality.rule) so this module
# stays free of cross-package coupling beyond strings.
_SYMBOLIC_RULE_IDS = frozenset({
    "redshift_symbolic_color",
    "spectral_shift_blackhole",
})

# Truth-level buckets. Names mirror cosmic_engine.core.truth.TruthLevel
# values; we do the comparison on plain strings so this module imports
# nothing from core.
_TRUSTED_TRUTH_LEVELS = frozenset({
    "observed",
    "measured",
    "catalog_imported",
    "ephemeris_real",
})
_AI_TRUTH_LEVELS = frozenset({"ai_surrogate", "ai_generated", "ai", "neural"})


def audit_reality_view(view) -> dict[str, Any]:
    """Return a structured audit report for ``view``.

    Keys:
      - ``observer_id``
      - ``truth_distribution``: counts of each truth_level the view's
        scene_state recorded.
      - ``transformations``: stable, deterministic list of every
        transformation referenced by either ``view.metadata`` (e.g.
        ``"reality_rule:..."`` derived from ``active_rule_ids``) or
        the runtime-attached provenance summary.
      - ``active_rules``: copy of ``view.active_rule_ids``.
      - ``warnings``: result of :func:`detect_truth_mixing`.
    """
    scene = getattr(view, "scene_state", None)
    truth_distribution: dict[str, int] = {}
    if scene is not None:
        truth_distribution = dict(getattr(scene, "truth_level_counts", {}) or {})

    transformations: list[str] = []
    seen: set[str] = set()

    # 1) Pipeline transformations recorded by the runtime
    #    (``provenance_summary`` in view.metadata).
    summary = (view.metadata or {}).get("provenance_summary") or {}
    for label in summary.get("transformations", []) or []:
        if label not in seen:
            transformations.append(label)
            seen.add(label)

    # 2) Reality rules applied for this render — encoded as
    #    ``reality_rule:<id>`` so audit consumers can grep one prefix.
    for rule_id in getattr(view, "active_rule_ids", []) or []:
        label = f"{_RULE_PREFIX}{rule_id}"
        if label not in seen:
            transformations.append(label)
            seen.add(label)

    return {
        "observer_id": getattr(view, "observer_id", None),
        "truth_distribution": truth_distribution,
        "transformations": transformations,
        "active_rules": list(getattr(view, "active_rule_ids", []) or []),
        "warnings": detect_truth_mixing(view),
    }


def detect_truth_mixing(view) -> list[str]:
    """Return human-readable warnings about ambiguous truth mixing.

    Conditions detected:

    1. AI transformation applied (``ai_warp``, ``neural_perception``,
       ``ai_density``, …) on a view whose scene contains *observed*
       data and whose ``reality_metadata`` doesn't explicitly mark the
       output as ``use_neural_perception``. Catches "observed data
       silently rendered through an AI step" cases.
    2. Symbolic reality rule applied without a corresponding
       ``reality_metadata['color_mapping']`` or analogous marker.
       Catches "symbolic overlay with no inspectable label".
    3. ``reality_metadata['causality_mode'] == 'relaxed'`` while the
       view still reports ``visible_event_count > 0`` (relaxed
       causality should be flagged so consumers don't trust event
       visibility as a hard physical statement).
    """
    warnings: list[str] = []
    summary = (view.metadata or {}).get("provenance_summary") or {}
    transformations = summary.get("transformations", []) or []
    reality_meta = getattr(view, "reality_metadata", {}) or {}
    truth_dist = {}
    scene = getattr(view, "scene_state", None)
    if scene is not None:
        truth_dist = getattr(scene, "truth_level_counts", {}) or {}

    has_ai_step = any(
        t.startswith("ai_") or "neural" in t for t in transformations
    )
    has_observed = any(
        truth_dist.get(level, 0) > 0 for level in _TRUSTED_TRUTH_LEVELS
    )
    declared_neural = bool(reality_meta.get("use_neural_perception"))
    if has_ai_step and has_observed and not declared_neural:
        warnings.append(
            "observed data passed through an AI transformation without "
            "use_neural_perception declared in reality_metadata"
        )

    # Symbolic rule applied without a color_mapping marker.
    active_rule_ids = list(getattr(view, "active_rule_ids", []) or [])
    symbolic_rules = [r for r in active_rule_ids if r in _SYMBOLIC_RULE_IDS]
    if symbolic_rules and not reality_meta.get("color_mapping"):
        warnings.append(
            f"symbolic rule(s) {symbolic_rules} active but no "
            "color_mapping marker in reality_metadata"
        )

    if (
        reality_meta.get("causality_mode") == "relaxed"
        and getattr(view, "visible_event_count", 0) > 0
    ):
        warnings.append(
            "causality_mode='relaxed' but visible_event_count > 0 — "
            "event visibility is no longer a strict light-cone fact"
        )

    return warnings
