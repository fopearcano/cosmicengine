# Project: Cosmic AI Viewer

Goal:
Build a real-data, physics-grounded, AI-assisted cosmic perception engine.

Core rule:
AI may drive rendering/perception, but not replace truth metadata, coordinates, physics constraints, or source provenance.

Architecture:
- engine/: deterministic universe state
- data_pipeline/: catalog ingestion
- renderer/: photon-field and warp viewer
- ai/: neural renderer experiments
- apps/: runnable demos
- docs/: architecture notes

Coding rules:
- Small modules.
- Tests for every subsystem.
- No fake scientific certainty.
- Every generated object must carry truth_level and source.
- Keep approximations explicitly marked.
- Prefer simple working prototypes over giant abstractions.
- Never rewrite the whole repo without permission.