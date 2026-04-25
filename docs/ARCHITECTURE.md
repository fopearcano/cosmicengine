# Architecture

CosmicEngine is organized as a layered set of Python sub-packages
under `engine/cosmic_engine/`, with one optional client app under
`apps/ai_viewer/`. Every layer has a single responsibility and
imports only from layers below it.

## Layered overview

```
┌──────────────────────────────────────────────────────────────────┐
│ Layer 9 — distributed                                            │
│   Ray cluster bring-up, per-role entrypoints, health probes      │
├──────────────────────────────────────────────────────────────────┤
│ Layer 8 — runtime                                                │
│   CosmicRuntime, RuntimeServer, AI Viewer client                 │
├──────────────────────────────────────────────────────────────────┤
│ Layer 7 — meta layers (opt-in)                                   │
│   reality · provenance · adaptive · synthesis · multiscale       │
├──────────────────────────────────────────────────────────────────┤
│ Layer 6 — observer + time                                        │
│   Observer · ObserverManager · RealityView · proper time         │
│   EventStore · light-cone causality                              │
├──────────────────────────────────────────────────────────────────┤
│ Layer 5 — rendering + AI                                         │
│   photon · Gaussian splat · GPU/WebGPU · neural warp · density   │
├──────────────────────────────────────────────────────────────────┤
│ Layer 4 — perception                                             │
│   ObserverState · transform_photon_field · vectorized AI         │
├──────────────────────────────────────────────────────────────────┤
│ Layer 3 — physics + cosmos + streaming                           │
│   N-body · Barnes-Hut · GR · ΛCDM · spatial index · LOD          │
├──────────────────────────────────────────────────────────────────┤
│ Layer 2 — data                                                   │
│   Gaia / SDSS / DESI / JPL ingestion · synthetic galaxy field    │
├──────────────────────────────────────────────────────────────────┤
│ Layer 1 — core                                                   │
│   Vector3 · UniverseObject · TruthLevel · units · clock          │
└──────────────────────────────────────────────────────────────────┘
```

A higher layer **may** depend on any lower layer. A lower layer
**must not** import a higher one — broken in code review, not in
runtime checks.

## Data flow for one render

```
catalog CSV / synth spec
        │
        ▼
   data ingest  ──►  truth_level + source set on every UniverseObject
        │
        ▼
   UniverseRegistry  ──►  truth_tracker.register_entity(...)
        │
        ▼
   CosmicRuntime.step(dt)
        │      ├─ advance clock (coordinate_time_t)
        │      ├─ optional physics step (truth_tracker += "physics_<backend>")
        │      └─ advance every observer's proper_time_tau
        ▼
   render_for_observer(id, camera)
        │      ├─ select active objects (LOD + streaming + multiscale)
        │      ├─ build RuleContext, evaluate reality_rule_engine
        │      ├─ photon-field samples ──► perception transform
        │      │                            (warp factor, AI warp,
        │      │                             truth_tracker += "perception_warp" / "ai_warp")
        │      ├─ render PPM (or Gaussian splat / WebGPU)
        │      ├─ collect adaptive feedback, emit suggestions
        │      └─ run audit, attach warnings
        ▼
   RealityView(observer_id, scene_state, frame_data,
               metadata, reality_metadata, provenance_summary,
               audit_warnings, feedback_summary, adaptive_suggestions)
```

Every arrow above is deterministic; AI is invoked through clearly
labelled hooks (`AIWarpModel`, `SpacetimeFieldModel`, `ONNXDensityModel`).

## Cross-cutting concepts

### TruthLevel

A `TruthLevel` enum (in `cosmic_engine.core.truth`) tags every
`UniverseObject`:

| Level | Meaning |
|---|---|
| `OBSERVED` | direct telescope measurement |
| `CATALOG_IMPORTED` | imported from a public catalog (Gaia, SDSS, DESI) |
| `EPHEMERIS_REAL` | JPL or analogous high-precision ephemeris |
| `PHYSICS_SIMULATED` | output of a physics step on real-world inputs |
| `COSMOLOGICAL_MODEL` | from a parameterised cosmology (ΛCDM, etc.) |
| `PROCEDURAL_APPROXIMATION` | one-off filler, e.g. synthetic galaxy field |
| `AI_SURROGATE` | output of a neural surrogate model |
| `FICTIONAL_PLACEHOLDER` | clearly fake content |
| `SYNTHETIC_GENERATED` | from a `UniverseSpec` via the synthesis layer |

The provenance audit (`provenance.audit.detect_truth_mixing`) flags
combinations like *"AI step ran on observed data without a
`use_neural_perception` marker"*.

### RealityView

The single, structured payload every render produces. See
[`MODULES.md`](MODULES.md#observer) for the full field list. Every
optional feature in the engine surfaces through a `RealityView`
field — no hidden global state.

### Determinism

- All randomness is **seeded** (`numpy.random.default_rng(seed)`).
- All ordering (active rules, list_records, list_observers, etc.) is
  sorted by id or priority.
- Synthesis with the same seed produces byte-identical positions.
- Adaptive suggestions are emitted in a fixed order
  (model-first, rules-second).

### Opt-in design

Every advanced layer is opt-in:

- `runtime.scale_manager = None` → no multiscale
- `runtime.reality_rule_engine = None` → scientific mode
- `runtime.adaptive_engine = None` → no feedback collection
- `runtime.observer_manager` empty → single-observer behaviour

Single-observer scientific renders pay zero cost for these features.

## Where to read next

- Module-level reference: [`MODULES.md`](MODULES.md)
- Demo index: [`DEMOS.md`](DEMOS.md)
- Truth / audit deep dive: [`PROVENANCE.md`](PROVENANCE.md)
