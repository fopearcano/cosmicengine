# Provenance and truth integrity

CosmicEngine separates four categories of data + transformation:

1. **Observed** — direct measurement (Gaia, JPL ephemeris).
2. **Simulated** — output of deterministic physics on real inputs.
3. **AI-generated** — output of a neural surrogate or warp model.
4. **Symbolic / experimental** — overlays driven by reality rules
   (symbolic redshift coloring, relaxed causality, …).

The provenance layer makes those categories *inspectable* on every
rendered output. Nothing is hidden, nothing is silently mixed.

## Truth levels

`cosmic_engine.core.truth.TruthLevel` (string enum):

| Value | Origin |
|---|---|
| `observed` | Direct telescope measurement. |
| `catalog_imported` | Imported from a public catalog (Gaia DR3, SDSS, DESI). |
| `ephemeris_real` | High-precision ephemeris (JPL Horizons-style). |
| `physics_simulated` | Output of a physics step on real inputs. |
| `cosmological_model` | From a parameterised cosmology (e.g. ΛCDM). |
| `procedural_approximation` | One-off filler (synthetic galaxy field). |
| `ai_surrogate` | Output of a neural surrogate model. |
| `fictional_placeholder` | Clearly fake content for demos. |
| `synthetic_generated` | From a `UniverseSpec` via the synthesis layer. |

The first three (`observed`, `catalog_imported`, `ephemeris_real`)
are considered **trusted** by the audit layer.

## Provenance records

`cosmic_engine.provenance.ProvenanceRecord`:

```python
@dataclass
class ProvenanceRecord:
    entity_id: str
    source: str               # e.g. "gaia_dr3", "synthesized:standard_physics_42"
    truth_level: str          # mirrors TruthLevel
    transformations: list[str]  # ordered: pipeline steps + reality rules
    timestamp: float
    observer_id: str | None
```

`TruthTracker` is the in-memory dict of these. The runtime registers
every `UniverseObject` automatically (`add_objects`) and stamps
transformations as they run:

| Pipeline step | Transformation label |
|---|---|
| Physics step | `physics_<backend>` (e.g. `physics_leapfrog`) |
| Photon-field perception | `perception_warp` |
| AI warp model | `ai_warp` |
| Reality rule application | `reality_rule:<rule_id>` |
| Synthesis seeding | `synthesized:<spec_id>` |

Idempotency is built in: re-registering an entity never wipes its
transformation history.

## RealityView audit

Every render attaches:

```python
view.provenance_summary   # transformations, source_counts,
                          # truth_level_counts, tracked_entities,
                          # observed_entities, total_records
view.audit_warnings       # list[str] from detect_truth_mixing
```

`audit_reality_view(view)` produces a one-stop report:

```python
{
  "observer_id": "...",
  "truth_distribution": {"catalog_imported": 30, "synthetic_generated": 200, ...},
  "transformations": ["perception_warp", "ai_warp",
                      "reality_rule:warp_amplification", ...],
  "active_rules": ["warp_amplification", ...],
  "warnings": [...]
}
```

## What the audit detects

`detect_truth_mixing(view)` flags three deterministic conditions:

1. **AI-on-trusted without declaration.** An `ai_warp` /
   `neural_*` step ran on data whose `truth_level` is in
   `{observed, catalog_imported, ephemeris_real, measured}`, but
   `view.reality_metadata` doesn't carry
   `use_neural_perception=True`.

   *Example warning:*
   `"observed data passed through an AI transformation without
   use_neural_perception declared in reality_metadata"`

2. **Symbolic rule without color marker.** A symbolic rule
   (`redshift_symbolic_color`, `spectral_shift_blackhole`) is in
   `active_rule_ids`, but `reality_metadata` has no
   `color_mapping` set.

3. **Relaxed causality with visible events.**
   `reality_metadata['causality_mode'] == 'relaxed'` and
   `view.visible_event_count > 0` — visibility is no longer a
   strict light-cone fact.

These conditions are intentionally narrow; new ones land in
`provenance/audit.py` over time as new layers ship.

## Worked example

```python
from cosmic_engine.runtime import CosmicRuntime, RuntimeConfig
from cosmic_engine.observer import Observer
from cosmic_engine.reality import RealityRuleEngine, create_hypertravel_reality_preset
from cosmic_engine.provenance import audit_reality_view
from cosmic_engine.core.vector import Vector3
from cosmic_engine.rendering import SimpleCamera
from cosmic_engine.ai.base import AIWarpModel

class _Identity(AIWarpModel):
    def predict_direction(self, d, o): return d
    def predict_brightness(self, b, d, o): return b
    def predict_color(self, c, d, o): return c
    def confidence(self): return 1.0

rt = CosmicRuntime(config=RuntimeConfig(active_radius_m=1e30, max_active_objects=1000))
rt.load_sample_data()
obs = Observer(id="o", position_m=Vector3(0,-5e17,0),
               velocity_m_s=Vector3.zero(), forward=Vector3(0,1,0),
               up=Vector3(0,0,1), ai_warp_model=_Identity(),
               config={"output_ppm_path": "outputs/audit.ppm"})
rt.observer_manager.add_observer(obs)
cam = SimpleCamera(position_m=Vector3.zero(), forward=Vector3(0,1,0),
                   up=Vector3(0,0,1), fov_degrees=120, image_width=128, image_height=128)

# Scientific mode — AI step on observed Gaia data, no neural marker:
view = rt.render_for_observer("o", cam)
print(view.audit_warnings)
# ['observed data passed through an AI transformation without ...']

# Hypertravel preset — declares use_neural_perception:
rt.reality_rule_engine = RealityRuleEngine(create_hypertravel_reality_preset())
view = rt.render_for_observer("o", cam)
print(view.audit_warnings)
# []
```

## Adaptive feedback (Phase 37)

Drift between analytical and neural references is captured by
`AdaptiveEngine` (opt-in). Suggestions are emitted to
`view.adaptive_suggestions`, never auto-applied. See
[`MODULES.md`](MODULES.md#adaptive) for the full API.
