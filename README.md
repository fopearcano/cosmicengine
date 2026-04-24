# CosmicEngine

CosmicEngine is a data-driven, physics-grounded, AI-assisted cosmic perception engine.

## Future modules

- `engine` — deterministic universe state
- `renderer` — photon-field and warp viewer
- `ai` — neural renderer experiments
- `data_pipeline` — catalog ingestion

## Current status

- Phase 1 core object model implemented (`TruthLevel`, `CosmicObjectType`, `Vector3`, `UniverseObject`, `UniverseRegistry`).
- Time system (`SimulationClock`, Julian-Date helpers), units, and RA/Dec ↔ Cartesian coordinates.
- No simulation, rendering, or AI yet.

## Phase 2: Data ingestion

- Basic CSV star catalog loader implemented (`cosmic_engine.data`).
- Supports RA/Dec → Cartesian conversion and distance in light-years.
- Registry integration available via `load_star_catalog_into_registry`.
- Sample catalog at `data/sample_stars.csv`; demo at `examples/load_star_catalog_demo.py`.
