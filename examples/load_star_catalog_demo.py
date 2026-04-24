"""Demo: load the bundled sample star catalog into a registry."""

from __future__ import annotations

from pathlib import Path
from pprint import pprint

from cosmic_engine.core.registry import UniverseRegistry
from cosmic_engine.data.star_catalog import load_star_catalog_into_registry

_REPO_ROOT = Path(__file__).resolve().parent.parent
_CATALOG = _REPO_ROOT / "data" / "sample_stars.csv"


def main() -> None:
    registry = UniverseRegistry()
    load_star_catalog_into_registry(str(_CATALOG), registry)

    objects = registry.list_objects()
    print(f"total objects: {len(objects)}")
    print("first 3 objects:")
    for obj in objects[:3]:
        pprint(obj.to_dict())


if __name__ == "__main__":
    main()
