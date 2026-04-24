"""In-memory registry of :class:`UniverseObject` instances."""

from __future__ import annotations

from typing import Any

from cosmic_engine.core.object_types import CosmicObjectType
from cosmic_engine.core.universe_object import UniverseObject


class UniverseRegistry:
    """A keyed collection of universe objects.

    The registry is authoritative for object identity: IDs are unique,
    and lookups / listings / type queries all route through it.
    """

    def __init__(self) -> None:
        self._objects: dict[str, UniverseObject] = {}

    def add_object(self, obj: UniverseObject) -> None:
        """Insert ``obj``. Raises :class:`ValueError` on duplicate ID."""
        if obj.id in self._objects:
            raise ValueError(f"duplicate object id: {obj.id!r}")
        self._objects[obj.id] = obj

    def remove_object(self, object_id: str) -> None:
        """Remove the object with this ID, or do nothing if absent."""
        self._objects.pop(object_id, None)

    def get_object(self, object_id: str) -> UniverseObject | None:
        """Return the object with this ID, or ``None`` if not present."""
        return self._objects.get(object_id)

    def list_objects(self) -> list[UniverseObject]:
        """Return all registered objects as a list."""
        return list(self._objects.values())

    def query_by_type(
        self, object_type: CosmicObjectType
    ) -> list[UniverseObject]:
        """Return every registered object whose ``object_type`` matches."""
        return [o for o in self._objects.values() if o.object_type == object_type]

    def to_dict(self) -> dict[str, Any]:
        """Serialize the registry to a plain dict."""
        return {"objects": [o.to_dict() for o in self._objects.values()]}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> UniverseRegistry:
        """Rebuild a registry from its :meth:`to_dict` form."""
        registry = cls()
        for entry in data.get("objects", []):
            registry.add_object(UniverseObject.from_dict(entry))
        return registry
