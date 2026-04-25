"""In-memory provenance registry."""

from __future__ import annotations

from cosmic_engine.provenance.provenance_record import ProvenanceRecord


class TruthTracker:
    """O(1) entity-id → :class:`ProvenanceRecord` lookup.

    Designed to be cheap to call from hot paths: every method is a
    dict op, none of them allocate when no work is needed, and the
    optional transformation-recording entry points silently no-op for
    unknown ids so callers never have to gate them with an existence
    check.
    """

    def __init__(self) -> None:
        self.records: dict[str, ProvenanceRecord] = {}

    def register_entity(
        self,
        entity_id: str,
        source: str,
        truth_level: str,
        *,
        timestamp: float = 0.0,
        observer_id: str | None = None,
    ) -> ProvenanceRecord:
        """Insert a fresh record. Re-registering an id is a no-op.

        We deliberately don't overwrite — re-registering should be
        cheap and idempotent so callers can safely register the whole
        registry on every load without dropping the transformation
        history.
        """
        if not entity_id:
            raise ValueError("entity_id must be a non-empty string")
        existing = self.records.get(entity_id)
        if existing is not None:
            return existing
        record = ProvenanceRecord(
            entity_id=entity_id,
            source=source,
            truth_level=truth_level,
            timestamp=float(timestamp),
            observer_id=observer_id,
        )
        self.records[entity_id] = record
        return record

    def add_transformation(self, entity_id: str, transformation: str) -> bool:
        """Append a transformation to ``entity_id``'s record.

        Returns ``True`` if the entity was known and the
        transformation was appended, ``False`` otherwise. We return
        a bool rather than raising so hot-path callers can fire-and-
        forget without wrapping every call in a try/except.
        """
        record = self.records.get(entity_id)
        if record is None:
            return False
        record.add_transformation(transformation)
        return True

    def add_transformation_to_all(self, transformation: str) -> int:
        """Append ``transformation`` to *every* tracked record.

        Returns the number of records touched. Useful for pipeline
        steps (e.g. ``"physics_nbody"``) that affect the whole active
        set in one go.
        """
        for record in self.records.values():
            record.add_transformation(transformation)
        return len(self.records)

    def get_record(self, entity_id: str) -> ProvenanceRecord | None:
        return self.records.get(entity_id)

    def list_records(self) -> list[ProvenanceRecord]:
        """Return all records sorted by entity_id for determinism."""
        return [self.records[i] for i in sorted(self.records)]

    def __len__(self) -> int:
        return len(self.records)

    def __contains__(self, entity_id: object) -> bool:
        return entity_id in self.records
