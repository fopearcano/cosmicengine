"""Deterministic reality-rule evaluator."""

from __future__ import annotations

from cosmic_engine.reality.rule import RealityRule
from cosmic_engine.reality.rule_context import RuleContext


class RealityRuleEngine:
    """Apply a deterministic, priority-ordered chain of rules to a context."""

    def __init__(self, rules: list[RealityRule] | None = None) -> None:
        self.rules: list[RealityRule] = []
        if rules:
            for r in rules:
                self.add_rule(r)

    def add_rule(self, rule: RealityRule) -> None:
        """Append a rule. Rejects duplicate ids to keep evaluation deterministic."""
        if any(r.id == rule.id for r in self.rules):
            raise ValueError(
                f"RealityRuleEngine already has a rule with id {rule.id!r}"
            )
        self.rules.append(rule)

    def remove_rule(self, rule_id: str) -> None:
        """Drop ``rule_id``; raises :class:`KeyError` if unknown."""
        for i, r in enumerate(self.rules):
            if r.id == rule_id:
                del self.rules[i]
                return
        raise KeyError(rule_id)

    def list_rules(self, enabled_only: bool = False) -> list[RealityRule]:
        """Return rules sorted by ``(-priority, id)`` for deterministic order."""
        rules = (
            [r for r in self.rules if r.enabled]
            if enabled_only
            else list(self.rules)
        )
        # Higher priority applies first; ties broken by id for stability.
        return sorted(rules, key=lambda r: (-int(r.priority), r.id))

    def evaluate(self, context: RuleContext) -> RuleContext:
        """Apply enabled rules in priority order; record applied ids."""
        ordered = self.list_rules(enabled_only=True)
        current = context.clone()
        for rule in ordered:
            if not rule.applies_to_context(current):
                continue
            new_ctx = rule.apply(current)
            if rule.id not in new_ctx.active_rule_ids:
                new_ctx.active_rule_ids = list(new_ctx.active_rule_ids) + [rule.id]
            current = new_ctx
        return current

    def __len__(self) -> int:
        return len(self.rules)
