from __future__ import annotations

from collections.abc import Callable, Mapping

from .base import CapabilityMatch, RouteDecision, RoutingRequest


RuleMatcher = Callable[[RoutingRequest], bool]


class RuleBasedSemanticRouter:
    """Select capabilities with deterministic local semantic rules."""

    def __init__(self, rules: Mapping[str, RuleMatcher] | None = None) -> None:
        self._rules = dict(rules or {})

    def route(self, request: RoutingRequest) -> RouteDecision:
        matches = tuple(
            CapabilityMatch(capability)
            for capability, matcher in self._rules.items()
            if matcher(request)
        )
        return RouteDecision(matches)
