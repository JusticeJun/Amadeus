from __future__ import annotations

from collections.abc import Callable, Mapping

from .base import CapabilityMatch, RouteDecision, RoutingRequest


RuleMatcher = Callable[[RoutingRequest], bool]
PlanningDetector = Callable[[RoutingRequest, frozenset[str]], str | None]


class RuleBasedSemanticRouter:
    """Select capabilities with deterministic local semantic rules."""

    def __init__(
        self,
        rules: Mapping[str, RuleMatcher] | None = None,
        planning_detectors: tuple[PlanningDetector, ...] = (),
    ) -> None:
        self._rules = dict(rules or {})
        self._planning_detectors = planning_detectors

    def route(self, request: RoutingRequest) -> RouteDecision:
        matches = tuple(
            CapabilityMatch(capability)
            for capability, matcher in self._rules.items()
            if matcher(request)
        )
        capabilities = frozenset(match.capability for match in matches)
        planning_reason = next((
            reason
            for detector in self._planning_detectors
            if (reason := detector(request, capabilities))
        ), "")
        return RouteDecision(
            matches,
            planning_required=bool(planning_reason),
            planning_reason=planning_reason,
        )
