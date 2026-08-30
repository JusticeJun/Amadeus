from __future__ import annotations

from .base import RouteDecision, RoutingRequest, SemanticRouter
from .rule_based import PlanningDetector


class PlanningGuardSemanticRouter:
    """Add execution-safety planning state without changing ML predictions."""

    def __init__(self, base: SemanticRouter, detectors: tuple[PlanningDetector, ...]) -> None:
        self._base = base
        self._detectors = detectors

    def route(self, request: RoutingRequest) -> RouteDecision:
        decision = self._base.route(request)
        reason = next((
            reason for detector in self._detectors
            if (reason := detector(request, decision.required_capabilities))
        ), "")
        return RouteDecision(decision.matches, bool(reason), reason)


class CapabilityFilterSemanticRouter:
    """Apply an explicit risk policy to predictions before execution routing."""

    def __init__(self, base: SemanticRouter, allowed: frozenset[str]) -> None:
        self._base = base
        self._allowed = allowed

    def route(self, request: RoutingRequest) -> RouteDecision:
        decision = self._base.route(request)
        matches = tuple(match for match in decision.matches if match.capability in self._allowed)
        return RouteDecision(matches, decision.planning_required, decision.planning_reason)
