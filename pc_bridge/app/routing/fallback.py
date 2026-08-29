from __future__ import annotations

from .base import RouteDecision, RoutingRequest, SemanticRouter


class NoMatchFallbackSemanticRouter:
    """Use a secondary semantic router only when the safe fast path has no match."""

    def __init__(self, primary: SemanticRouter, fallback: SemanticRouter) -> None:
        self._primary = primary
        self._fallback = fallback

    def route(self, request: RoutingRequest) -> RouteDecision:
        primary = self._primary.route(request)
        return primary if primary.matches or primary.planning_required else self._fallback.route(request)

