from __future__ import annotations

from ..music_control import MusicSemanticInterpreter
from .base import CapabilityMatch, RouteDecision, RoutingRequest, SemanticRouter
from .music_control_rules import detect_conditional_music_planning


class MusicFallbackSemanticRouter:
    """Add only Music capability misses resolved by the Music interpreter."""

    def __init__(self, base: SemanticRouter, interpreter: MusicSemanticInterpreter) -> None:
        self._base = base
        self._interpreter = interpreter

    def route(self, request: RoutingRequest) -> RouteDecision:
        decision = self._base.route(request)
        if "music_control" in decision.required_capabilities or not self._interpreter.is_candidate(
            request.text, request.history,
        ):
            return decision
        result = self._interpreter.interpret(request.text, request.history)
        if result.error_code == "not_music":
            return decision
        matches = decision.matches + (CapabilityMatch("music_control"),)
        capabilities = frozenset(match.capability for match in matches)
        reason = decision.planning_reason or detect_conditional_music_planning(
            request, capabilities,
        ) or ""
        return RouteDecision(matches, bool(reason), reason)
