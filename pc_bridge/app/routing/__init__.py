"""Semantic capability selection for Amadeus requests."""

from .base import CapabilityMatch, RouteDecision, RoutingRequest, SemanticRouter
from .rule_based import RuleBasedSemanticRouter
from .weather_rules import matches_weather_request

__all__ = [
    "CapabilityMatch",
    "RouteDecision",
    "RoutingRequest",
    "RuleBasedSemanticRouter",
    "SemanticRouter",
    "matches_weather_request",
]
