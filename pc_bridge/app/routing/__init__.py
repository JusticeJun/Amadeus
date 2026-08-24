"""Semantic capability selection for Amadeus requests."""

from .base import CapabilityMatch, RouteDecision, RoutingRequest, SemanticRouter
from .defaults import create_default_semantic_router
from .pc_control_rules import (
    detect_conditional_pc_planning,
    matches_pc_control_request,
)
from .rule_based import RuleBasedSemanticRouter
from .weather_rules import matches_weather_request

__all__ = [
    "CapabilityMatch",
    "RouteDecision",
    "RoutingRequest",
    "RuleBasedSemanticRouter",
    "SemanticRouter",
    "create_default_semantic_router",
    "detect_conditional_pc_planning",
    "matches_pc_control_request",
    "matches_weather_request",
]
