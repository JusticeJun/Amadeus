"""Semantic capability selection for Amadeus requests."""

from .base import CapabilityMatch, RouteDecision, RoutingRequest, SemanticRouter
from .capabilities import CAPABILITIES, CAPABILITY_NAMES, CapabilityDefinition
from .defaults import create_default_semantic_router, create_rule_based_semantic_router
from .local_ml import LocalMlSemanticRouter, ModelArtifactError
from .sentence_ml import SentenceMlSemanticRouter
from .pc_control_rules import (
    detect_conditional_pc_planning,
    matches_pc_control_request,
)
from .music_control_rules import (
    detect_conditional_music_planning,
    matches_music_control_request,
)
from .rule_based import RuleBasedSemanticRouter
from .weather_rules import matches_weather_request

__all__ = [
    "CapabilityMatch",
    "CapabilityDefinition",
    "CAPABILITIES",
    "CAPABILITY_NAMES",
    "LocalMlSemanticRouter",
    "SentenceMlSemanticRouter",
    "ModelArtifactError",
    "RouteDecision",
    "RoutingRequest",
    "RuleBasedSemanticRouter",
    "SemanticRouter",
    "create_default_semantic_router",
    "create_rule_based_semantic_router",
    "detect_conditional_pc_planning",
    "detect_conditional_music_planning",
    "matches_music_control_request",
    "matches_pc_control_request",
    "matches_weather_request",
]
