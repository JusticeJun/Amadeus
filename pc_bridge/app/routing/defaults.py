from __future__ import annotations

from ..pc_control.registry import AppRegistry
from .pc_control_rules import (
    detect_conditional_pc_planning,
    excludes_weather_lookup_for_pc_discussion,
    matches_conditional_weather_pc_request,
    matches_pc_control_request,
)
from .rule_based import RuleBasedSemanticRouter
from .weather_rules import matches_weather_request


def create_default_semantic_router(apps: AppRegistry) -> RuleBasedSemanticRouter:
    return RuleBasedSemanticRouter(
        {
            "weather": lambda request: (
                matches_weather_request(request)
                and not excludes_weather_lookup_for_pc_discussion(request)
            ) or matches_conditional_weather_pc_request(request, apps),
            "pc_control": lambda request: matches_pc_control_request(request, apps),
        },
        planning_detectors=(detect_conditional_pc_planning,),
    )
