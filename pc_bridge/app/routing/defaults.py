from __future__ import annotations

from ..pc_control.registry import AppRegistry
from ..music_control import MusicSemanticInterpreter
from .pc_control_rules import (
    detect_conditional_pc_planning,
    excludes_weather_lookup_for_pc_discussion,
    matches_conditional_weather_pc_request,
    matches_pc_control_request,
)
from .music_control_rules import (
    detect_conditional_music_planning,
    excludes_generic_pc_media_route,
    matches_conditional_weather_music_request,
    matches_music_control_request,
)
from .rule_based import RuleBasedSemanticRouter
from .music_fallback import MusicFallbackSemanticRouter
from .base import SemanticRouter
from .weather_rules import matches_weather_request


def create_default_semantic_router(
    apps: AppRegistry,
    music_interpreter: MusicSemanticInterpreter | None = None,
) -> SemanticRouter:
    router = RuleBasedSemanticRouter(
        {
            "weather": lambda request: (
                matches_weather_request(request)
                and not excludes_weather_lookup_for_pc_discussion(request)
            ) or matches_conditional_weather_pc_request(request, apps)
            or matches_conditional_weather_music_request(request),
            "pc_control": lambda request: (
                matches_pc_control_request(request, apps)
                and not excludes_generic_pc_media_route(request)
            ),
            "music_control": matches_music_control_request,
        },
        planning_detectors=(
            detect_conditional_pc_planning,
            detect_conditional_music_planning,
        ),
    )
    return MusicFallbackSemanticRouter(router, music_interpreter) if music_interpreter else router
