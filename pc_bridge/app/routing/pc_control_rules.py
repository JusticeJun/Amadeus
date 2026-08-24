from __future__ import annotations

import re

from ..pc_control.registry import AppRegistry
from .base import RoutingRequest


_LAUNCH_INTENT = (
    "켜줘", "켜봐", "열어줘", "열어봐", "실행해줘", "실행해봐", "실행시켜",
)
_KNOWN_UNSUPPORTED_APPS = ("포토샵",)
_CONDITIONAL_WEATHER = re.compile(r"(?:비|눈|날씨).{0,12}(?:오|내리|나쁘|좋).{0,5}면")


def matches_pc_control_request(request: RoutingRequest, apps: AppRegistry) -> bool:
    text = "".join(request.text.lower().split())
    launch_intent = any(intent in text for intent in _LAUNCH_INTENT)
    app_launch = launch_intent and (
        apps.resolve_alias(text) is not None
        or any(app in text for app in _KNOWN_UNSUPPORTED_APPS)
    )
    volume_action = ("볼륨" in text or "소리" in text) and any(word in text for word in (
        "올려", "높여", "키워", "크게", "내려", "낮춰", "줄여", "작게", "설정", "맞춰",
        "해줘",
    ))
    mute_action = "음소거" in text and any(word in text for word in (
        "해줘", "켜줘", "풀어", "해제", "시켜",
    ))
    media_target = any(word in text for word in (
        "다음곡", "다음노래", "다음트랙", "이전곡", "이전노래", "이전트랙",
    ))
    media_command = any(word in text for word in ("넘겨", "재생해", "틀어", "바꿔"))
    media_action = (media_target and (
        media_command or text in {
            "다음곡", "다음노래", "다음트랙", "이전곡", "이전노래", "이전트랙",
        }
    )) or bool(re.search(
        r"(?:(?:음악|노래)(?:을|를)?(?:일시정지|재생)|다시재생|재생멈춰)",
        text,
    ))
    return app_launch or volume_action or mute_action or media_action


def excludes_weather_lookup_for_pc_discussion(request: RoutingRequest) -> bool:
    """Reject a narrow comparison hard-negative without changing Weather rules."""
    compact = "".join(request.text.lower().split())
    return "날씨앱" in compact and any(word in compact for word in ("비교", "중에", "무거"))


def matches_conditional_weather_pc_request(
    request: RoutingRequest,
    apps: AppRegistry,
) -> bool:
    compact = "".join(request.text.lower().split())
    return bool(_CONDITIONAL_WEATHER.search(compact)) \
        and matches_pc_control_request(request, apps)


def detect_conditional_pc_planning(
    request: RoutingRequest,
    capabilities: frozenset[str],
) -> str | None:
    if {"weather", "pc_control"} <= capabilities:
        compact = "".join(request.text.lower().split())
        if _CONDITIONAL_WEATHER.search(compact):
            return "conditional_tool_dependency"
    return None
