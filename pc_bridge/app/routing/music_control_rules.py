from __future__ import annotations

import re

from .base import RoutingRequest


_WEATHER_CONDITION = re.compile(
    r"(?:비|눈|날씨|기온|온도|밖|바깥).{0,30}(?:면|하면|으면|일때|인경우|넘으면)",
)


def matches_music_control_request(request: RoutingRequest) -> bool:
    text = "".join(request.text.lower().split())
    play_request = any(word in text for word in (
        "틀어줘", "틀어줄래", "재생해줘", "재생해줄래", "일시정지해줘", "잠깐멈춰",
        "일시정지좀해줘", "재생멈춰", "넘겨줘", "돌아가줘",
    ))
    query = any(word in text for word in (
        "내플레이리스트뭐있", "플레이리스트목록", "지금무슨곡",
        "지금무슨노래", "지금뭐재생", "현재재생곡",
    ))
    unsupported_library_mutation = "플레이리스트" in text and any(
        word in text for word in ("추가해줘", "삭제해줘")
    )
    short_transport = text in {"다음곡", "다음노래", "이전곡", "이전노래"}
    return query or play_request or short_transport or unsupported_library_mutation


def excludes_generic_pc_media_route(request: RoutingRequest) -> bool:
    compact = "".join(request.text.lower().split())
    pc_specific = any(word in compact for word in (
        "볼륨", "소리", "음소거", "켜줘", "열어줘", "띄워줘", "실행해줘",
    ))
    return matches_music_control_request(request) and not pc_specific


def matches_conditional_weather_music_request(request: RoutingRequest) -> bool:
    compact = "".join(request.text.lower().split())
    return bool(_WEATHER_CONDITION.search(compact)) and matches_music_control_request(request)


def detect_conditional_music_planning(
    request: RoutingRequest,
    capabilities: frozenset[str],
) -> str | None:
    compact = "".join(request.text.lower().split())
    if "music_control" in capabilities and _WEATHER_CONDITION.search(compact):
        return "conditional_tool_dependency"
    return None
