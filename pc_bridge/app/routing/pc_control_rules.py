from __future__ import annotations

import re

from ..pc_control.registry import AppRegistry
from .base import RoutingRequest


_LAUNCH_INTENT = (
    "켜줘", "켜봐", "켜줄래", "켜주고", "열어줘", "열어봐", "열어줄래",
    "열어주고", "띄워줘", "띄워주고",
    "실행해줘", "실행해봐", "실행시켜",
)
_KNOWN_UNSUPPORTED_APPS = ("포토샵", "엣지", "그림판")
_WEATHER_DEPENDENCY = re.compile(
    r"(?:비|눈|날씨|기온|온도|밖|바깥).{0,30}"
    r"(?:면|하면|으면|일때|인경우|넘으면)",
)
_VOLUME_UP_COMMANDS = ("올려", "올리고", "높여", "높이고", "키워", "크게해")
_VOLUME_DOWN_COMMANDS = ("내려", "내리고", "낮춰", "낮추고", "줄여", "줄이고", "작게해")


def matches_pc_control_request(request: RoutingRequest, apps: AppRegistry) -> bool:
    text = "".join(request.text.lower().split())
    launch_intent = any(intent in text for intent in _LAUNCH_INTENT)
    app_launch = launch_intent and (
        apps.resolve_alias(text) is not None
        or any(app in text for app in _KNOWN_UNSUPPORTED_APPS)
    )
    volume_subject = "볼륨" in text or "소리" in text
    volume_absolute = volume_subject and bool(re.search(
        r"(?:볼륨|소리)(?:을|를)?\d{1,3}(?:%|퍼센트)?(?:로|으로)?"
        r"(?:해줘|해봐|맞춰(?:줘|봐)?|설정(?:해줘|해봐)?|"
        r"줄여(?:줘|봐)|낮춰(?:줘|봐)|올려(?:줘|봐))",
        text,
    ))
    volume_relative = volume_subject and any(
        word in text for word in _VOLUME_UP_COMMANDS + _VOLUME_DOWN_COMMANDS
    )
    volume_action = volume_absolute or volume_relative
    mute_action = (
        "음소거" in text and any(word in text for word in (
            "해줘", "켜줘", "풀어", "해제", "시켜", "걸어", "하고",
        ))
    ) or bool(re.search(r"소리(?:를)?(?:아예)?(?:꺼|끄)|다시소리나게", text))
    media_target = any(word in text for word in (
        "다음곡", "다음노래", "다음트랙", "이전곡", "이전노래", "이전트랙",
    ))
    media_command = any(word in text for word in (
        "넘겨", "재생해", "틀어", "바꿔", "돌아가",
    ))
    media_action = (media_target and (
        media_command or text in {
            "다음곡", "다음노래", "다음트랙", "이전곡", "이전노래", "이전트랙",
        }
    )) or bool(re.search(
        r"(?:한곡넘겨|아까노래(?:로)?돌아가|재생(?:잠깐)?멈춰|"
        r"(?:음악|노래)(?:을|를)?(?:일시정지|재생|다시틀어)|다시재생|재생멈춰)",
        text,
    ))
    unsafe_or_unsupported = bool(re.search(
        r"(?:(?:powershell|cmd|pwsh|del|dir)|--\w+|https?://|"
        r"(?:컴퓨터|pc)(?:를)?(?:재부팅|종료)|(?:창)?닫아줘)",
        text,
    ))
    return app_launch or volume_action or mute_action or media_action \
        or unsafe_or_unsupported


def excludes_weather_lookup_for_pc_discussion(request: RoutingRequest) -> bool:
    """Reject a narrow comparison hard-negative without changing Weather rules."""
    compact = "".join(request.text.lower().split())
    return "날씨앱" in compact and any(word in compact for word in ("비교", "중에", "무거"))


def matches_conditional_weather_pc_request(
    request: RoutingRequest,
    apps: AppRegistry,
) -> bool:
    compact = "".join(request.text.lower().split())
    return bool(_WEATHER_DEPENDENCY.search(compact)) \
        and matches_pc_control_request(request, apps)


def detect_conditional_pc_planning(
    request: RoutingRequest,
    capabilities: frozenset[str],
) -> str | None:
    if "pc_control" in capabilities:
        compact = "".join(request.text.lower().split())
        if _WEATHER_DEPENDENCY.search(compact):
            return "conditional_tool_dependency"
    return None
