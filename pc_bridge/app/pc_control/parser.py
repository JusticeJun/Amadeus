from __future__ import annotations

import re

from .actions import PcAction, PcActionParseResult, PcActionType
from .registry import AppRegistry


_UNSAFE_INPUT = re.compile(
    r"(?:[;&|]|\b(?:powershell|cmd(?:\.exe)?|pwsh)\b|/(?:c|k)\b|-command\b)",
    re.IGNORECASE,
)


class RuleBasedPcActionParser:
    """Extract the small, supported PC action set without executing text."""

    def __init__(self, apps: AppRegistry, volume_step: int = 10) -> None:
        if not 1 <= volume_step <= 100:
            raise ValueError("volume step must be between 1 and 100")
        self._apps = apps
        self._volume_step = volume_step

    def parse(self, user_text: str) -> PcActionParseResult:
        if _UNSAFE_INPUT.search(user_text):
            return PcActionParseResult(error_code="unsafe_input")
        compact = "".join(user_text.lower().split())
        actions: list[PcAction] = []

        app_id = self._apps.resolve_alias(compact)
        launch_intent = any(word in compact for word in (
            "켜줘", "켜봐", "열어줘", "열어봐", "실행해줘", "실행해봐", "실행시켜",
        ))
        if app_id and launch_intent:
            actions.append(PcAction(PcActionType.LAUNCH_APP, target=app_id))

        volume_subject = "볼륨" in compact or "소리" in compact
        absolute = re.search(r"(?:볼륨|소리)(?:을|를)?(\d{1,3})(?:%|퍼센트)?(?:로|으로)?", compact)
        if absolute and any(word in compact for word in ("해줘", "맞춰", "설정", "해봐")):
            amount = int(absolute.group(1))
            if not 0 <= amount <= 100:
                return PcActionParseResult(error_code="invalid_volume")
            actions.append(PcAction(PcActionType.SET_VOLUME, amount=amount))
        elif volume_subject and any(word in compact for word in (
            "올려줘", "높여줘", "키워줘", "크게해줘", "좀키워", "좀올려",
        )):
            actions.append(PcAction(PcActionType.ADJUST_VOLUME, amount=self._volume_step))
        elif volume_subject and any(word in compact for word in (
            "내려줘", "낮춰줘", "줄여줘", "작게해줘", "좀줄여", "좀내려",
        )):
            actions.append(PcAction(PcActionType.ADJUST_VOLUME, amount=-self._volume_step))

        if "음소거" in compact and any(word in compact for word in (
            "풀어줘", "해제해줘", "끄고", "풀어",
        )):
            actions.append(PcAction(PcActionType.UNMUTE))
        elif "음소거" in compact and any(word in compact for word in (
            "해줘", "켜줘", "해봐", "시켜",
        )):
            actions.append(PcAction(PcActionType.MUTE))

        next_media = any(word in compact for word in ("다음곡", "다음노래", "다음트랙"))
        previous_media = any(word in compact for word in ("이전곡", "이전노래", "이전트랙"))
        media_command = any(word in compact for word in ("넘겨", "재생해", "틀어", "바꿔"))
        if next_media and (media_command or compact in {"다음곡", "다음노래", "다음트랙"}):
            actions.append(PcAction(PcActionType.MEDIA_NEXT))
        elif previous_media and (
            media_command or compact in {"이전곡", "이전노래", "이전트랙"}
        ):
            actions.append(PcAction(PcActionType.MEDIA_PREVIOUS))
        elif re.search(
            r"(?:(?:음악|노래)(?:을|를)?(?:일시정지|재생)|다시재생|재생멈춰)",
            compact,
        ):
            actions.append(PcAction(PcActionType.MEDIA_PLAY_PAUSE))

        if actions:
            return PcActionParseResult(tuple(_deduplicate(actions)))
        if launch_intent:
            return PcActionParseResult(error_code="unsupported_app")
        return PcActionParseResult(error_code="unsupported_action")


def _deduplicate(actions: list[PcAction]) -> list[PcAction]:
    result: list[PcAction] = []
    for action in actions:
        if action not in result:
            result.append(action)
    return result
