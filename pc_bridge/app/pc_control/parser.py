from __future__ import annotations

import re

from .actions import PcAction, PcActionParseResult, PcActionType
from .registry import AppRegistry


_UNSAFE_INPUT = re.compile(
    r"(?:[;&|]|\b(?:powershell|cmd(?:\.exe)?|pwsh|del)\b|/(?:c|k)\b|"
    r"-command\b|--[\w-]+|https?://|\b[\w-]+\.(?:com|net|org)"
    r"(?![a-z0-9.-])|[a-z]:\\)",
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
            "켜줘", "켜봐", "켜줄래", "켜주고", "열어줘", "열어봐", "열어줄래",
            "열어주고", "띄워줘", "띄워주고",
            "실행해줘", "실행해봐", "실행시켜",
        ))
        if app_id and launch_intent:
            actions.append(PcAction(PcActionType.LAUNCH_APP, target=app_id))

        volume_subject = "볼륨" in compact or "소리" in compact
        maximum = volume_subject and bool(re.search(
            r"(?:볼륨|소리)(?:을|를)?(?:좀|아예)?(?:최대|최대치)(?:로|으로)?"
            r"(?:해줘|해봐|맞춰(?:줘|봐)?|설정(?:해줘|해봐)?|"
            r"올려(?:줘|봐)|높여(?:줘|봐)|키워(?:줘|봐)?)",
            compact,
        ))
        absolute = re.search(
            r"(?:볼륨|소리)(?:을|를)?(\d{1,3})(?:%|퍼센트)?(?:로|으로)?"
            r"(?:해줘|해봐|맞춰(?:줘|봐)?|설정(?:해줘|해봐)?|"
            r"줄여(?:줘|봐)|낮춰(?:줘|봐)|올려(?:줘|봐)|하고)",
            compact,
        )
        if maximum:
            actions.append(PcAction(PcActionType.SET_VOLUME, amount=100))
        elif absolute:
            amount = int(absolute.group(1))
            if not 0 <= amount <= 100:
                return PcActionParseResult(error_code="invalid_volume")
            actions.append(PcAction(PcActionType.SET_VOLUME, amount=amount))
        elif volume_subject and any(word in compact for word in (
            "올려줘", "올리고", "높여줘", "높이고", "키워줘", "크게해줘",
            "좀키워", "좀올려",
        )):
            actions.append(PcAction(PcActionType.ADJUST_VOLUME, amount=self._volume_step))
        elif volume_subject and any(word in compact for word in (
            "내려줘", "내리고", "낮춰줘", "낮추고", "줄여줘", "줄이고",
            "작게해줘", "좀줄여", "좀내려",
        )):
            actions.append(PcAction(PcActionType.ADJUST_VOLUME, amount=-self._volume_step))

        if "음소거" in compact and any(word in compact for word in (
            "풀어줘", "해제해줘", "끄고", "풀어",
        )):
            actions.append(PcAction(PcActionType.UNMUTE))
        elif "다시소리나게" in compact:
            actions.append(PcAction(PcActionType.UNMUTE))
        elif "음소거" in compact and any(word in compact for word in (
            "해줘", "켜줘", "해봐", "시켜", "걸어줘", "하고",
        )):
            actions.append(PcAction(PcActionType.MUTE))
        elif re.search(r"소리(?:를)?(?:아예)?(?:꺼|끄)", compact):
            actions.append(PcAction(PcActionType.MUTE))

        next_media = any(word in compact for word in ("다음곡", "다음노래", "다음트랙"))
        previous_media = any(word in compact for word in ("이전곡", "이전노래", "이전트랙"))
        media_command = any(word in compact for word in (
            "넘겨", "재생해", "틀어", "바꿔", "돌아가",
        ))
        if next_media and (media_command or compact in {"다음곡", "다음노래", "다음트랙"}):
            actions.append(PcAction(PcActionType.MEDIA_NEXT))
        elif previous_media and (
            media_command or compact in {"이전곡", "이전노래", "이전트랙"}
        ):
            actions.append(PcAction(PcActionType.MEDIA_PREVIOUS))
        elif re.search(
            r"(?:한곡넘겨|아까노래(?:로)?돌아가|재생(?:잠깐)?멈춰|"
            r"(?:음악|노래)(?:을|를)?(?:일시정지|재생|다시틀어)|다시재생|재생멈춰)",
            compact,
        ):
            if "한곡넘겨" in compact:
                actions.append(PcAction(PcActionType.MEDIA_NEXT))
            elif "아까노래" in compact:
                actions.append(PcAction(PcActionType.MEDIA_PREVIOUS))
            else:
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
