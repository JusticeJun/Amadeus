from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol


class PcActionType(str, Enum):
    LAUNCH_APP = "launch_app"
    ADJUST_VOLUME = "adjust_volume"
    SET_VOLUME = "set_volume"
    MUTE = "mute"
    UNMUTE = "unmute"
    MEDIA_PLAY_PAUSE = "media_play_pause"
    MEDIA_NEXT = "media_next"
    MEDIA_PREVIOUS = "media_previous"


@dataclass(frozen=True)
class PcAction:
    action_type: PcActionType
    target: str = ""
    amount: int | None = None

    def __post_init__(self) -> None:
        if self.action_type is PcActionType.LAUNCH_APP and not self.target:
            raise ValueError("launch_app requires a target")
        if self.action_type in {PcActionType.ADJUST_VOLUME, PcActionType.SET_VOLUME}:
            if self.amount is None:
                raise ValueError(f"{self.action_type.value} requires an amount")
        elif self.amount is not None:
            raise ValueError(f"{self.action_type.value} does not accept an amount")
        if self.action_type is PcActionType.SET_VOLUME \
                and self.amount is not None and not 0 <= self.amount <= 100:
            raise ValueError("volume must be between 0 and 100")


@dataclass(frozen=True)
class PcActionParseResult:
    actions: tuple[PcAction, ...] = ()
    error_code: str = ""

    @property
    def ok(self) -> bool:
        return bool(self.actions) and not self.error_code


class PcActionParser(Protocol):
    def parse(self, user_text: str) -> PcActionParseResult: ...


@dataclass(frozen=True)
class PcActionResult:
    action: PcAction
    ok: bool
    data: dict[str, object]
    error: str = ""


class PcController(Protocol):
    def execute(self, action: PcAction) -> PcActionResult: ...
