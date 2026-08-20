from __future__ import annotations

from dataclasses import dataclass, field
import json
import re
from typing import Any


ALLOWED_EMOTIONS = frozenset(
    {"neutral", "happy", "shy", "pout", "surprised", "thinking", "sleep", "listening"}
)


def _clamp(value: Any, default: float, minimum: float, maximum: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return min(maximum, max(minimum, parsed))


@dataclass(frozen=True)
class SpeechStyle:
    speed: float = 1.0
    pitch: float = 0.0
    energy: float = 1.0

    @classmethod
    def from_mapping(cls, value: Any) -> "SpeechStyle":
        mapping = value if isinstance(value, dict) else {}
        return cls(
            speed=_clamp(mapping.get("speed"), 1.0, 0.75, 1.25),
            pitch=_clamp(mapping.get("pitch"), 0.0, -6.0, 6.0),
            energy=_clamp(mapping.get("energy"), 1.0, 0.7, 1.3),
        )


@dataclass(frozen=True)
class LlmResult:
    reply: str
    emotion: str = "neutral"
    speech_style: SpeechStyle = field(default_factory=SpeechStyle)

    @classmethod
    def parse(cls, raw: str) -> "LlmResult":
        text = raw.strip()
        payload: dict[str, Any] | None = None
        candidates = [text]
        fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if fenced:
            candidates.insert(0, fenced.group(1))
        braces = re.search(r"\{.*\}", text, re.DOTALL)
        if braces:
            candidates.append(braces.group(0))
        for candidate in candidates:
            try:
                decoded = json.loads(candidate)
            except (json.JSONDecodeError, TypeError):
                continue
            if isinstance(decoded, dict):
                payload = decoded
                break

        if payload is None:
            recovered = re.sub(r"```(?:json)?|```", "", text).strip()
            return cls(reply=recovered or "잠시 생각을 정리하지 못했어요.")

        reply = str(payload.get("reply") or "").strip()
        if not reply:
            reply = "잠시 생각을 정리하지 못했어요."
        emotion = str(payload.get("emotion") or "neutral").strip().lower()
        if emotion not in ALLOWED_EMOTIONS:
            emotion = "neutral"
        return cls(reply=reply, emotion=emotion,
                   speech_style=SpeechStyle.from_mapping(payload.get("speech_style")))


@dataclass(frozen=True)
class ChatMessage:
    role: str
    content: str

