from __future__ import annotations

from dataclasses import dataclass, field
from difflib import SequenceMatcher
import json
import re
from typing import Any


ALLOWED_EMOTIONS = frozenset(
    {"neutral", "happy", "shy", "pout", "surprised", "thinking", "sleep", "listening"}
)
RESPONSE_EMOTIONS = frozenset(
    {"neutral", "happy", "shy", "pout", "surprised", "thinking"}
)
FALLBACK_REPLY = "잠시 생각을 정리하지 못했어요."


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
            return cls(reply=normalize_reply(recovered) or FALLBACK_REPLY)

        reply = str(payload.get("reply") or "").strip()
        if not reply:
            reply = FALLBACK_REPLY
        reply = normalize_reply(reply) or FALLBACK_REPLY
        emotion = str(payload.get("emotion") or "neutral").strip().lower()
        if emotion not in RESPONSE_EMOTIONS:
            emotion = "neutral"
        return cls(reply=reply, emotion=emotion,
                   speech_style=SpeechStyle.from_mapping(payload.get("speech_style")))


@dataclass(frozen=True)
class ChatMessage:
    role: str
    content: str


def normalize_reply(value: str, max_chars: int = 650, max_sentences: int = 5) -> str:
    """Make model output safe and natural for Korean TTS without changing its meaning."""
    text = re.sub(r"```(?:\w+)?|```", "", str(value))
    text = re.sub(r"[*_#`]", "", text)
    text = re.sub(r"\([^)]*(?:행동|표정|웃음|한숨)[^)]*\)", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    sentence_ends = list(re.finditer(r"[.!?](?=\s|$)", text))
    if len(sentence_ends) > max_sentences:
        text = text[:sentence_ends[max_sentences - 1].end()].strip()
    if len(text) <= max_chars:
        return text
    shortened = text[:max_chars + 1]
    boundary = max(shortened.rfind("."), shortened.rfind("?"), shortened.rfind("!"))
    if boundary >= max_chars // 2:
        return shortened[:boundary + 1].strip()
    boundary = max(shortened.rfind(","), shortened.rfind(" "))
    return shortened[:boundary if boundary > 0 else max_chars].rstrip(" ,") + "."


def is_repetitive_reply(reply: str, user_text: str, history: list[ChatMessage]) -> bool:
    """Detect direct echoes and near-duplicate recent assistant answers."""
    candidate = _comparison_text(reply)
    if not candidate:
        return True
    user = _comparison_text(user_text)
    if user and candidate == user:
        return True
    if len(user) >= 4 and _too_similar(candidate, user, threshold=0.86):
        return True
    recent_pairs: list[tuple[str, str]] = []
    for index, item in enumerate(history):
        if item.role != "assistant":
            continue
        prior_user = next(
            (history[cursor].content for cursor in range(index - 1, -1, -1)
             if history[cursor].role == "user"),
            "",
        )
        recent_pairs.append((_comparison_text(prior_user), _comparison_text(item.content)))
    for prior_user, previous_reply in recent_pairs[-4:]:
        if not previous_reply or not _too_similar(candidate, previous_reply, threshold=0.82):
            continue
        # Repeated factual questions (for example, asking a name again) may
        # legitimately require the same answer.
        if prior_user and _too_similar(user, prior_user, threshold=0.72):
            continue
        return True
    return False


def _comparison_text(value: str) -> str:
    return re.sub(r"[^0-9A-Za-z가-힣]", "", value).lower()


def _too_similar(left: str, right: str, threshold: float) -> bool:
    if left == right:
        return True
    shorter, longer = sorted((left, right), key=len)
    if len(shorter) >= 7 and shorter in longer and len(shorter) / len(longer) >= 0.78:
        return True
    return SequenceMatcher(None, left, right).ratio() >= threshold

