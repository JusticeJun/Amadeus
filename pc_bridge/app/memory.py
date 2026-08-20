from __future__ import annotations

from .models import ChatMessage


class ConversationMemory:
    def __init__(self, max_recent_messages: int = 12) -> None:
        self._max_recent = max_recent_messages
        self._recent: list[ChatMessage] = []
        self._summary = ""

    def messages(self) -> list[ChatMessage]:
        prefix = [ChatMessage("system", f"이전 대화 요약: {self._summary}")] if self._summary else []
        return prefix + list(self._recent)

    def add_turn(self, user_text: str, assistant_text: str) -> None:
        self._recent.extend([
            ChatMessage("user", user_text),
            ChatMessage("assistant", assistant_text),
        ])
        if len(self._recent) > self._max_recent:
            removed = self._recent[:-self._max_recent]
            self._recent = self._recent[-self._max_recent:]
            fragments = [f"{item.role}: {item.content}" for item in removed]
            joined = " | ".join(fragments)
            self._summary = (self._summary + " | " + joined).strip(" |")[-1200:]

