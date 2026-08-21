from __future__ import annotations

import re

from .models import ChatMessage


class ConversationMemory:
    def __init__(self, max_recent_messages: int = 8,
                 max_summary_chars: int = 420) -> None:
        self._max_recent = max_recent_messages
        self._max_summary_chars = max_summary_chars
        self._recent: list[ChatMessage] = []
        self._context: list[str] = []
        self._facts: list[str] = []

    def messages(self) -> list[ChatMessage]:
        memory = self._memory_text()
        prefix = [ChatMessage("system", memory)] if memory else []
        return prefix + list(self._recent)

    def add_turn(self, user_text: str, assistant_text: str) -> None:
        self._recent.extend([
            ChatMessage("user", user_text),
            ChatMessage("assistant", assistant_text),
        ])
        if len(self._recent) > self._max_recent:
            removed = self._recent[:-self._max_recent]
            self._recent = self._recent[-self._max_recent:]
            self._remember(removed)

    def _remember(self, removed: list[ChatMessage]) -> None:
        for item in removed:
            text = re.sub(r"\s+", " ", item.content).strip()
            if self._is_low_information(text):
                continue
            if item.role == "user" and self._looks_like_fact(text):
                self._append_unique(self._facts, text[:140])
                continue
            label = "사용자" if item.role == "user" else "크리스"
            self._append_unique(self._context, f"{label}: {text[:120]}")
        self._trim_memory()

    def _memory_text(self) -> str:
        sections = []
        if self._facts:
            sections.append("사용자 핵심 정보: " + " / ".join(self._facts))
        if self._context:
            sections.append("이전 맥락: " + " / ".join(self._context))
        return " | ".join(sections)

    def _trim_memory(self) -> None:
        while len(self._memory_text()) > self._max_summary_chars and self._context:
            self._context.pop(0)
        while len(self._memory_text()) > self._max_summary_chars and len(self._facts) > 1:
            self._facts.pop(0)

    @staticmethod
    def _append_unique(items: list[str], value: str) -> None:
        if value not in items:
            items.append(value)

    @staticmethod
    def _is_low_information(text: str) -> bool:
        compact = re.sub(r"[^0-9A-Za-z가-힣]", "", text).lower()
        return len(compact) < 3 or compact in {
            "안녕", "그래", "응", "아니", "고마워", "알겠어", "뭐해",
        }

    @staticmethod
    def _looks_like_fact(text: str) -> bool:
        markers = (
            "내 이름", "나는 ", "난 ", "내가 ", "좋아", "싫어", "기억해",
            "프로젝트", "라고 불러", "라고 해", "목표", "원해",
        )
        return any(marker in text for marker in markers)

