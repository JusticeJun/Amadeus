from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class ToolResult:
    name: str
    ok: bool
    data: dict[str, Any]
    error: str = ""

class Tool(Protocol):
    name: str

    def run(self, user_text: str) -> ToolResult: ...

    def build_llm_context(self, result: ToolResult) -> str: ...


@dataclass(frozen=True)
class ToolExecution:
    result: ToolResult
    llm_context: str
