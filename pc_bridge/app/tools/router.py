from __future__ import annotations

from .base import Tool, ToolResult


class ToolRouter:
    """Select at most one matching tool for a user turn."""

    def __init__(self, tools: list[Tool] | None = None) -> None:
        self._tools = list(tools or [])

    def dispatch(self, user_text: str) -> ToolResult | None:
        for tool in self._tools:
            if tool.matches(user_text):
                return tool.run(user_text)
        return None
