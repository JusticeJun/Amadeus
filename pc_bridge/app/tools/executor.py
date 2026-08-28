from __future__ import annotations

from collections.abc import Iterable

from ..models import ChatMessage
from ..routing import RouteDecision
from .base import Tool, ToolExecution, ToolResult


_UNAVAILABLE_CONTEXT = (
    "요청한 기능을 지금 처리하지 못했다. 내부 오류, API, 도구나 시스템을 언급하지 말고 "
    "지금은 처리할 수 없다고 크리스의 말투로 짧게 답한다."
)


class ToolExecutor:
    """Execute capabilities selected by a SemanticRouter."""

    def __init__(self, tools: Iterable[Tool] | None = None) -> None:
        self._tools: dict[str, Tool] = {}
        for tool in tools or ():
            if tool.name in self._tools:
                raise ValueError(f"duplicate tool capability: {tool.name}")
            self._tools[tool.name] = tool

    def execute(
        self,
        decision: RouteDecision,
        user_text: str,
        history: tuple[ChatMessage, ...] = (),
    ) -> tuple[ToolExecution, ...]:
        if decision.planning_required:
            return ()
        executions: list[ToolExecution] = []
        for match in decision.matches:
            tool = self._tools.get(match.capability)
            if tool is None:
                result = ToolResult(
                    match.capability,
                    False,
                    {},
                    f"capability is not registered: {match.capability}",
                )
                executions.append(ToolExecution(result, _UNAVAILABLE_CONTEXT))
                continue
            try:
                contextual_run = getattr(tool, "run_with_context", None)
                result = contextual_run(user_text, history) if contextual_run else tool.run(user_text)
                context = tool.build_llm_context(result)
            except Exception as exc:  # Keep one capability from terminating the turn.
                result = ToolResult(
                    match.capability,
                    False,
                    {},
                    f"unexpected tool error: {exc}",
                )
                context = _UNAVAILABLE_CONTEXT
            executions.append(ToolExecution(
                result,
                context,
                side_effecting=bool(getattr(tool, "side_effecting", False)),
            ))
        return tuple(executions)

    def continue_clarification(
        self, capability: str, pending: dict[str, object], user_text: str,
        history: tuple[ChatMessage, ...] = (),
    ) -> ToolExecution | None:
        """Attempt one bounded, capability-owned clarification continuation."""
        tool = self._tools.get(capability)
        continuation = getattr(tool, "continue_clarification", None) if tool else None
        if continuation is None:
            return None
        result = continuation(pending, user_text, history)
        if result is None:
            return None
        return ToolExecution(
            result, tool.build_llm_context(result),
            side_effecting=bool(getattr(tool, "side_effecting", False)),
        )
