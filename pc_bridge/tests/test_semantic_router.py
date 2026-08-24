from __future__ import annotations

from app.models import ChatMessage
import pytest

from app.routing import (
    CapabilityMatch,
    create_default_semantic_router,
    RouteDecision,
    RoutingRequest,
    RuleBasedSemanticRouter,
)
from app.pc_control import AppDefinition, AppRegistry
from app.tools import ToolExecutor
from app.tools.base import ToolResult


class StubTool:
    def __init__(self, name: str, *, ok: bool = True) -> None:
        self.name = name
        self.ok = ok
        self.calls: list[str] = []

    def run(self, user_text: str) -> ToolResult:
        self.calls.append(user_text)
        return ToolResult(self.name, self.ok, {"source": self.name}, "failed" if not self.ok else "")

    def build_llm_context(self, result: ToolResult) -> str:
        return f"{self.name}:{'ok' if result.ok else 'unavailable'}"


class RaisingTool(StubTool):
    def run(self, user_text: str) -> ToolResult:
        raise RuntimeError("private failure")


def test_rule_router_returns_multiple_capabilities_without_executing_tools() -> None:
    observed_history: list[ChatMessage] = []

    def weather_rule(request: RoutingRequest) -> bool:
        observed_history.extend(request.history)
        return "비" in request.text

    router = RuleBasedSemanticRouter({
        "weather": weather_rule,
        "media_control": lambda request: "음악" in request.text and "틀어" in request.text,
    })
    request = RoutingRequest(
        "오늘 비 오는지 확인하고 음악 틀어줘.",
        (ChatMessage("user", "밖에 나갈 거야."),),
    )

    decision = router.route(request)

    assert decision.required_capabilities == frozenset({"weather", "media_control"})
    assert observed_history == [ChatMessage("user", "밖에 나갈 거야.")]
    assert all(match.confidence is None for match in decision.matches)


def test_tool_executor_runs_every_selected_capability_and_keeps_context_separate() -> None:
    weather = StubTool("weather")
    media = StubTool("media_control")
    router = RuleBasedSemanticRouter({
        "weather": lambda request: True,
        "media_control": lambda request: True,
    })
    decision = router.route(RoutingRequest("날씨를 확인하고 음악을 틀어줘."))

    executions = ToolExecutor([weather, media]).execute(decision, "복합 요청")

    assert [item.result.name for item in executions] == ["weather", "media_control"]
    assert [item.llm_context for item in executions] == ["weather:ok", "media_control:ok"]
    assert weather.calls == ["복합 요청"]
    assert media.calls == ["복합 요청"]


def test_unregistered_capability_is_isolated_as_a_safe_execution_failure() -> None:
    decision = RuleBasedSemanticRouter({"calendar": lambda request: True}).route(
        RoutingRequest("내 일정 확인해줘.")
    )

    executions = ToolExecutor().execute(decision, "내 일정 확인해줘.")

    assert len(executions) == 1
    assert not executions[0].result.ok
    assert "calendar" in executions[0].result.error
    assert "calendar" not in executions[0].llm_context
    assert "capability is not registered" not in executions[0].llm_context


def test_empty_route_executes_nothing() -> None:
    assert ToolExecutor().execute(RouteDecision(), "그냥 대화하자.") == ()


def test_one_tool_failure_does_not_prevent_other_selected_capabilities() -> None:
    decision = RuleBasedSemanticRouter({
        "weather": lambda request: True,
        "media_control": lambda request: True,
    }).route(RoutingRequest("날씨를 확인하고 음악을 틀어줘."))

    executions = ToolExecutor([
        RaisingTool("weather"),
        StubTool("media_control"),
    ]).execute(decision, "복합 요청")

    assert not executions[0].result.ok
    assert "private failure" in executions[0].result.error
    assert "private failure" not in executions[0].llm_context
    assert executions[1].result.ok


def test_route_decision_rejects_invalid_confidence_and_duplicate_capabilities() -> None:
    with pytest.raises(ValueError, match="confidence"):
        CapabilityMatch("weather", 1.1)
    with pytest.raises(ValueError, match="duplicate"):
        RouteDecision((CapabilityMatch("weather"), CapabilityMatch("weather")))


def _routing_apps() -> AppRegistry:
    return AppRegistry((
        AppDefinition("chrome", frozenset({"chrome", "크롬"}), ()),
        AppDefinition("notepad", frozenset({"notepad", "메모장"}), ()),
    ))


def test_default_router_selects_pc_control_without_keyword_hard_negatives() -> None:
    router = create_default_semantic_router(_routing_apps())

    assert router.route(RoutingRequest("크롬 켜줘")).required_capabilities == {
        "pc_control",
    }
    assert router.route(RoutingRequest("소리 좀 작게 해줘")).required_capabilities == {
        "pc_control",
    }
    assert not router.route(RoutingRequest("크롬이랑 엣지 중 뭐가 좋아?")).matches
    assert not router.route(RoutingRequest("이 노래 볼륨이 작은 이유가 뭐야?")).matches


def test_default_router_supports_independent_multilabel_requests() -> None:
    decision = create_default_semantic_router(_routing_apps()).route(
        RoutingRequest("오늘 날씨 알려주고 크롬 켜줘")
    )

    assert decision.required_capabilities == {"weather", "pc_control"}
    assert not decision.planning_required


def test_conditional_multitool_request_is_preserved_but_not_executed() -> None:
    weather = StubTool("weather")
    pc_control = StubTool("pc_control")
    decision = create_default_semantic_router(_routing_apps()).route(
        RoutingRequest("오늘 비 오면 크롬 켜줘")
    )

    executions = ToolExecutor([weather, pc_control]).execute(decision, "오늘 비 오면 크롬 켜줘")

    assert decision.required_capabilities == {"weather", "pc_control"}
    assert decision.planning_required
    assert decision.planning_reason == "conditional_tool_dependency"
    assert executions == ()
    assert weather.calls == []
    assert pc_control.calls == []
