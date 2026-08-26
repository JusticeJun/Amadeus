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


@pytest.mark.parametrize(("text", "expected"), [
    ("오늘 날씨 알려주고 제이팝 플레이리스트 틀어줘", {"weather", "music_control"}),
    ("볼륨 30으로 하고 마리골드 틀어줘", {"pc_control", "music_control"}),
    (
        "날씨 알려주고 볼륨 줄이고 제이팝 플레이리스트 틀어줘",
        {"weather", "pc_control", "music_control"},
    ),
])
def test_default_router_supports_music_multilabel_requests(text, expected) -> None:
    decision = create_default_semantic_router(_routing_apps()).route(RoutingRequest(text))

    assert decision.required_capabilities == expected
    assert not decision.planning_required


@pytest.mark.parametrize("text", [
    "지금 무슨 노래야?",
    "일시정지해줘",
    "일시정지 좀 해줘",
    "다시 재생해줘",
    "다음 곡",
    "이전 곡",
])
def test_default_router_routes_supported_music_transport_actions(text) -> None:
    decision = create_default_semantic_router(_routing_apps()).route(RoutingRequest(text))

    assert decision.required_capabilities == {"music_control"}


@pytest.mark.parametrize("text", [
    "백넘버의 수평선 재생해줄래?",
    "백넘버 수평선 틀어줘",
    "백넘버가 부른 수평선 틀어줘",
    "수평선이라는 노래 틀어줘",
])
def test_default_router_routes_representative_song_request_forms(text) -> None:
    decision = create_default_semantic_router(_routing_apps()).route(RoutingRequest(text))

    assert decision.required_capabilities == {"music_control"}


@pytest.mark.parametrize("text", [
    "아이묭 마리골드 틀어",
    "백넘버 노래 들려",
    "재즈 음악 재생해",
])
def test_default_router_routes_music_imperatives(text) -> None:
    decision = create_default_semantic_router(_routing_apps()).route(RoutingRequest(text))

    assert decision.required_capabilities == {"music_control"}


def test_default_router_routes_implicit_current_heat_regression() -> None:
    decision = create_default_semantic_router(_routing_apps()).route(RoutingRequest(
        "오늘도 밖에 많이 덥나? 요즘 개덥던데 진짜로",
    ))

    assert decision.required_capabilities == {"weather"}


def test_conditional_music_request_preserves_capabilities_and_blocks_side_effect() -> None:
    weather = StubTool("weather")
    music = StubTool("music_control")
    decision = create_default_semantic_router(_routing_apps()).route(
        RoutingRequest("비 오면 제이팝 플레이리스트 틀어줘")
    )

    executions = ToolExecutor([weather, music]).execute(decision, "조건부 음악 요청")

    assert decision.required_capabilities == {"weather", "music_control"}
    assert decision.planning_required
    assert executions == ()
    assert not weather.calls and not music.calls


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
