from __future__ import annotations

from dataclasses import dataclass

from app.models import ChatMessage
from app.music_control import (
    MusicActionType, MusicSemanticInterpreter, RuleBasedMusicActionParser,
)
from app.pc_control import default_app_registry
from app.routing import RoutingRequest, create_default_semantic_router
from app.semantic_llm import (
    SemanticLlmError, SemanticLlmMetrics, SemanticLlmResponse,
)
from app.tools import MusicControlTool


def action_data(action_type="play_song", **overrides):
    return {
        "type": action_type,
        "title": "",
        "artist": "",
        "playlist": "",
        "alternate_queries": [],
        "artist_explicit": False,
        **overrides,
    }


@dataclass
class FakeSemanticClient:
    data: dict[str, object] | None = None
    error: SemanticLlmError | None = None

    def __post_init__(self):
        self.requests = []

    def complete(self, request):
        self.requests.append(request)
        if self.error:
            raise self.error
        return SemanticLlmResponse(self.data or {}, SemanticLlmMetrics(
            provider="fake", model="semantic-test", elapsed_seconds=0.125,
            input_tokens=20, output_tokens=10,
        ))


def test_transport_uses_rule_fast_path_without_llm() -> None:
    client = FakeSemanticClient()
    interpreter = MusicSemanticInterpreter(RuleBasedMusicActionParser(), client)

    result = interpreter.interpret("일시정지해줘")

    assert result.action.action_type is MusicActionType.PAUSE
    assert client.requests == []
    assert interpreter.metrics.fast_path_hits == 1


def test_llm_interpretation_uses_only_recent_music_context_and_is_cached() -> None:
    client = FakeSemanticClient({"status": "parsed", "actions": [action_data(
        title="마리골드", artist="아이묭", alternate_queries=["aimyon Marigold"],
        artist_explicit=True,
    )]})
    interpreter = MusicSemanticInterpreter(RuleBasedMusicActionParser(), client)
    history = (
        ChatMessage("user", "어제 점심 뭐 먹었지?"),
        ChatMessage("assistant", "백넘버 노래를 재생했어."),
        ChatMessage("user", "날씨도 좋았어."),
    )

    first = interpreter.interpret("그 노래 다시 틀어줘", history)
    second = interpreter.interpret("그 노래 다시 틀어줘", history)

    assert first == second
    assert first.action.title == "마리골드"
    assert first.action.alternate_queries == ("aimyon Marigold",)
    assert len(client.requests) == 1
    assert client.requests[0].input["music_context"] == [
        "assistant: 백넘버 노래를 재생했어.",
    ]
    assert interpreter.metrics.input_tokens == 20


def test_music_specific_fallback_recovers_rule_routing_miss() -> None:
    client = FakeSemanticClient({"status": "parsed", "actions": [action_data(
        "play_playlist", playlist="Backnumber",
    )]})
    interpreter = MusicSemanticInterpreter(RuleBasedMusicActionParser(), client)
    router = create_default_semantic_router(default_app_registry(), interpreter)

    decision = router.route(RoutingRequest("Backnumber 플레이리스트 있잖아 틀어"))

    assert decision.required_capabilities == {"music_control"}
    assert interpreter.interpret("Backnumber 플레이리스트 있잖아 틀어").action.playlist == (
        "Backnumber"
    )
    assert len(client.requests) == 1


def test_ambiguous_response_routes_to_safe_non_execution_result() -> None:
    client = FakeSemanticClient({"status": "ambiguous", "actions": []})
    interpreter = MusicSemanticInterpreter(RuleBasedMusicActionParser(), client)
    router = create_default_semantic_router(default_app_registry(), interpreter)

    decision = router.route(RoutingRequest("그 플레이리스트 틀어줘"))
    result = interpreter.interpret("그 플레이리스트 틀어줘")

    assert decision.required_capabilities == {"music_control"}
    assert not result.ok and result.error_code == "ambiguous"


def test_invalid_or_excessive_alternates_are_rejected() -> None:
    client = FakeSemanticClient({"status": "parsed", "actions": [action_data(
        title="수평선", alternate_queries=["1", "2", "3", "4", "5"],
    )]})
    interpreter = MusicSemanticInterpreter(RuleBasedMusicActionParser(), client)

    result = interpreter.interpret("수평선 좀 들려줄래")

    assert not result.ok and result.error_code == "schema_violation"
    assert interpreter.metrics.errors == 1


def test_provider_failure_does_not_fallback_to_untrusted_complex_rule() -> None:
    client = FakeSemanticClient(error=SemanticLlmError("timeout", "slow"))
    interpreter = MusicSemanticInterpreter(RuleBasedMusicActionParser(), client)

    result = interpreter.interpret("안녕 크리스 적적한데 백넘버 플레이리스트 틀어줘")

    assert not result.ok and result.error_code == "semantic_timeout"
    assert interpreter.metrics.timeouts == 1


def test_provider_failure_can_use_simple_single_action_rule() -> None:
    client = FakeSemanticClient(error=SemanticLlmError("rate_limit", "quota"))
    interpreter = MusicSemanticInterpreter(RuleBasedMusicActionParser(), client)

    result = interpreter.interpret("마리골드 틀어줘")

    assert result.ok and result.action.title == "마리골드"
    assert interpreter.metrics.rate_limits == 1


def test_schema_failure_never_reaches_music_controller() -> None:
    client = FakeSemanticClient({"status": "parsed", "actions": [action_data(
        title="수평선", alternate_queries=["same", "same"],
    )]})
    interpreter = MusicSemanticInterpreter(RuleBasedMusicActionParser(), client)

    class MustNotExecuteController:
        def execute(self, action):
            raise AssertionError("invalid semantic output must not execute")

    result = MusicControlTool(interpreter, MustNotExecuteController()).run(
        "수평선 좀 들려줄래",
    )

    assert not result.ok
    assert result.data["reason"] == "schema_violation"
