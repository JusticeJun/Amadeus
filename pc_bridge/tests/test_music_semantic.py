from __future__ import annotations

from dataclasses import dataclass

from app.models import ChatMessage
from app.music_control import (
    MusicAction, MusicActionType, MusicItem, MusicSemanticInterpreter,
    PlaylistItem, RuleBasedMusicActionParser,
)
from app.music_control.controller import CatalogQueryVariant
from app.pc_control import default_app_registry
from app.routing import RoutingRequest, create_default_semantic_router
from app.routing.music_control_rules import matches_music_control_request
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


class SequenceSemanticClient(FakeSemanticClient):
    def __init__(self, responses):
        super().__init__()
        self.responses = iter(responses)

    def complete(self, request):
        self.data = next(self.responses)
        return super().complete(request)


def test_transport_uses_rule_fast_path_without_llm() -> None:
    client = FakeSemanticClient()
    interpreter = MusicSemanticInterpreter(RuleBasedMusicActionParser(), client)

    result = interpreter.interpret("일시정지해줘")

    assert result.action.action_type is MusicActionType.PAUSE
    assert client.requests == []
    assert interpreter.metrics.fast_path_hits == 1


def test_common_pause_transport_paraphrases_use_rule_fast_path() -> None:
    client = FakeSemanticClient()
    interpreter = MusicSemanticInterpreter(RuleBasedMusicActionParser(), client)

    for text in (
        "노래 꺼줘",
        "음악 꺼",
        "재생 멈춰",
        "재생 잠깐 멈춰줘",
        "일시정지해",
    ):
        result = interpreter.interpret(text)
        assert result.action.action_type is MusicActionType.PAUSE
        assert matches_music_control_request(RoutingRequest(text))
    assert client.requests == []


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


def test_named_playlist_any_track_is_a_complete_playlist_action() -> None:
    client = FakeSemanticClient({
        "status": "parsed",
        "actions": [action_data(
            "play_playlist", playlist="백넘버", alternate_queries=["BackNumber"],
        )],
    })
    interpreter = MusicSemanticInterpreter(RuleBasedMusicActionParser(), client)
    result = interpreter.interpret(
        "안녕 크리스 백넘버 플레이리스트에서 아무곡이나 틀어줘",
    )
    assert result.ok
    assert result.action.action_type.value == "play_playlist"
    assert result.action.playlist == "백넘버"
    assert result.action.alternate_queries == ("BackNumber",)
    assert "any/random song" in client.requests[0].system_prompt
    assert "assistant is named Chris" in client.requests[0].system_prompt


def test_clarification_semantics_separate_wrapper_from_artist_slot() -> None:
    client = FakeSemanticClient({
        "status": "parsed", "actions": [action_data(
            title="수평선", artist="백넘버",
            alternate_queries=["back number Suiheisen"], artist_explicit=True,
        )],
    })
    interpreter = MusicSemanticInterpreter(RuleBasedMusicActionParser(), client)

    result = interpreter.continue_clarification({
        "kind": "missing_entity", "field": "artist",
        "action": {"type": "play_song", "title": "수평선"},
    }, "당연히 백넘버지")

    assert result.ok
    assert result.action.artist == "백넘버"
    assert result.action.alternate_queries == ("back number Suiheisen",)
    assert "discourse wrappers" in client.requests[0].input["utterance"]


def test_missing_cross_script_alternate_uses_bounded_equivalence_fallback() -> None:
    client = SequenceSemanticClient((
        {"status": "parsed", "actions": [action_data(
            title="한국어 발음 곡명", artist="한국어 발음 가수",
            artist_explicit=True,
        )]},
        {"variants": [{
            "artist": "Canonical Artist", "title": "Canonical Title",
        }]},
    ))
    interpreter = MusicSemanticInterpreter(RuleBasedMusicActionParser(), client)

    result = interpreter.interpret("한국어 발음 가수 한국어 발음 곡명 틀어줘")
    variants = interpreter.rewrite_track_queries(
        result.action.artist, result.action.title,
    )

    assert result.action.alternate_queries == ()
    assert variants == (CatalogQueryVariant("Canonical Artist", "Canonical Title"),)
    assert [request.task for request in client.requests] == [
        "music_interpretation", "music_catalog_query_rewrite",
    ]


def test_recovery_deduplicates_hints_individually_instead_of_dropping_batch() -> None:
    client = FakeSemanticClient({
        "variants": [
            {"artist": "surface artist", "title": "surface title"},
            {"artist": "Canonical Artist", "title": "Canonical Title"},
            {"artist": "Canonical  Artist", "title": "Canonical Title"},
        ],
    })
    interpreter = MusicSemanticInterpreter(RuleBasedMusicActionParser(), client)
    action = MusicAction(
        MusicActionType.PLAY_SONG, title="surface title", artist="surface artist",
        source_query="surface artist surface title",
    )

    result = interpreter.rewrite_track_queries(action.artist, action.title)

    assert result == (CatalogQueryVariant("Canonical Artist", "Canonical Title"),)


def test_query_rewriter_rejects_malformed_or_excessive_output() -> None:
    malformed = MusicSemanticInterpreter(
        RuleBasedMusicActionParser(), FakeSemanticClient({
            "variants": [{"artist": "Artist", "title": "Title", "id": "invented"}],
        }),
    )
    excessive = MusicSemanticInterpreter(
        RuleBasedMusicActionParser(), FakeSemanticClient({
            "variants": [
                {"artist": f"Artist {index}", "title": f"Title {index}"}
                for index in range(5)
            ],
        }),
    )

    assert malformed.rewrite_track_queries("surface artist", "surface title") == ()
    assert excessive.rewrite_track_queries("surface artist", "surface title") == ()


def test_query_rewriter_provider_failure_returns_no_variants() -> None:
    interpreter = MusicSemanticInterpreter(
        RuleBasedMusicActionParser(), FakeSemanticClient(error=SemanticLlmError(
            "provider_error", "provider unavailable",
        )),
    )

    result = interpreter.rewrite_track_queries("surface artist", "surface title")

    assert result == ()
    assert interpreter.metrics.llm_calls == 1


def test_query_rewriter_prompt_prioritizes_cross_script_catalog_surfaces() -> None:
    client = FakeSemanticClient({"variants": []})
    interpreter = MusicSemanticInterpreter(RuleBasedMusicActionParser(), client)

    interpreter.rewrite_track_queries("localized artist", "localized title")

    request = client.requests[0]
    assert request.input == {
        "artist_surface": "localized artist", "title_surface": "localized title",
    }
    assert "original-language artist/title" in request.system_prompt
    assert "romanized forms" in request.system_prompt
    assert "Korean spelling correction alone is not" in request.system_prompt


def test_verb_like_korean_title_is_preserved_before_final_playback_command() -> None:
    client = FakeSemanticClient({
        "status": "parsed", "actions": [action_data(
            title="verb-like title", artist="artist surface", artist_explicit=True,
        )],
    })
    interpreter = MusicSemanticInterpreter(RuleBasedMusicActionParser(), client)

    result = interpreter.interpret("artist surface verb-like title 틀어줘")

    assert result.ok
    assert result.action.artist == "artist surface"
    assert result.action.title == "verb-like title"
    assert "title surface itself looks imperative" in client.requests[0].system_prompt


def test_rule_parser_keeps_generic_request_shaped_titles() -> None:
    parser = RuleBasedMusicActionParser()

    for title in ("노래해줘", "기억해줘", "기다려줘"):
        result = parser.parse(f"artist {title} 틀어줘")
        assert result.ok
        assert result.action.artist == "artist"
        assert result.action.title == title


def test_song_candidate_judge_can_only_return_supplied_index() -> None:
    client = FakeSemanticClient({
        "status": "match", "candidate_index": 1,
        "title_equivalent": True, "artist_equivalent": True,
    })
    interpreter = MusicSemanticInterpreter(RuleBasedMusicActionParser(), client)
    candidates = (
        MusicItem("wrong", "Wrong", "Artist"),
        MusicItem("right", "Canonical", "Artist", search_rank=6),
    )

    selected = interpreter.judge_song_candidate(MusicAction(
        MusicActionType.PLAY_SONG, title="translated", artist="spoken artist",
    ), candidates)

    assert selected is not None and selected.index == 1
    assert client.requests[0].task == "music_song_candidate_judgment"
    assert "item_id" not in str(client.requests[0].input)


def test_candidate_judge_rejects_uncertain_or_out_of_range_selection() -> None:
    uncertain = MusicSemanticInterpreter(
        RuleBasedMusicActionParser(),
        FakeSemanticClient({
            "status": "uncertain", "candidate_index": -1,
            "title_equivalent": False, "artist_equivalent": False,
        }),
    )
    invalid = MusicSemanticInterpreter(
        RuleBasedMusicActionParser(),
        FakeSemanticClient({
            "status": "match", "candidate_index": 4,
            "title_equivalent": True, "artist_equivalent": True,
        }),
    )
    candidates = (MusicItem("only", "Only", "Artist"),)
    action = MusicAction(MusicActionType.PLAY_SONG, title="surface")

    uncertain_result = uncertain.judge_song_candidate(action, candidates)
    invalid_result = invalid.judge_song_candidate(action, candidates)
    assert uncertain_result.rejection_reason == "no_match"
    assert invalid_result.rejection_reason == "invalid_index"


def test_song_candidate_judge_rejects_artist_only_match() -> None:
    interpreter = MusicSemanticInterpreter(
        RuleBasedMusicActionParser(),
        FakeSemanticClient({
            "status": "match", "candidate_index": 0,
            "title_equivalent": False, "artist_equivalent": True,
        }),
    )

    result = interpreter.judge_song_candidate(
        MusicAction(MusicActionType.PLAY_SONG, title="explicit", artist="artist"),
        (MusicItem("top", "Other", "Artist"),),
    )
    assert not result.approved
    assert result.rejection_reason == "title_not_equivalent"


def test_song_candidate_judge_preserves_artist_and_provider_failures() -> None:
    artist_mismatch = MusicSemanticInterpreter(
        RuleBasedMusicActionParser(), FakeSemanticClient({
            "status": "match", "candidate_index": 0,
            "title_equivalent": True, "artist_equivalent": False,
        }),
    ).judge_song_candidate(
        MusicAction(
            MusicActionType.PLAY_SONG, title="surface", artist="performer",
            artist_explicit=True,
        ),
        (MusicItem("candidate", "Canonical", "Other Performer"),),
    )
    provider_interpreter = MusicSemanticInterpreter(
        RuleBasedMusicActionParser(),
        FakeSemanticClient(error=SemanticLlmError(
            "provider_error",
            "Groq HTTP 400 request_id=safe-request "
            "type=invalid_request_error code=json_schema_invalid",
        )),
    )
    provider_failure = provider_interpreter.judge_song_candidate(
        MusicAction(MusicActionType.PLAY_SONG, title="surface"),
        (MusicItem("candidate", "Canonical", "Performer"),),
    )
    schema_failure = MusicSemanticInterpreter(
        RuleBasedMusicActionParser(), FakeSemanticClient(error=SemanticLlmError(
            "provider_error", "Groq HTTP 400 schema_mismatch=$.candidate_index:type",
        )),
    ).judge_song_candidate(
        MusicAction(MusicActionType.PLAY_SONG, title="surface"),
        (MusicItem("candidate", "Canonical", "Performer"),),
    )
    malformed = MusicSemanticInterpreter(
        RuleBasedMusicActionParser(), FakeSemanticClient(error=SemanticLlmError(
            "malformed_response", "invalid structured response",
        )),
    ).judge_song_candidate(
        MusicAction(MusicActionType.PLAY_SONG, title="surface"),
        (MusicItem("candidate", "Canonical", "Performer"),),
    )

    assert artist_mismatch.rejection_reason == "artist_not_equivalent"
    assert provider_failure.error_category == "provider_error"
    assert provider_interpreter.metrics.errors == 1
    assert provider_interpreter.metrics.last_error == (
        "Groq HTTP 400 request_id=safe-request "
        "type=invalid_request_error code=json_schema_invalid"
    )
    assert schema_failure.error_category == "schema_error"
    assert malformed.error_category == "malformed_response"


def test_playlist_candidate_judge_uses_existing_names_only() -> None:
    client = FakeSemanticClient({
        "status": "match", "candidate_index": 0,
        "title_equivalent": True, "artist_equivalent": True,
    })
    interpreter = MusicSemanticInterpreter(RuleBasedMusicActionParser(), client)
    candidates = (
        PlaylistItem("one", "Existing One"),
        PlaylistItem("two", "Existing Two"),
    )

    selected = interpreter.judge_playlist_candidate("spoken name", candidates)

    assert selected == 0
    assert client.requests[0].input["candidates"] == [
        {"index": 0, "name": "Existing One"},
        {"index": 1, "name": "Existing Two"},
    ]


def test_playlist_candidate_selection_returns_bounded_ambiguity() -> None:
    client = FakeSemanticClient({"status": "ambiguous", "candidate_index": -1})
    interpreter = MusicSemanticInterpreter(RuleBasedMusicActionParser(), client)

    result = interpreter.select_playlist_candidate(
        "자연어 축약", ("Existing One", "Existing Two"),
    )

    assert result.outcome == "ambiguous"
    assert not result.approved
    assert client.requests[0].schema == {
        "type": "object",
        "properties": {
            "status": {
                "type": "string", "enum": ["match", "ambiguous", "no_match"],
            },
            "candidate_index": {"type": "integer"},
        },
        "required": ["status", "candidate_index"],
        "additionalProperties": False,
    }


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


def test_provider_failure_candidate_is_routed_to_safe_tool_failure() -> None:
    client = FakeSemanticClient(error=SemanticLlmError("provider_error", "HTTP 400"))
    interpreter = MusicSemanticInterpreter(RuleBasedMusicActionParser(), client)
    router = create_default_semantic_router(default_app_registry(), interpreter)

    class MustNotExecuteController:
        def execute(self, action):
            raise AssertionError("failed interpretation must not execute")

    decision = router.route(RoutingRequest("아이묭 마리골드 틀어"))
    result = MusicControlTool(interpreter, MustNotExecuteController()).run(
        "아이묭 마리골드 틀어",
    )

    assert decision.required_capabilities == {"music_control"}
    assert not result.ok
    assert result.data["reason"] == "semantic_provider_error"
    assert "HTTP 400" in result.error


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
