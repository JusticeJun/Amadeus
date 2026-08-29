from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.models import ChatMessage
from app.pc_control import RuleBasedPcActionParser, default_app_registry
from app.routing import (
    RoutingRequest,
    RuleBasedSemanticRouter,
    create_default_semantic_router,
    matches_weather_request,
)
from app.tools import ToolExecutor
from evaluation import (
    RoutingCase,
    RoutingContext,
    evaluate_routing,
    load_corpora,
    load_corpus,
)


CORPUS = Path(__file__).resolve().parents[1] / "evaluation" / "cases" / "weather.jsonl"
PC_CORPUS = CORPUS.with_name("pc_control.jsonl")
CROSS_CORPUS = CORPUS.with_name("cross_capability.jsonl")
MUSIC_CORPUS = CORPUS.with_name("music_control.jsonl")
MUSIC_CROSS_CORPUS = CORPUS.with_name("music_cross_capability.jsonl")


def test_corpus_covers_semantic_boundary_categories() -> None:
    cases = load_corpus(CORPUS)
    categories = {case.category for case in cases}
    assert {
        "explicit_positive", "negative", "hard_negative", "implicit_positive",
        "mixed_intent", "regression", "context_required", "unsupported_capability",
        "ambiguous",
    } <= categories
    labels = {case.text: case.expected_tools for case in cases}
    assert labels["오늘 날씨 어때?"] == frozenset({"weather"})
    assert labels["오늘 날씨가 안 좋아서 그런가 기분이 안 좋네."] == frozenset()
    assert labels["밖에 추워?"] == frozenset({"weather"})
    assert labels["'추워'와 '쌀쌀해'의 차이가 뭐야?"] == frozenset()
    assert labels["오늘 날씨 좋네."] == frozenset()
    assert labels["오늘 날씨 좋지?"] == frozenset({"weather"})
    assert labels["오늘 선크림 발라야 할까?"] == frozenset({"uv"})
    assert labels["어제 날씨 어땠어?"] == frozenset({"weather_history"})
    assert labels["태풍 지금 어디쯤이야?"] == frozenset({"typhoon_tracking"})
    assert any(case.expected_tools is None for case in cases)
    assert any(case.context for case in cases)
    assert any("minimal_pair" in case.tags for case in cases)
    assert any(case.reason for case in cases)


def test_evaluator_calculates_multilabel_errors_and_excludes_ambiguous_cases() -> None:
    cases = [
        RoutingCase("tp", "weather yes", frozenset({"weather"}), "positive"),
        RoutingCase("fp", "weather no", frozenset(), "hard_negative"),
        RoutingCase("fn", "implicit weather", frozenset({"weather"}), "implicit_positive"),
        RoutingCase("tn", "plain chat", frozenset(), "negative"),
        RoutingCase(
            "multi", "weather and heat", frozenset({"weather", "hardware_control"}),
            "mixed_intent",
        ),
        RoutingCase("ambiguous", "maybe", None, "ambiguous"),
    ]
    predicted = {
        "weather yes": {"weather"},
        "weather no": {"weather"},
        "implicit weather": set(),
        "plain chat": set(),
        "weather and heat": {"weather", "hardware_control"},
        "maybe": {"weather"},
    }
    report = evaluate_routing(cases, lambda case: predicted[case.text], latency_iterations=1)
    weather = report.tool_metrics["weather"]
    hardware = report.tool_metrics["hardware_control"]
    assert (weather.true_positive, weather.false_positive) == (2, 1)
    assert (weather.false_negative, weather.true_negative) == (1, 1)
    assert weather.precision == pytest.approx(2 / 3)
    assert weather.recall == pytest.approx(2 / 3)
    assert hardware.precision == pytest.approx(1.0)
    assert hardware.recall == pytest.approx(1.0)
    assert report.scored_cases == 5
    assert report.ambiguous_cases == 1
    assert [item.case_id for item in report.false_positives] == ["fp"]
    assert [item.case_id for item in report.false_negatives] == ["fn"]
    assert report.latency.samples == len(cases)


def test_report_separates_standard_context_unsupported_and_ambiguous_slices() -> None:
    cases = [
        RoutingCase("standard", "today weather", frozenset({"weather"}), "explicit_positive"),
        RoutingCase(
            "context", "tomorrow?", frozenset({"weather"}), "context_required",
            context=(RoutingContext("user", "It is raining today."),),
        ),
        RoutingCase(
            "unsupported", "UV now", frozenset({"uv"}), "unsupported_capability",
        ),
        RoutingCase("ambiguous", "Is it okay?", None, "ambiguous"),
    ]
    predictions = {"today weather": {"weather"}}
    report = evaluate_routing(
        cases,
        lambda case: predictions.get(case.text, set()),
        latency_iterations=1,
    )
    assert report.slice_metrics["standard"].cases == 1
    assert report.slice_metrics["standard"].tool_metrics["weather"].recall == 1.0
    assert report.slice_metrics["context_required"].cases == 1
    assert report.slice_metrics["context_required"].tool_metrics["weather"].recall == 0.0
    assert report.slice_metrics["unsupported_capability"].cases == 1
    assert report.slice_metrics["unsupported_capability"].tool_metrics["uv"].recall == 0.0
    assert report.slice_metrics["ambiguous"].cases == 1
    assert report.slice_metrics["ambiguous"].scored_cases == 0


def test_corpus_loader_rejects_duplicate_ids(tmp_path: Path) -> None:
    entry = {"id": "duplicate", "text": "오늘 날씨 어때?", "expected_tools": ["weather"], "category": "positive"}
    path = tmp_path / "duplicate.jsonl"
    path.write_text("\n".join((json.dumps(entry), json.dumps(entry))), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate corpus id"):
        load_corpus(path)


def test_multiple_capability_corpora_reject_cross_file_duplicate_ids(tmp_path: Path) -> None:
    entry = {"id": "shared", "text": "sample", "expected_tools": [], "category": "negative"}
    first = tmp_path / "weather.jsonl"
    second = tmp_path / "calendar.jsonl"
    first.write_text(json.dumps(entry), encoding="utf-8")
    second.write_text(json.dumps(entry), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate corpus id across files"):
        load_corpora((first, second))


def test_rule_router_preserves_weather_evaluation_baseline() -> None:
    router = RuleBasedSemanticRouter({"weather": matches_weather_request})

    def predict(case: RoutingCase) -> set[str]:
        history = tuple(ChatMessage(turn.role, turn.content) for turn in case.context)
        decision = router.route(RoutingRequest(case.text, history))
        return set(decision.required_capabilities)

    report = evaluate_routing(load_corpus(CORPUS), predict, latency_iterations=1)
    weather = report.tool_metrics["weather"]
    assert (weather.true_positive, weather.false_positive, weather.false_negative) == (29, 8, 26)


def test_pc_and_cross_corpora_cover_multicapability_boundaries() -> None:
    pc_cases = load_corpus(PC_CORPUS)
    cross_cases = load_corpus(CROSS_CORPUS)

    assert {
        "explicit_positive", "implicit_positive", "hard_negative", "minimal_pair",
        "context_required", "ambiguous", "unsupported_action", "security_boundary",
    } <= {
        case.category for case in pc_cases
    }
    assert len(pc_cases) == 79
    assert len(cross_cases) == 34
    assert sum("minimal_pair" in case.tags for case in pc_cases) >= 7
    assert any(case.expected_tools == frozenset({"weather", "pc_control"}) for case in cross_cases)
    assert any(case.category == "planning_required" for case in cross_cases)
    assert any(case.expected_tools == frozenset() for case in cross_cases)


def test_expanded_corpora_preserve_current_rule_router_baseline() -> None:
    router = create_default_semantic_router(default_app_registry())

    def predict(case: RoutingCase) -> set[str]:
        history = tuple(ChatMessage(turn.role, turn.content) for turn in case.context)
        return set(router.route(RoutingRequest(case.text, history)).required_capabilities)

    pc_report = evaluate_routing(load_corpus(PC_CORPUS), predict, latency_iterations=1)
    cross_report = evaluate_routing(load_corpus(CROSS_CORPUS), predict, latency_iterations=1)

    pc = pc_report.tool_metrics["pc_control"]
    music = pc_report.tool_metrics["music_control"]
    assert (pc.true_positive, pc.false_positive, pc.false_negative) == (40, 0, 1)
    assert (music.true_positive, music.false_positive, music.false_negative) == (9, 0, 0)
    cross_pc = cross_report.tool_metrics["pc_control"]
    cross_music = cross_report.tool_metrics["music_control"]
    cross_weather = cross_report.tool_metrics["weather"]
    assert (cross_pc.true_positive, cross_pc.false_positive, cross_pc.false_negative) == (19, 0, 0)
    assert (cross_music.true_positive, cross_music.false_positive, cross_music.false_negative) == (3, 0, 0)
    assert (
        cross_weather.true_positive,
        cross_weather.false_positive,
        cross_weather.false_negative,
    ) == (17, 3, 2)


def test_all_planning_cases_preserve_capabilities_and_block_execution() -> None:
    router = create_default_semantic_router(default_app_registry())
    cases = [
        case for case in load_corpora((CROSS_CORPUS, MUSIC_CROSS_CORPUS))
        if case.category == "planning_required"
    ]
    decisions = []
    for case in cases:
        history = tuple(ChatMessage(turn.role, turn.content) for turn in case.context)
        decisions.append(router.route(RoutingRequest(case.text, history)))

    assert len(cases) == 10
    assert all(decision.planning_required for decision in decisions)
    assert all(
        decision.required_capabilities == case.expected_tools
        for case, decision in zip(cases, decisions)
    )
    class MustNotRunTool:
        name = "pc_control"

        def run(self, user_text):
            raise AssertionError(f"PC side effect attempted for: {user_text}")

        def build_llm_context(self, result):
            raise AssertionError("planning-required result should not exist")

    class MustNotRunMusicTool(MustNotRunTool):
        name = "music_control"

    executor = ToolExecutor((MustNotRunTool(), MustNotRunMusicTool()))
    assert all(executor.execute(decision, case.text) == () for case, decision in zip(
        cases, decisions,
    ))


def test_corpus_distinguishes_capability_routing_from_execution_support() -> None:
    apps = default_app_registry()
    router = create_default_semantic_router(apps)
    parser = RuleBasedPcActionParser(apps)
    cases = load_corpus(PC_CORPUS)
    unsupported = [
        case for case in cases
        if case.category in {"unsupported_action", "security_boundary"}
    ]

    assert unsupported
    assert all(
        "pc_control" in router.route(RoutingRequest(case.text)).required_capabilities
        for case in unsupported
    )
    assert all(not parser.parse(case.text).ok for case in unsupported)


def test_corpus_preserves_paths_natural_urls_and_toggle_semantics() -> None:
    cases = {case.case_id: case for case in load_corpus(PC_CORPUS)}

    assert cases["pc_security_003"].text == r"메모장 켜고 C:\password.txt 열어줘"
    assert cases["pc_security_005"].text == r"계산기 실행한 다음 del C:\temp\data.txt 해줘"
    assert cases["pc_security_006"].text == "크롬 켜고 example.com도 열어줘"
    assert not any("](" in case.text for case in cases.values())
    assert "capability_migration" in cases["pc_media_positive_005"].tags


def test_weather_matcher_predictions_are_unchanged_by_default_router() -> None:
    apps = default_app_registry()
    default_router = create_default_semantic_router(apps)
    old_router = RuleBasedSemanticRouter({"weather": matches_weather_request})

    for case in load_corpus(CORPUS):
        request = RoutingRequest(case.text)
        old_weather = "weather" in old_router.route(request).required_capabilities
        new_weather = "weather" in default_router.route(request).required_capabilities
        assert new_weather == old_weather, case.case_id


def test_music_corpora_cover_actions_hard_negatives_and_multilabel_cases() -> None:
    music = load_corpus(MUSIC_CORPUS)
    cross = load_corpus(MUSIC_CROSS_CORPUS)

    assert {"explicit_positive", "hard_negative", "minimal_pair", "ambiguous"} <= {
        case.category for case in music
    }
    assert any("playlist_track" in case.tags for case in music)
    assert any(case.expected_tools == {"weather", "music_control"} for case in cross)
    assert any(
        case.expected_tools == {"weather", "pc_control", "music_control"}
        for case in cross
    )
    assert sum(case.category == "planning_required" for case in cross) == 3
