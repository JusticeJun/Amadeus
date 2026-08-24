from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.models import ChatMessage
from app.routing import RoutingRequest, RuleBasedSemanticRouter, matches_weather_request
from evaluation import (
    RoutingCase,
    RoutingContext,
    evaluate_routing,
    load_corpora,
    load_corpus,
)


CORPUS = Path(__file__).resolve().parents[1] / "evaluation" / "cases" / "weather.jsonl"


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
    assert (weather.true_positive, weather.false_positive, weather.false_negative) == (28, 8, 26)
