from __future__ import annotations

import json
from pathlib import Path

import pytest

from evaluation import RoutingCase, evaluate_routing, load_corpus


CORPUS = Path(__file__).resolve().parents[1] / "evaluation" / "intent_routing.jsonl"


def test_corpus_covers_semantic_boundary_categories() -> None:
    cases = load_corpus(CORPUS)
    categories = {case.category for case in cases}
    assert {
        "positive", "negative", "hard_negative", "implicit_positive",
        "mixed_intent", "regression", "ambiguous",
    } <= categories
    labels = {case.text: case.expected_tools for case in cases}
    assert labels["오늘 날씨 어때?"] == frozenset({"weather"})
    assert labels["오늘 날씨가 안 좋아서 그런가 기분이 안 좋네."] == frozenset()
    assert labels["밖에 추워?"] == frozenset({"weather"})
    assert labels["'추워'와 '쌀쌀해'의 차이가 뭐야?"] == frozenset()
    assert any(case.expected_tools is None for case in cases)


def test_evaluator_calculates_multilabel_errors_and_excludes_ambiguous_cases() -> None:
    cases = [
        RoutingCase("tp", "weather yes", frozenset({"weather"}), "positive"),
        RoutingCase("fp", "weather no", frozenset(), "hard_negative"),
        RoutingCase("fn", "implicit weather", frozenset({"weather"}), "implicit_positive"),
        RoutingCase("tn", "plain chat", frozenset(), "negative"),
        RoutingCase("ambiguous", "maybe", None, "ambiguous"),
    ]
    predicted = {
        "weather yes": {"weather"},
        "weather no": {"weather"},
        "implicit weather": set(),
        "plain chat": set(),
        "maybe": {"weather"},
    }
    report = evaluate_routing(cases, predicted.__getitem__, latency_iterations=1)
    weather = report.tool_metrics["weather"]
    assert (weather.true_positive, weather.false_positive) == (1, 1)
    assert (weather.false_negative, weather.true_negative) == (1, 1)
    assert weather.precision == pytest.approx(0.5)
    assert weather.recall == pytest.approx(0.5)
    assert report.scored_cases == 4
    assert report.ambiguous_cases == 1
    assert [item.case_id for item in report.false_positives] == ["fp"]
    assert [item.case_id for item in report.false_negatives] == ["fn"]
    assert report.latency.samples == len(cases)


def test_corpus_loader_rejects_duplicate_ids(tmp_path: Path) -> None:
    entry = {"id": "duplicate", "text": "오늘 날씨 어때?", "expected_tools": ["weather"], "category": "positive"}
    path = tmp_path / "duplicate.jsonl"
    path.write_text("\n".join((json.dumps(entry), json.dumps(entry))), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate corpus id"):
        load_corpus(path)
