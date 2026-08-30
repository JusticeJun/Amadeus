from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.routing import RoutingRequest, SentenceMlSemanticRouter
from app.routing.composite import CapabilityFilterSemanticRouter


class FakeEncoder:
    def __init__(self, vector) -> None:
        self.vector = vector
        self.calls = []

    def encode(self, sentences, *, normalize_embeddings, show_progress_bar=False):
        self.calls.append((sentences, normalize_embeddings, show_progress_bar))
        return self.vector


def write_artifact(path: Path, threshold: float = 0.5) -> Path:
    path.write_text(json.dumps({
        "schema_version": 1,
        "labels": ["weather", "music_control"],
        "encoder": {"name": "fake", "dimension": 2},
        "models": {
            "weather": {"intercept": 0.0, "threshold": threshold, "coefficients": [2.0, 0.0]},
            "music_control": {"intercept": 0.0, "threshold": 0.9, "coefficients": [0.0, 2.0]},
        },
    }), encoding="utf-8")
    return path


def test_sentence_router_uses_normalized_embedding_and_threshold(tmp_path: Path) -> None:
    encoder = FakeEncoder([1.0, 0.0])
    router = SentenceMlSemanticRouter(write_artifact(tmp_path / "model.json"), encoder)

    decision = router.route(RoutingRequest("간접 날씨 질문"))

    assert decision.required_capabilities == {"weather"}
    assert encoder.calls == [("간접 날씨 질문", True, False)]


def test_sentence_router_rejects_all_labels_as_no_match(tmp_path: Path) -> None:
    router = SentenceMlSemanticRouter(
        write_artifact(tmp_path / "model.json", threshold=0.9),
        FakeEncoder([0.0, 0.0]),
    )

    assert not router.route(RoutingRequest("일반 대화")).matches


def test_sentence_router_rejects_encoder_dimension_mismatch(tmp_path: Path) -> None:
    router = SentenceMlSemanticRouter(
        write_artifact(tmp_path / "model.json"),
        FakeEncoder([1.0]),
    )

    with pytest.raises(ValueError, match="unexpected embedding dimension"):
        router.route(RoutingRequest("날씨"))


def test_execution_policy_blocks_sentence_ml_side_effect_labels(tmp_path: Path) -> None:
    raw = SentenceMlSemanticRouter(
        write_artifact(tmp_path / "model.json"),
        FakeEncoder([1.0, 2.0]),
    )
    safe = CapabilityFilterSemanticRouter(raw, frozenset({"weather"}))

    assert raw.route(RoutingRequest("ambiguous semantic request")).required_capabilities == {
        "weather", "music_control",
    }
    assert safe.route(RoutingRequest("ambiguous semantic request")).required_capabilities == {
        "weather",
    }
