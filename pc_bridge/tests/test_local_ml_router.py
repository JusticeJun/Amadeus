from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.routing import (
    CAPABILITY_NAMES,
    LocalMlSemanticRouter,
    ModelArtifactError,
    RoutingRequest,
)
from app.routing.defaults import DEFAULT_MODEL_PATH


def _write_artifact(path: Path, *, threshold: float = 0.5) -> Path:
    path.write_text(json.dumps({
        "schema_version": 1,
        "labels": ["weather"],
        "preprocessing": {"character_ngrams": {"minimum": 2, "maximum": 2}},
        "idf": {"비 ": 1.0},
        "models": {"weather": {"bias": 0.0, "threshold": threshold, "weights": {"비 ": 2.0}}},
    }, ensure_ascii=False), encoding="utf-8")
    return path


def test_local_ml_router_is_deterministic_and_returns_confidence(tmp_path: Path) -> None:
    router = LocalMlSemanticRouter(_write_artifact(tmp_path / "model.json"))

    first = router.route(RoutingRequest("비"))
    second = router.route(RoutingRequest("비"))

    assert first == second
    assert first.required_capabilities == {"weather"}
    assert first.matches[0].confidence is not None


def test_local_ml_router_respects_threshold_boundary(tmp_path: Path) -> None:
    accepted = LocalMlSemanticRouter(_write_artifact(tmp_path / "accepted.json", threshold=0.5))
    rejected = LocalMlSemanticRouter(_write_artifact(tmp_path / "rejected.json", threshold=0.500001))

    assert accepted.route(RoutingRequest("아무 말")).required_capabilities == {"weather"}
    assert not rejected.route(RoutingRequest("아무 말")).matches


@pytest.mark.parametrize("content", ["{}", "not-json"])
def test_local_ml_router_rejects_malformed_artifacts(tmp_path: Path, content: str) -> None:
    path = tmp_path / "broken.json"
    path.write_text(content, encoding="utf-8")

    with pytest.raises(ModelArtifactError):
        LocalMlSemanticRouter(path)


def test_local_ml_router_rejects_missing_artifact(tmp_path: Path) -> None:
    with pytest.raises(ModelArtifactError):
        LocalMlSemanticRouter(tmp_path / "missing.json")


def test_prepared_dataset_has_provenance_and_disjoint_splits() -> None:
    root = Path(__file__).resolve().parents[1] / "training" / "semantic_routing" / "prepared"
    splits = {}
    for split in ("train", "validation"):
        rows = [json.loads(line) for line in (root / f"{split}.jsonl").read_text(encoding="utf-8").splitlines()]
        assert all({"text", "capabilities", "source", "source_split", "source_intent", "adaptation"} <= row.keys() for row in rows)
        assert all(set(row["capabilities"]) <= set(CAPABILITY_NAMES) for row in rows)
        splits[split] = {"".join(row["text"].casefold().split()) for row in rows}
    assert not splits["train"] & splits["validation"]


def test_external_mapping_is_explicit_and_uses_capability_vocabulary() -> None:
    path = Path(__file__).resolve().parents[1] / "training" / "semantic_routing" / "sources.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    assert set(manifest["sources"]) == {
        "massive-1.1-ko-KR", "hwu64-2019-en", "clinc150-uci-full",
    }
    for source in manifest["sources"].values():
        mapping = source["intent_mapping"]
        assert mapping
        assert all(set(labels) <= set(CAPABILITY_NAMES) for labels in mapping.values())


def test_production_and_research_artifacts_are_explicitly_separate() -> None:
    research = DEFAULT_MODEL_PATH.with_name("semantic-router-v2-external-research.json")

    assert DEFAULT_MODEL_PATH.name == "semantic-router-v1.json"
    assert research.exists()
    assert LocalMlSemanticRouter(DEFAULT_MODEL_PATH)
    assert LocalMlSemanticRouter(research)


def test_versioned_model_predicts_multilabel_and_no_match_validation_cases() -> None:
    router = LocalMlSemanticRouter(DEFAULT_MODEL_PATH)
    path = Path(__file__).resolve().parents[1] / "training" / "semantic_routing" / "prepared" / "validation.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

    assert any(
        len(row["capabilities"]) > 1
        and router.route(RoutingRequest(row["text"])).required_capabilities == set(row["capabilities"])
        for row in rows
    )
    assert any(
        not row["capabilities"] and not router.route(RoutingRequest(row["text"])).matches
        for row in rows
    )
