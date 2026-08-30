"""Evaluate the frozen SetFit research candidate without tuning on holdouts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time

import numpy as np


BRIDGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BRIDGE_ROOT))

from app.models import ChatMessage  # noqa: E402
from app.pc_control import default_app_registry  # noqa: E402
from app.routing import CAPABILITY_NAMES, RoutingRequest, create_rule_based_semantic_router  # noqa: E402
from evaluation import load_corpora  # noqa: E402


DEFAULT_MODEL = BRIDGE_ROOT / "research_artifacts" / "semantic-router-v4-setfit"
PREPARED = BRIDGE_ROOT / "training" / "semantic_routing" / "prepared"
EVALUATION = BRIDGE_ROOT / "evaluation" / "cases"


def load_rows(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def metrics(targets: np.ndarray, predicted: np.ndarray) -> dict[str, object]:
    result: dict[str, object] = {}
    true_positive = int(np.sum(targets & predicted))
    false_positive = int(np.sum(~targets & predicted))
    false_negative = int(np.sum(targets & ~predicted))
    result["micro"] = prf(true_positive, false_positive, false_negative)
    result["exact_match"] = float(np.mean(np.all(targets == predicted, axis=1)))
    result["labels"] = {
        label: prf(
            int(np.sum(targets[:, index] & predicted[:, index])),
            int(np.sum(~targets[:, index] & predicted[:, index])),
            int(np.sum(targets[:, index] & ~predicted[:, index])),
        )
        for index, label in enumerate(CAPABILITY_NAMES)
    }
    no_match = ~np.any(targets, axis=1)
    result["no_match"] = {
        "cases": int(np.sum(no_match)),
        "correct_rejections": int(np.sum(no_match & ~np.any(predicted, axis=1))),
        "false_promotions": int(np.sum(no_match & np.any(predicted, axis=1))),
    }
    return result


def prf(tp: int, fp: int, fn: int) -> dict[str, float | int]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def targets_for(capabilities: list[list[str]]) -> np.ndarray:
    return np.asarray([[label in labels for label in CAPABILITY_NAMES] for labels in capabilities])


def directory_fingerprint(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    for item in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
        payload = item.read_bytes()
        size += len(payload)
        digest.update(item.relative_to(path).as_posix().encode())
        digest.update(b"\0")
        digest.update(payload)
    return size, digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--latency-iterations", type=int, default=100)
    args = parser.parse_args()
    from setfit import SetFitModel

    metadata = json.loads((args.model / "amadeus_metadata.json").read_text(encoding="utf-8"))
    model = SetFitModel.from_pretrained(args.model, local_files_only=True)
    thresholds = np.asarray([metadata["thresholds"][label] for label in CAPABILITY_NAMES])

    def predict(texts: list[str]) -> tuple[np.ndarray, np.ndarray]:
        scores = np.asarray(model.predict_proba(texts, batch_size=args.batch_size, show_progress_bar=True))
        return scores, scores >= thresholds

    report: dict[str, object] = {"model": metadata}
    for split in ("validation", "external_test"):
        rows = load_rows(PREPARED / f"{split}.jsonl")
        _, predicted = predict([str(row["text"]) for row in rows])
        report[split] = metrics(
            targets_for([list(row["capabilities"]) for row in rows]),
            predicted,
        )

    cases = [
        case for case in load_corpora(sorted(EVALUATION.glob("*.jsonl")))
        if case.expected_tools is not None
    ]
    texts = []
    for case in cases:
        history = " ".join(turn.content for turn in case.context[-2:])
        texts.append(f"{history} [current] {case.text}" if history else case.text)
    _, standalone = predict(texts)
    holdout_targets = targets_for([list(case.expected_tools or ()) for case in cases])
    report["independent_holdout_standalone"] = metrics(holdout_targets, standalone)

    rule_router = create_rule_based_semantic_router(default_app_registry())
    hybrid = np.zeros_like(standalone)
    for index, case in enumerate(cases):
        history = tuple(ChatMessage(turn.role, turn.content) for turn in case.context)
        decision = rule_router.route(RoutingRequest(case.text, history))
        rule_labels = set(decision.required_capabilities)
        if rule_labels:
            hybrid[index] = [label in rule_labels for label in CAPABILITY_NAMES]
        elif standalone[index, 0]:
            hybrid[index, 0] = True
    report["independent_holdout_hybrid"] = metrics(holdout_targets, hybrid)

    multi = np.sum(holdout_targets, axis=1) > 1
    report["multi_label_holdout"] = metrics(holdout_targets[multi], standalone[multi])
    report["multi_label_pairs"] = {
        "+".join(pair): metrics(holdout_targets[mask], standalone[mask])
        for pair in (
            ("weather", "music_control"),
            ("weather", "pc_control"),
            ("music_control", "pc_control"),
        )
        if np.any(mask := np.asarray([
            set(case.expected_tools or ()) == set(pair) for case in cases
        ]))
    }
    boundary = np.asarray(["boundary_slice" in case.tags for case in cases])
    report["weather_boundary"] = metrics(holdout_targets[boundary], standalone[boundary])

    sample = "오늘 반팔만 입어도 괜찮을까?"
    model.predict_proba([sample], show_progress_bar=False)
    timings = []
    for _ in range(args.latency_iterations):
        started = time.perf_counter()
        model.predict_proba([sample], show_progress_bar=False)
        timings.append((time.perf_counter() - started) * 1000)
    size, sha256 = directory_fingerprint(args.model)
    head = args.model / "model_head.pkl"
    report["runtime"] = {
        "warm_cpu_mean_ms": statistics.mean(timings),
        "warm_cpu_p95_ms": sorted(timings)[int(len(timings) * 0.95) - 1],
        "artifact_bytes": size,
        "classifier_head_bytes": head.stat().st_size,
        "artifact_sha256": sha256,
    }
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(payload, encoding="utf-8")
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
