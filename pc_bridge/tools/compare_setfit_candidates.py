"""Compare baseline and multi-label-augmented SetFit artifacts on frozen slices."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
import time
import warnings

import numpy as np


BRIDGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BRIDGE_ROOT))

from app.models import ChatMessage  # noqa: E402
from app.pc_control import default_app_registry  # noqa: E402
from app.routing import CAPABILITY_NAMES, RoutingRequest, create_rule_based_semantic_router  # noqa: E402
from evaluation import load_corpora  # noqa: E402
from tools.evaluate_setfit_semantic_router import (  # noqa: E402
    directory_fingerprint,
    load_rows,
    metrics,
    targets_for,
)


BASELINE = BRIDGE_ROOT / "research_artifacts" / "semantic-router-v4-setfit"
CANDIDATE = BRIDGE_ROOT / "research_artifacts" / "semantic-router-v5-setfit-multilabel-augmented"
PREPARED = BRIDGE_ROOT / "training" / "semantic_routing" / "prepared"
EVALUATION = BRIDGE_ROOT / "evaluation" / "cases"
MANUAL_CASES = (
    ("오늘 밖에서 공부해도 괜찮을까?", ()),
    ("오늘 반팔만 입어도 괜찮을까?", ("weather",)),
    ("나가려는데 겉옷 챙기는 게 좋을까?", ("weather",)),
    ("오늘 반팔만 입어도 될지 알려주고 신나는 노래도 하나 틀어줘", ("weather", "music_control")),
    ("오늘 반팔 입어도 될지 알려주고 컴퓨터 소리도 좀 줄여줘", ("weather", "pc_control")),
    ("신나는 노래 하나 틀어주고 컴퓨터 소리도 조금 줄여줘", ("music_control", "pc_control")),
)


class Candidate:
    def __init__(self, path: Path) -> None:
        from setfit import SetFitModel
        from transformers.utils import logging as transformers_logging

        transformers_logging.set_verbosity_error()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            self.model = SetFitModel.from_pretrained(path, local_files_only=True)
        self.path = path
        self.metadata = json.loads((path / "amadeus_metadata.json").read_text(encoding="utf-8"))
        self.thresholds = np.asarray([
            self.metadata["thresholds"][label] for label in CAPABILITY_NAMES
        ])

    def scores(self, texts: list[str]) -> np.ndarray:
        return np.asarray(self.model.predict_proba(texts, batch_size=128, show_progress_bar=False))


def score_slices(targets: np.ndarray, predicted: np.ndarray) -> dict[str, object]:
    single = np.sum(targets, axis=1) == 1
    oos = np.sum(targets, axis=1) == 0
    multi = np.sum(targets, axis=1) > 1
    result = {"all": metrics(targets, predicted)}
    if np.any(single):
        result["single_label"] = metrics(targets[single], predicted[single])
    if np.any(oos):
        result["oos"] = metrics(targets[oos], predicted[oos])
    if np.any(multi):
        result["multi_label"] = metrics(targets[multi], predicted[multi])
    return result


def absolute_deltas(baseline: object, candidate: object) -> object:
    if isinstance(baseline, dict) and isinstance(candidate, dict):
        return {
            key: absolute_deltas(baseline[key], candidate[key])
            for key in baseline.keys() & candidate.keys()
        }
    if isinstance(baseline, (int, float)) and isinstance(candidate, (int, float)):
        return candidate - baseline
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, default=BASELINE)
    parser.add_argument("--candidate", type=Path, default=CANDIDATE)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--latency-iterations", type=int, default=100)
    args = parser.parse_args()
    models = {"baseline": Candidate(args.baseline), "augmented": Candidate(args.candidate)}
    results: dict[str, dict[str, object]] = {name: {} for name in models}

    for split in ("validation", "external_test"):
        rows = load_rows(PREPARED / f"{split}.jsonl")
        texts = [str(row["text"]) for row in rows]
        targets = targets_for([list(row["capabilities"]) for row in rows])
        for name, model in models.items():
            predicted = model.scores(texts) >= model.thresholds
            results[name][split] = score_slices(targets, predicted)

    cases = [
        case for case in load_corpora(sorted(EVALUATION.glob("*.jsonl")))
        if case.expected_tools is not None
    ]
    texts = []
    for case in cases:
        history = " ".join(turn.content for turn in case.context[-2:])
        texts.append(f"{history} [current] {case.text}" if history else case.text)
    targets = targets_for([list(case.expected_tools or ()) for case in cases])
    boundary = np.asarray(["boundary_slice" in case.tags for case in cases])
    pair_masks = {
        "+".join(pair): np.asarray([set(case.expected_tools or ()) == set(pair) for case in cases])
        for pair in (
            ("weather", "music_control"),
            ("weather", "pc_control"),
            ("music_control", "pc_control"),
        )
    }
    rule_router = create_rule_based_semantic_router(default_app_registry())
    for name, model in models.items():
        scores = model.scores(texts)
        predicted = scores >= model.thresholds
        results[name]["independent_holdout"] = score_slices(targets, predicted)
        results[name]["weather_boundary"] = metrics(targets[boundary], predicted[boundary])
        results[name]["multi_label_pairs"] = {
            pair: metrics(targets[mask], predicted[mask]) for pair, mask in pair_masks.items()
        }
        hybrid = np.zeros_like(predicted)
        for index, case in enumerate(cases):
            history = tuple(ChatMessage(turn.role, turn.content) for turn in case.context)
            decision = rule_router.route(RoutingRequest(case.text, history))
            rule_labels = decision.required_capabilities
            if rule_labels:
                hybrid[index] = [label in rule_labels for label in CAPABILITY_NAMES]
            elif predicted[index, 0]:
                hybrid[index, 0] = True
        results[name]["independent_holdout_hybrid"] = metrics(targets, hybrid)

        manual_scores = model.scores([text for text, _ in MANUAL_CASES])
        results[name]["manual_cases"] = [
            {
                "input": text,
                "expected_labels": list(expected),
                "raw_scores": dict(zip(CAPABILITY_NAMES, (float(value) for value in row))),
                "thresholds": dict(zip(CAPABILITY_NAMES, (float(value) for value in model.thresholds))),
                "accepted_labels": [
                    label for label, score, threshold in zip(CAPABILITY_NAMES, row, model.thresholds)
                    if score >= threshold
                ],
            }
            for (text, expected), row in zip(MANUAL_CASES, manual_scores)
        ]

        sample = MANUAL_CASES[1][0]
        model.scores([sample])
        timings = []
        for _ in range(args.latency_iterations):
            started = time.perf_counter()
            model.scores([sample])
            timings.append((time.perf_counter() - started) * 1000)
        size, fingerprint = directory_fingerprint(model.path)
        results[name]["artifact"] = {
            "thresholds": dict(zip(CAPABILITY_NAMES, (float(value) for value in model.thresholds))),
            "bytes": size,
            "classifier_head_bytes": (model.path / "model_head.pkl").stat().st_size,
            "sha256": fingerprint,
            "warm_cpu_mean_ms": statistics.mean(timings),
            "warm_cpu_p95_ms": sorted(timings)[int(len(timings) * 0.95) - 1],
        }

    report = {
        "baseline": results["baseline"],
        "augmented": results["augmented"],
        "absolute_deltas_augmented_minus_baseline": absolute_deltas(
            results["baseline"], results["augmented"]
        ),
    }
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(payload, encoding="utf-8")
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
