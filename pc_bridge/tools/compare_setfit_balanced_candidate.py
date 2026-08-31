"""Compare all three SetFit corpus experiments on frozen and reviewed slices."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics
import time

import numpy as np

from app.models import ChatMessage
from app.pc_control import default_app_registry
from app.routing import CAPABILITY_NAMES, RoutingRequest, create_rule_based_semantic_router
from evaluation import load_corpora
from tools.compare_setfit_candidates import Candidate, MANUAL_CASES, score_slices
from tools.evaluate_setfit_semantic_router import directory_fingerprint, load_rows, metrics, targets_for


BRIDGE_ROOT = Path(__file__).resolve().parents[1]
PREPARED = BRIDGE_ROOT / "training" / "semantic_routing" / "prepared"
BALANCED_DATA = BRIDGE_ROOT / "research_data" / "semantic-routing-setfit-balanced"
EVALUATION = BRIDGE_ROOT / "evaluation" / "cases"
MODEL_PATHS = {
    "original": BRIDGE_ROOT / "research_artifacts" / "semantic-router-v4-setfit",
    "positive_only": BRIDGE_ROOT / "research_artifacts" / "semantic-router-v5-setfit-multilabel-augmented",
    "balanced": BRIDGE_ROOT / "research_artifacts" / "semantic-router-v6-setfit-balanced",
}
SOURCE = "amadeus-reviewed-semantic-routing-v1"


def grouped_metrics(
    rows: list[dict[str, object]], targets: np.ndarray, predicted: np.ndarray, field: str
) -> dict[str, object]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        groups[str(row["semantic"][field])].append(index)
    return {
        value: metrics(targets[indices], predicted[indices])
        for value, indices in sorted(groups.items())
    }


def semantic_development_report(
    model: Candidate, rows: list[dict[str, object]]
) -> dict[str, object]:
    targets = targets_for([list(row["capabilities"]) for row in rows])
    predicted = model.scores([str(row["text"]) for row in rows]) >= model.thresholds
    pair_groups: dict[str, list[int]] = defaultdict(list)
    pair_composition_groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        domains = list(row["semantic"]["domains"])
        if len(domains) == 2:
            pair = "+".join(sorted(domains))
            pair_groups[pair].append(index)
            pair_composition_groups[(pair, str(row["semantic"]["composition"]))].append(index)
    return {
        "all": score_slices(targets, predicted),
        "interaction": grouped_metrics(rows, targets, predicted, "interaction"),
        "request_form": grouped_metrics(rows, targets, predicted, "request_form"),
        "routing_role": grouped_metrics(rows, targets, predicted, "routing_role"),
        "composition": grouped_metrics(rows, targets, predicted, "composition"),
        "capability_pairs": {
            pair: metrics(targets[indices], predicted[indices])
            for pair, indices in sorted(pair_groups.items())
        },
        "pair_compositions": {
            pair: {
                composition: metrics(targets[indices], predicted[indices])
                for (group_pair, composition), indices in sorted(pair_composition_groups.items())
                if group_pair == pair
            }
            for pair in sorted(pair_groups)
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original", type=Path, default=MODEL_PATHS["original"])
    parser.add_argument("--positive-only", type=Path, default=MODEL_PATHS["positive_only"])
    parser.add_argument("--balanced", type=Path, default=MODEL_PATHS["balanced"])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--latency-iterations", type=int, default=100)
    args = parser.parse_args()
    models = {
        "original": Candidate(args.original),
        "positive_only": Candidate(args.positive_only),
        "balanced": Candidate(args.balanced),
    }
    results: dict[str, dict[str, object]] = {name: {} for name in models}

    for split in ("validation", "external_test"):
        rows = load_rows(PREPARED / f"{split}.jsonl")
        texts = [str(row["text"]) for row in rows]
        targets = targets_for([list(row["capabilities"]) for row in rows])
        for name, model in models.items():
            predicted = model.scores(texts) >= model.thresholds
            results[name][split] = score_slices(targets, predicted)

    semantic_rows = [
        row for row in load_rows(BALANCED_DATA / "validation.jsonl")
        if row.get("source") == SOURCE
    ]
    for name, model in models.items():
        results[name]["reviewed_semantic_development"] = semantic_development_report(
            model, semantic_rows
        )

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
        "models": results,
        "reviewed_slice_note": "development validation used for threshold selection; not independent holdout",
    }
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(payload, encoding="utf-8")
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
