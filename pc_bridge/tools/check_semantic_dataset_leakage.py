"""Check exact normalized leakage against independent evaluation without tuning on it."""

from __future__ import annotations

import json
from pathlib import Path
import unicodedata


BRIDGE_ROOT = Path(__file__).resolve().parents[1]


def normalize(text: str) -> str:
    return "".join(unicodedata.normalize("NFKC", text).casefold().split())


def texts(paths: list[Path]) -> set[str]:
    result = set()
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                result.add(normalize(str(json.loads(line)["text"])))
    return result


def trigrams(text: str) -> frozenset[str]:
    value = normalize(text)
    return frozenset(value[index:index + 3] for index in range(max(0, len(value) - 2)))


def near_overlap(left: set[str], right: set[str], threshold: float = 0.85) -> tuple[int, float]:
    right_features = [(value, trigrams(value)) for value in right]
    matches = 0
    maximum = 0.0
    for value in left:
        features = trigrams(value)
        if not features:
            continue
        for other, other_features in right_features:
            if not other_features or min(len(value), len(other)) / max(len(value), len(other)) < 0.65:
                continue
            score = len(features & other_features) / len(features | other_features)
            maximum = max(maximum, score)
            matches += score >= threshold
    return matches, maximum


def main() -> int:
    prepared = BRIDGE_ROOT / "training" / "semantic_routing" / "prepared"
    evaluation = BRIDGE_ROOT / "evaluation" / "cases"
    train = texts([prepared / "train.jsonl"])
    validation = texts([prepared / "validation.jsonl"])
    holdout = texts(sorted(evaluation.glob("*.jsonl")))
    train_near, train_max = near_overlap(train, holdout)
    validation_near, validation_max = near_overlap(validation, holdout)
    report = {
        "train_holdout_normalized_overlap": len(train & holdout),
        "validation_holdout_normalized_overlap": len(validation & holdout),
        "near_duplicate_metric": "character-trigram Jaccard >= 0.85; audit-only",
        "train_holdout_near_overlap": train_near,
        "validation_holdout_near_overlap": validation_near,
        "train_holdout_max_similarity": round(train_max, 4),
        "validation_holdout_max_similarity": round(validation_max, 4),
        "semantic_leakage_assurance": "provenance/process separation; no embedding-based semantic claim",
        "policy": "report-only; holdout text must not drive data generation, model selection, or thresholds",
    }
    print(json.dumps(report, indent=2))
    return int(bool((train | validation) & holdout))


if __name__ == "__main__":
    raise SystemExit(main())
