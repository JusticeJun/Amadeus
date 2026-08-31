"""Audit exact, normalized, and heuristic near-overlap in the balanced corpus."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

from tools.augment_setfit_multilabel_data import FROZEN_MANUAL_UTTERANCES, normalize


BRIDGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = BRIDGE_ROOT / "research_data" / "semantic-routing-setfit-balanced"
BASE_DATA = BRIDGE_ROOT / "research_data" / "semantic-routing-setfit-multilabel"
SOURCE = "amadeus-reviewed-semantic-routing-v1"


def load_rows(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def trigrams(text: str) -> frozenset[str]:
    value = normalize(text)
    return frozenset(value[index:index + 3] for index in range(max(0, len(value) - 2)))


def near_overlap(
    left: list[dict[str, object]],
    right: list[dict[str, object]],
    *,
    same_collection: bool = False,
    threshold: float = 0.85,
) -> tuple[int, float]:
    matches = 0
    maximum = 0.0
    right_features = [(normalize(str(row["text"])), trigrams(str(row["text"]))) for row in right]
    for left_index, row in enumerate(left):
        value = normalize(str(row["text"]))
        features = trigrams(str(row["text"]))
        for right_index, (other, other_features) in enumerate(right_features):
            if same_collection and right_index <= left_index:
                continue
            if not features or not other_features:
                continue
            if min(len(value), len(other)) / max(len(value), len(other)) < 0.65:
                continue
            score = len(features & other_features) / len(features | other_features)
            maximum = max(maximum, score)
            matches += score >= threshold
    return matches, maximum


def exact_conflicts(rows: list[dict[str, object]]) -> int:
    groups: dict[str, set[tuple[str, ...]]] = defaultdict(set)
    for row in rows:
        groups[normalize(str(row["text"]))].add(tuple(sorted(str(item) for item in row["capabilities"])))
    return sum(len(label_sets) > 1 for label_sets in groups.values())


def audit(data_dir: Path) -> dict[str, object]:
    train_all = load_rows(data_dir / "train.jsonl")
    validation_all = load_rows(data_dir / "validation.jsonl")
    added_train = [row for row in train_all if row.get("source") == SOURCE]
    added_validation = [row for row in validation_all if row.get("source") == SOURCE]
    added = {"train": added_train, "validation": added_validation}
    base_keys = {
        split: {normalize(str(row["text"])) for row in load_rows(BASE_DATA / f"{split}.jsonl")}
        for split in ("train", "validation", "external_test")
    }
    holdout_rows = []
    for path in sorted((BRIDGE_ROOT / "evaluation" / "cases").glob("*.jsonl")):
        holdout_rows.extend(load_rows(path))
    holdout_keys = {normalize(str(row["text"])) for row in holdout_rows}
    manual_keys = {normalize(text) for text in FROZEN_MANUAL_UTTERANCES}
    split_reports: dict[str, object] = {}
    for split, rows in added.items():
        keys = {normalize(str(row["text"])) for row in rows}
        within_near, within_max = near_overlap(rows, rows, same_collection=True)
        holdout_near, holdout_max = near_overlap(rows, holdout_rows)
        split_reports[split] = {
            "rows": len(rows),
            "normalized_unique": len(keys),
            "conflicting_labels": exact_conflicts(rows),
            "base_exact_overlap": {name: len(keys & values) for name, values in base_keys.items()},
            "holdout_normalized_overlap": len(keys & holdout_keys),
            "holdout_near_overlap": holdout_near,
            "holdout_max_similarity": round(holdout_max, 4),
            "manual_normalized_overlap": len(keys & manual_keys),
            "within_split_near_overlap": within_near,
            "within_split_max_similarity": round(within_max, 4),
            "routing_roles": Counter(str(row["semantic"]["routing_role"]) for row in rows),
            "label_sets": Counter("+".join(row["capabilities"]) or "no_match" for row in rows),
        }
    cross_near, cross_max = near_overlap(added_train, added_validation)
    train_keys = {normalize(str(row["text"])) for row in added_train}
    validation_keys = {normalize(str(row["text"])) for row in added_validation}
    return {
        "near_duplicate_metric": "character-trigram Jaccard >= 0.85; audit-only",
        "splits": split_reports,
        "train_validation_normalized_overlap": len(train_keys & validation_keys),
        "train_validation_near_overlap": cross_near,
        "train_validation_max_similarity": round(cross_max, 4),
        "semantic_leakage_assurance": "provenance/process separation; no embedding-based semantic claim",
    }


def has_blocking_failure(report: dict[str, object]) -> bool:
    if report["train_validation_normalized_overlap"] or report["train_validation_near_overlap"]:
        return True
    for split in report["splits"].values():
        if split["rows"] != split["normalized_unique"] or split["conflicting_labels"]:
            return True
        if split["holdout_normalized_overlap"] or split["holdout_near_overlap"]:
            return True
        if split["manual_normalized_overlap"] or split["within_split_near_overlap"]:
            return True
        if any(split["base_exact_overlap"].values()):
            return True
    return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = audit(args.data_dir)
    payload = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(payload, encoding="utf-8")
    print(payload)
    return int(has_blocking_failure(report))


if __name__ == "__main__":
    raise SystemExit(main())
