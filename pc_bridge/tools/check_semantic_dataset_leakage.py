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


def main() -> int:
    prepared = BRIDGE_ROOT / "training" / "semantic_routing" / "prepared"
    evaluation = BRIDGE_ROOT / "evaluation" / "cases"
    train = texts([prepared / "train.jsonl"])
    validation = texts([prepared / "validation.jsonl"])
    holdout = texts(sorted(evaluation.glob("*.jsonl")))
    report = {
        "train_holdout_normalized_overlap": len(train & holdout),
        "validation_holdout_normalized_overlap": len(validation & holdout),
        "policy": "report-only; holdout text must not drive data generation, model selection, or thresholds",
    }
    print(json.dumps(report, indent=2))
    return int(bool((train | validation) & holdout))


if __name__ == "__main__":
    raise SystemExit(main())
