"""Train the deterministic local semantic routing artifact offline."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random
import runpy
import unicodedata


BRIDGE_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_SCHEMA = 1


CAPABILITIES = tuple(runpy.run_path(str(BRIDGE_ROOT / "app" / "routing" / "capabilities.py"))["CAPABILITIES"])
CAPABILITY_NAMES = tuple(item.name for item in CAPABILITIES)


def character_ngrams(text: str, minimum: int, maximum: int) -> Counter[str]:
    normalized = " " + " ".join(unicodedata.normalize("NFKC", text).casefold().split()) + " "
    return Counter(
        normalized[start:start + size]
        for size in range(minimum, maximum + 1)
        for start in range(max(0, len(normalized) - size + 1))
    )


DATA_DIR = BRIDGE_ROOT / "training" / "semantic_routing"
DEFAULT_OUTPUT = BRIDGE_ROOT / "app" / "routing" / "artifacts" / "semantic-router-v1.json"
SEED = 1729


def load_examples(path: Path) -> list[tuple[str, frozenset[str]]]:
    examples = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        item = json.loads(line)
        labels = frozenset(item.get("capabilities", item.get("labels", [])))
        unknown = labels - set(CAPABILITY_NAMES)
        if unknown:
            raise ValueError(f"unknown labels at {path}:{number}: {sorted(unknown)}")
        examples.append((str(item["text"]), labels))
    return examples


def vectorize(text: str, idf: dict[str, float]) -> dict[str, float]:
    counts = character_ngrams(text, 2, 5)
    values = {key: (1 + math.log(count)) * idf[key] for key, count in counts.items() if key in idf}
    norm = math.sqrt(sum(value * value for value in values.values())) or 1.0
    return {key: value / norm for key, value in values.items()}


def probability(weights: dict[str, float], bias: float, values: dict[str, float]) -> float:
    score = bias + sum(value * weights.get(key, 0.0) for key, value in values.items())
    return 1 / (1 + math.exp(-max(-30.0, min(30.0, score))))


def train_label(vectors, targets, *, epochs: int = 35, rate: float = 0.35, l2: float = 0.0005):
    weights: dict[str, float] = {}
    positives = sum(targets)
    bias = math.log((positives + 1) / (len(targets) - positives + 1))
    order = list(range(len(vectors)))
    rng = random.Random(SEED)
    for epoch in range(epochs):
        rng.shuffle(order)
        step = rate / math.sqrt(1 + epoch * 0.08)
        for index in order:
            values, target = vectors[index], targets[index]
            error = target - probability(weights, bias, values)
            bias += step * error
            for key, value in values.items():
                updated = weights.get(key, 0.0) * (1 - step * l2) + step * error * value
                if abs(updated) > 1e-8:
                    weights[key] = updated
    return weights, bias


def select_threshold(probabilities: list[float], targets: list[int], side_effecting: bool) -> float:
    candidates = []
    beta2 = 0.25 if side_effecting else 1.0
    for threshold in (value / 100 for value in range(20, 91)):
        tp = fp = fn = 0
        for score, target in zip(probabilities, targets):
            predicted = score >= threshold
            tp += int(predicted and target)
            fp += int(predicted and not target)
            fn += int(not predicted and target)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        metric = ((1 + beta2) * precision * recall / (beta2 * precision + recall)) if precision + recall else 0.0
        candidates.append((fp == 0, metric, recall, threshold))
    zero_false_positive = [candidate for candidate in candidates if candidate[0]]
    pool = zero_false_positive or candidates
    return max(pool, key=lambda item: (item[1], item[2], item[3]))[3]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    train_path = DATA_DIR / "prepared" / "train.jsonl"
    validation_path = DATA_DIR / "prepared" / "validation.jsonl"
    if not train_path.exists() or not validation_path.exists():
        raise FileNotFoundError("run tools/prepare_semantic_dataset.py before training")
    train, validation = load_examples(train_path), load_examples(validation_path)
    train_rows = [json.loads(line) for line in train_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    source_counts = Counter(str(row.get("source", "unknown")) for row in train_rows)
    document_frequency: Counter[str] = Counter()
    for text, _ in train:
        document_frequency.update(character_ngrams(text, 2, 5).keys())
    idf = {key: math.log((1 + len(train)) / (1 + count)) + 1 for key, count in document_frequency.items()}
    train_vectors = [vectorize(text, idf) for text, _ in train]
    validation_vectors = [vectorize(text, idf) for text, _ in validation]
    models = {}
    for definition in CAPABILITIES:
        train_targets = [int(definition.name in labels) for _, labels in train]
        validation_targets = [int(definition.name in labels) for _, labels in validation]
        weights, bias = train_label(train_vectors, train_targets)
        weights = dict(sorted(weights.items(), key=lambda item: abs(item[1]), reverse=True)[:12000])
        scores = [probability(weights, bias, values) for values in validation_vectors]
        models[definition.name] = {
            "bias": bias,
            "threshold": select_threshold(scores, validation_targets, definition.side_effecting),
            "weights": dict(sorted(weights.items())),
        }
    fingerprint = hashlib.sha256(train_path.read_bytes() + b"\0" + validation_path.read_bytes()).hexdigest()
    retained_features = {feature for model in models.values() for feature in model["weights"]}
    artifact = {
        "schema_version": ARTIFACT_SCHEMA,
        "model_version": "semantic-router-v1",
        "algorithm": "character-ngram-tfidf-ovr-logistic-regression",
        "seed": SEED,
        "dataset_sha256": fingerprint,
        "training_examples": len(train),
        "validation_examples": len(validation),
        "training_sources": dict(sorted(source_counts.items())),
        "threshold_strategy": "validation-zero-false-positive-first; side-effects-use-F0.5",
        "labels": list(CAPABILITY_NAMES),
        "preprocessing": {"normalization": "NFKC-casefold-whitespace", "character_ngrams": {"minimum": 2, "maximum": 5}},
        "idf": {key: idf[key] for key in sorted(retained_features)},
        "models": models,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, ensure_ascii=False, sort_keys=True, separators=(",", ":")), encoding="utf-8")
    print(f"wrote {args.output} ({args.output.stat().st_size} bytes)")
    print("thresholds " + " ".join(f"{label}={models[label]['threshold']:.2f}" for label in CAPABILITY_NAMES))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
