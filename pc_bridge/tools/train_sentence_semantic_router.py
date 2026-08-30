"""Train a lightweight OvR classifier over frozen multilingual sentence embeddings."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression


BRIDGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BRIDGE_ROOT))

from app.routing import CAPABILITY_NAMES  # noqa: E402


DATA_DIR = BRIDGE_ROOT / "training" / "semantic_routing" / "prepared"
DEFAULT_OUTPUT = (
    BRIDGE_ROOT / "app" / "routing" / "artifacts"
    / "semantic-router-v3-multilingual-minilm-research.json"
)
ENCODER_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
SEED = 1729


def load_rows(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def select_threshold(scores: np.ndarray, targets: np.ndarray) -> float:
    candidates = []
    for threshold in np.arange(0.05, 0.951, 0.01):
        predicted = scores >= threshold
        true_positive = int(np.sum(predicted & targets))
        false_positive = int(np.sum(predicted & ~targets))
        false_negative = int(np.sum(~predicted & targets))
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
        beta_squared = 0.25
        metric = (
            (1 + beta_squared) * precision * recall / (beta_squared * precision + recall)
            if precision + recall else 0.0
        )
        candidates.append((metric, precision, recall, float(threshold)))
    return max(candidates)[3]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--encoder", default=ENCODER_NAME)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    train_path = DATA_DIR / "train.jsonl"
    validation_path = DATA_DIR / "validation.jsonl"
    train_rows, validation_rows = load_rows(train_path), load_rows(validation_path)
    encoder = SentenceTransformer(args.encoder, local_files_only=True)
    train_vectors = encoder.encode(
        [str(row["text"]) for row in train_rows],
        batch_size=args.batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    validation_vectors = encoder.encode(
        [str(row["text"]) for row in validation_rows],
        batch_size=args.batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    models = {}
    for label in CAPABILITY_NAMES:
        train_targets = np.asarray([label in row["capabilities"] for row in train_rows], dtype=bool)
        validation_targets = np.asarray(
            [label in row["capabilities"] for row in validation_rows], dtype=bool,
        )
        classifier = LogisticRegression(
            C=1.0,
            max_iter=1000,
            random_state=SEED,
            solver="liblinear",
        ).fit(train_vectors, train_targets)
        validation_scores = classifier.predict_proba(validation_vectors)[:, 1]
        models[label] = {
            "intercept": float(classifier.intercept_[0]),
            "threshold": select_threshold(validation_scores, validation_targets),
            "coefficients": [float(value) for value in classifier.coef_[0]],
        }
    fingerprint = hashlib.sha256(train_path.read_bytes() + b"\0" + validation_path.read_bytes()).hexdigest()
    artifact = {
        "schema_version": 1,
        "model_version": "semantic-router-v3-multilingual-minilm-research",
        "algorithm": "frozen-sentence-transformer-ovr-logistic-regression",
        "seed": SEED,
        "dataset_sha256": fingerprint,
        "training_examples": len(train_rows),
        "validation_examples": len(validation_rows),
        "labels": list(CAPABILITY_NAMES),
        "encoder": {
            "name": args.encoder,
            "dimension": int(train_vectors.shape[1]),
            "normalize_embeddings": True,
        },
        "threshold_strategy": "per-label validation F0.5 for OOS precision",
        "models": models,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(artifact, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    print(f"wrote {args.output} ({args.output.stat().st_size} bytes)")
    print("thresholds " + " ".join(
        f"{label}={models[label]['threshold']:.2f}" for label in CAPABILITY_NAMES
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
