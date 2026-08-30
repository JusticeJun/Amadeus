"""Train the final SetFit semantic-routing research candidate offline."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import shutil
import sys

import numpy as np


BRIDGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BRIDGE_ROOT))

from app.routing import CAPABILITY_NAMES  # noqa: E402


DATA_DIR = BRIDGE_ROOT / "training" / "semantic_routing" / "prepared"
DEFAULT_OUTPUT = BRIDGE_ROOT / "research_artifacts" / "semantic-router-v4-setfit"
ENCODER_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
SEED = 1729


def load_rows(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def multilabel_targets(rows: list[dict[str, object]]) -> np.ndarray:
    return np.asarray([
        [int(label in row["capabilities"]) for label in CAPABILITY_NAMES]
        for row in rows
    ])


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


def resolve_cached_encoder(model_id: str) -> str:
    candidate = Path(model_id)
    if candidate.exists():
        return str(candidate)
    from huggingface_hub import scan_cache_dir

    matches = [repo for repo in scan_cache_dir().repos if repo.repo_id == model_id]
    if not matches or not matches[0].revisions:
        raise FileNotFoundError(f"encoder is not available in the local Hugging Face cache: {model_id}")
    revision = max(matches[0].revisions, key=lambda item: item.last_modified)
    return str(revision.snapshot_path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--encoder", default=ENCODER_NAME)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--inference-batch-size", type=int, default=128)
    parser.add_argument("--num-iterations", type=int, default=1)
    parser.add_argument("--body-epochs", type=int, default=1)
    parser.add_argument("--body-learning-rate", type=float, default=2e-5)
    args = parser.parse_args()

    from datasets import Dataset
    from setfit import SetFitModel, Trainer, TrainingArguments
    import torch

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    train_path = DATA_DIR / "train.jsonl"
    validation_path = DATA_DIR / "validation.jsonl"
    train_rows = load_rows(train_path)
    validation_rows = load_rows(validation_path)
    train_targets = multilabel_targets(train_rows)
    validation_targets = multilabel_targets(validation_rows).astype(bool)
    train_dataset = Dataset.from_dict({
        "text": [str(row["text"]) for row in train_rows],
        "label": train_targets.tolist(),
    })
    encoder_path = resolve_cached_encoder(args.encoder)
    model = SetFitModel.from_pretrained(
        encoder_path,
        multi_target_strategy="one-vs-rest",
        labels=list(CAPABILITY_NAMES),
        normalize_embeddings=True,
        local_files_only=True,
    )
    training_args = TrainingArguments(
        output_dir=str(args.output.parent / "setfit-checkpoints"),
        batch_size=(args.batch_size, args.inference_batch_size),
        num_epochs=(args.body_epochs, 1),
        num_iterations=args.num_iterations,
        body_learning_rate=args.body_learning_rate,
        seed=SEED,
        report_to="none",
        save_strategy="no",
    )
    Trainer(model=model, args=training_args, train_dataset=train_dataset).train()
    validation_scores = np.asarray(model.predict_proba(
        [str(row["text"]) for row in validation_rows],
        batch_size=args.inference_batch_size,
        show_progress_bar=True,
    ))
    thresholds = {
        label: select_threshold(validation_scores[:, index], validation_targets[:, index])
        for index, label in enumerate(CAPABILITY_NAMES)
    }
    if args.output.exists():
        shutil.rmtree(args.output)
    model.save_pretrained(args.output)
    setfit_config = args.output / "config_setfit.json"
    setfit_config.write_text(
        json.dumps(json.loads(setfit_config.read_text(encoding="utf-8")), sort_keys=True),
        encoding="utf-8",
    )
    fingerprint = hashlib.sha256(train_path.read_bytes() + b"\0" + validation_path.read_bytes()).hexdigest()
    metadata = {
        "schema_version": 1,
        "model_version": "semantic-router-v4-setfit-research",
        "algorithm": "setfit-multilabel-one-vs-rest-logistic-regression",
        "setfit_version": __import__("setfit").__version__,
        "seed": SEED,
        "dataset_sha256": fingerprint,
        "training_examples": len(train_rows),
        "validation_examples": len(validation_rows),
        "labels": list(CAPABILITY_NAMES),
        "encoder": args.encoder,
        "normalize_embeddings": True,
        "multi_target_strategy": "one-vs-rest",
        "contrastive": {
            "loss": "CosineSimilarityLoss",
            "num_iterations": args.num_iterations,
            "epochs": args.body_epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.body_learning_rate,
        },
        "classification_head": "sklearn OneVsRestClassifier(LogisticRegression)",
        "threshold_strategy": "per-label validation F0.5 for OOS precision",
        "thresholds": thresholds,
    }
    (args.output / "amadeus_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
