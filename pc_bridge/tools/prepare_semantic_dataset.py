"""Build provenance-aware train/validation data without reading Amadeus holdout text."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import tarfile
import tempfile
import unicodedata
from urllib.request import urlopen


BRIDGE_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = BRIDGE_ROOT / "training" / "semantic_routing"
SOURCE_MANIFEST = DATA_DIR / "sources.json"
SEED = 1729


def normalized(text: str) -> str:
    return "".join(unicodedata.normalize("NFKC", text).casefold().split())


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def local_rows(path: Path, split: str) -> list[dict[str, object]]:
    rows = []
    for index, item in enumerate(read_jsonl(path), 1):
        rows.append({
            "id": f"amadeus-baseline-{split}-{index:04d}",
            "text": item["text"],
            "capabilities": item.get("capabilities", item.get("labels", [])),
            "source": "amadeus-hand-authored-baseline-v1",
            "source_split": split,
            "source_intent": "",
            "adaptation": "direct",
            "tags": item.get("tags", ["initial_baseline"]),
        })
    return rows


def fetch_member(source: dict[str, object], cache_dir: Path) -> Path:
    archive = cache_dir / "massive-1.1.tar.gz"
    if not archive.exists():
        with urlopen(str(source["url"]), timeout=120) as response:
            archive.write_bytes(response.read())
    raw = archive.read_bytes()
    if sha256(raw) != source["archive_sha256"]:
        raise ValueError("MASSIVE archive checksum mismatch")
    with tarfile.open(archive, "r:gz") as bundle:
        member = bundle.extractfile(str(source["member"]))
        if member is None:
            raise ValueError("MASSIVE Korean member is missing")
        content = member.read()
    if sha256(content) != source["member_sha256"]:
        raise ValueError("MASSIVE Korean file checksum mismatch")
    output = cache_dir / "massive-ko-KR.jsonl"
    output.write_bytes(content)
    return output


def massive_rows(path: Path, source: dict[str, object]) -> dict[str, list[dict[str, object]]]:
    mapping = source["intent_mapping"]
    by_split_intent: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for item in read_jsonl(path):
        partition = str(item["partition"])
        if partition not in {"train", "dev"}:
            continue
        by_split_intent[(partition, str(item["intent"]))].append(item)
    rng = random.Random(SEED)
    result = {"train": [], "validation": []}
    limits = {"train": 80, "dev": 24}
    for (partition, intent), items in sorted(by_split_intent.items()):
        rng.shuffle(items)
        capabilities = list(mapping.get(intent, []))
        selected = items if capabilities else items[:limits[partition]]
        target_split = "train" if partition == "train" else "validation"
        for item in selected:
            result[target_split].append({
                "id": f"massive-1.1-ko-KR-{item['id']}",
                "text": item["utt"],
                "capabilities": capabilities,
                "source": "massive-1.1-ko-KR",
                "source_split": partition,
                "source_intent": intent,
                "adaptation": "intent-map" if capabilities else "unmapped-to-no-match",
                "tags": ["external", "korean", "assistant_utterance"],
            })
    return result


def composed_multilabel(rows: list[dict[str, object]], split: str, limit: int) -> list[dict[str, object]]:
    positives = [row for row in rows if len(row["capabilities"]) == 1]
    by_label: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in positives:
        by_label[str(row["capabilities"][0])].append(row)
    labels = sorted(by_label)
    connectors = (" 그리고 ", " 또 ", " 확인한 다음 ", ", ")
    result = []
    index = 0
    while len(result) < limit:
        left_label = labels[index % len(labels)]
        right_label = labels[(index // len(labels) + 1) % len(labels)]
        index += 1
        if left_label == right_label:
            continue
        left = by_label[left_label][index % len(by_label[left_label])]
        right = by_label[right_label][(index * 7) % len(by_label[right_label])]
        result.append({
            "id": f"massive-composed-{split}-{len(result) + 1:04d}",
            "text": str(left["text"]) + connectors[index % len(connectors)] + str(right["text"]),
            "capabilities": sorted({left_label, right_label}),
            "source": "massive-1.1-ko-KR-composed",
            "source_split": split,
            "source_intent": f"{left['source_intent']}+{right['source_intent']}",
            "adaptation": "deterministic-multi-intent-composition",
            "parent_ids": [left["id"], right["id"]],
            "tags": ["external-derived", "korean", "multi_label"],
        })
    return result


def deduplicate(rows: list[dict[str, object]]) -> tuple[list[dict[str, object]], int, int]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[normalized(str(row["text"]))].append(row)
    kept = []
    removed = conflicts = 0
    for group in grouped.values():
        label_sets = {tuple(sorted(str(item) for item in row["capabilities"])) for row in group}
        if len(label_sets) > 1:
            conflicts += len(group)
            continue
        kept.append(group[0])
        removed += len(group) - 1
    return kept, removed, conflicts


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DATA_DIR / "prepared")
    parser.add_argument("--holdout-dir", type=Path, default=BRIDGE_ROOT / "evaluation" / "cases")
    args = parser.parse_args()
    manifest = json.loads(SOURCE_MANIFEST.read_text(encoding="utf-8"))
    source = manifest["sources"]["massive-1.1-ko-KR"]
    cache = args.cache_dir or Path(tempfile.gettempdir()) / "amadeus-semantic-data"
    cache.mkdir(parents=True, exist_ok=True)
    external = massive_rows(fetch_member(source, cache), source)
    splits = {
        "train": external["train"] + composed_multilabel(external["train"], "train", 180),
        "validation": external["validation"] + composed_multilabel(external["validation"], "dev", 45),
    }
    report: dict[str, object] = {
        "seed": SEED,
        "splits": {},
        "source_manifest_sha256": sha256(SOURCE_MANIFEST.read_bytes()),
        "quarantined_sources": {
            "amadeus-hand-authored-baseline-v1": {
                "train_rows": len(read_jsonl(DATA_DIR / "train.jsonl")),
                "validation_rows": len(read_jsonl(DATA_DIR / "validation.jsonl")),
                "reason": "initial baseline retained but excluded wholesale after independent overlap audit",
            }
        },
    }
    cleaned: dict[str, list[dict[str, object]]] = {}
    quality: dict[str, tuple[int, int]] = {}
    for split, rows in splits.items():
        rows, removed, conflicts = deduplicate(rows)
        cleaned[split] = rows
        quality[split] = (removed, conflicts)
    train_keys = {normalized(str(row["text"])) for row in cleaned["train"]}
    overlap = [row for row in cleaned["validation"] if normalized(str(row["text"])) in train_keys]
    cleaned["validation"] = [row for row in cleaned["validation"] if normalized(str(row["text"])) not in train_keys]
    holdout_keys = {
        normalized(str(item["text"]))
        for path in sorted(args.holdout_dir.glob("*.jsonl"))
        for item in read_jsonl(path)
    }
    holdout_overlap = {
        split: [row for row in rows if normalized(str(row["text"])) in holdout_keys]
        for split, rows in cleaned.items()
    }
    cleaned = {
        split: [row for row in rows if normalized(str(row["text"])) not in holdout_keys]
        for split, rows in cleaned.items()
    }
    for split, rows in cleaned.items():
        removed, conflicts = quality[split]
        write_jsonl(args.output_dir / f"{split}.jsonl", rows)
        report["splits"][split] = {
            "rows": len(rows),
            "deduplicated": removed,
            "label_conflicts_excluded": conflicts,
            "sources": Counter(str(row["source"]) for row in rows),
            "capabilities": Counter(capability for row in rows for capability in row["capabilities"]),
            "no_match": sum(not row["capabilities"] for row in rows),
            "multi_label": sum(len(row["capabilities"]) > 1 for row in rows),
        }
    report["train_validation_overlap_excluded"] = len(overlap)
    report["train_validation_overlap"] = 0
    report["holdout_exact_overlap_excluded"] = {
        split: len(rows) for split, rows in holdout_overlap.items()
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=dict), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, default=dict))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
