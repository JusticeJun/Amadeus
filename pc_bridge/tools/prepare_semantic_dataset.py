"""Build provenance-aware train/validation data without reading Amadeus holdout text."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from difflib import SequenceMatcher
import hashlib
import json
from pathlib import Path
import random
import tarfile
import tempfile
import unicodedata
from urllib.request import urlopen
import zipfile


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


def fetch_file(source: dict[str, object], cache_dir: Path, filename: str) -> Path:
    output = cache_dir / filename
    if not output.exists():
        with urlopen(str(source["url"]), timeout=120) as response:
            output.write_bytes(response.read())
    if sha256(output.read_bytes()) != source["sha256"]:
        raise ValueError(f"source checksum mismatch: {source['name']}")
    return output


def fetch_zip_member(source: dict[str, object], cache_dir: Path, filename: str) -> Path:
    archive = cache_dir / filename
    if not archive.exists():
        with urlopen(str(source["url"]), timeout=120) as response:
            archive.write_bytes(response.read())
    if sha256(archive.read_bytes()) != source["archive_sha256"]:
        raise ValueError(f"archive checksum mismatch: {source['name']}")
    with zipfile.ZipFile(archive) as bundle:
        content = bundle.read(str(source["member"]))
    if sha256(content) != source["member_sha256"]:
        raise ValueError(f"member checksum mismatch: {source['name']}")
    output = cache_dir / (filename + ".json")
    output.write_bytes(content)
    return output


def massive_rows(path: Path, source: dict[str, object]) -> dict[str, list[dict[str, object]]]:
    mapping = source["intent_mapping"]
    by_split_intent: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for item in read_jsonl(path):
        partition = str(item["partition"])
        if partition not in {"train", "dev", "test"}:
            continue
        by_split_intent[(partition, str(item["intent"]))].append(item)
    rng = random.Random(SEED)
    result = {"train": [], "validation": [], "external_test": []}
    limits = {"train": 80, "dev": 24, "test": 24}
    for (partition, intent), items in sorted(by_split_intent.items()):
        rng.shuffle(items)
        capabilities = list(mapping.get(intent, []))
        selected = items if capabilities else items[:limits[partition]]
        target_split = {"train": "train", "dev": "validation", "test": "external_test"}[partition]
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


def hwu_rows(path: Path, source: dict[str, object]) -> dict[str, list[dict[str, object]]]:
    mapping = source["intent_mapping"]
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for item in csv.DictReader(handle, delimiter=";"):
            intent = f"{item['scenario']}/{item['intent']}"
            bucket = int(hashlib.sha256(item["userid"].encode()).hexdigest(), 16) % 10
            split = "external_test" if bucket == 0 else ("validation" if bucket == 1 else "train")
            grouped[(split, intent)].append(item)
    rng = random.Random(SEED + 1)
    result = {"train": [], "validation": [], "external_test": []}
    for (split, intent), items in sorted(grouped.items()):
        rng.shuffle(items)
        capabilities = list(mapping.get(intent, []))
        limit = 220 if split == "train" else 55
        if not capabilities:
            limit = 18 if split == "train" else 5
        for item in items[:limit]:
            text = item["answer"].strip() or item["answer_normalised"].strip()
            if not text:
                continue
            result[split].append({
                "id": f"hwu64-2019-en-{item['answerid']}",
                "text": text,
                "capabilities": capabilities,
                "source": "hwu64-2019-en",
                "source_split": split,
                "source_intent": intent,
                "adaptation": "intent-map" if capabilities else "balanced-no-match",
                "tags": ["external", "english", "assistant_utterance"],
            })
    return result


def clinc_rows(path: Path, source: dict[str, object]) -> dict[str, list[dict[str, object]]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    mapping = source["intent_mapping"]
    hard_negative = set(source["hard_negative_intents"])
    result = {"train": [], "validation": [], "external_test": []}
    source_splits = (
        ("train", "train"), ("oos_train", "train"),
        ("val", "validation"), ("oos_val", "validation"),
        ("test", "external_test"), ("oos_test", "external_test"),
    )
    for source_split, target_split in source_splits:
        grouped: dict[str, list[tuple[str, str]]] = defaultdict(list)
        for text, intent in data[source_split]:
            grouped[intent].append((text, intent))
        for intent, items in sorted(grouped.items()):
            capabilities = list(mapping.get(intent, []))
            if capabilities or intent == "oos":
                selected = items
            else:
                selected = items[:18 if target_split == "train" else (5 if target_split == "validation" else 8)]
            for index, (text, _) in enumerate(selected):
                tags = ["external", "english", "oos"] if intent == "oos" else ["external", "english"]
                if intent in hard_negative:
                    tags.append("hard_negative")
                result[target_split].append({
                    "id": f"clinc150-{source_split}-{intent}-{index:04d}",
                    "text": text,
                    "capabilities": capabilities,
                    "source": "clinc150-uci-full",
                    "source_split": source_split,
                    "source_intent": intent,
                    "adaptation": "intent-map" if capabilities else ("oos-to-no-match" if intent == "oos" else "balanced-no-match"),
                    "tags": tags,
                })
    return result


def amadeus_boundary_rows(split: str) -> list[dict[str, object]]:
    subjects = {
        "weather": ("날씨", "비", "기온", "우산", "습도"),
        "music_control": ("노래", "음악", "재생", "플레이리스트", "앨범"),
        "pc_control": ("크롬", "메모장", "컴퓨터 소리", "음소거", "브라우저"),
    }
    negative_forms = (
        ("{term}라는 말의 뜻을 설명해줘", "{term}의 개념이 궁금해"),
        ("{term}에 관한 짧은 이야기를 써줘", "{term}를 소재로 글을 적어줘"),
        ("{term} 기능은 보통 어떻게 동작해?", "{term}의 작동 원리를 알려줘"),
        ("어제 {term} 얘기를 들었어", "친구와 {term} 이야기를 했어"),
    )
    rows = []
    for capability, terms in subjects.items():
        for term_index, term in enumerate(terms):
            for form_index, forms in enumerate(negative_forms):
                if split == "validation" and (term_index + form_index) % 3:
                    continue
                form = forms[1 if split == "validation" else 0]
                rows.append({
                    "id": f"amadeus-boundary-{split}-{capability}-{term_index}-{form_index}",
                    "text": form.format(term=term),
                    "capabilities": [],
                    "source": "amadeus-template-boundary-v2",
                    "source_split": split,
                    "source_intent": capability,
                    "adaptation": "reviewed-template-generation",
                    "generation": {"template": form, "term": term},
                    "tags": ["generated", "hard_negative", f"confuses_{capability}"],
                })
    pair_terms = (("weather", "music_control"), ("weather", "pc_control"), ("music_control", "pc_control"))
    pair_forms = (
        ("{left}와 {right}의 공통점을 설명해줘", "{left}하고 {right}는 어떻게 달라?"),
        ("{left} 때문에 {right}가 달라지는 이유가 뭐야?", "{left}와 {right} 사이에 관계가 있어?"),
        ("{left}와 {right}를 소재로 문장을 만들어줘", "{left}, {right} 두 단어로 글을 써줘"),
    )
    for pair_index, (left, right) in enumerate(pair_terms):
        for form_index, forms in enumerate(pair_forms):
            left_term = subjects[left][(pair_index + form_index) % len(subjects[left])]
            right_term = subjects[right][(pair_index * 2 + form_index) % len(subjects[right])]
            form = forms[1 if split == "validation" else 0]
            rows.append({
                "id": f"amadeus-boundary-{split}-pair-{left}-{right}-{form_index}",
                "text": form.format(left=left_term, right=right_term),
                "capabilities": [],
                "source": "amadeus-template-boundary-v2",
                "source_split": split,
                "source_intent": f"{left}+{right}",
                "adaptation": "reviewed-template-generation",
                "generation": {"template": form, "terms": [left_term, right_term]},
                "tags": ["generated", "hard_negative", "capability_pair_confusion", f"pair_{left}_{right}"],
            })
    return rows


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


def remove_near_duplicates(rows: list[dict[str, object]]) -> tuple[list[dict[str, object]], int]:
    grouped: dict[tuple[str, str, tuple[str, ...]], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["source"]), str(row["source_intent"]), tuple(row["capabilities"]))].append(row)
    kept = []
    removed = 0
    for group in grouped.values():
        ordered = sorted(group, key=lambda row: normalized(str(row["text"])))
        previous = ""
        for row in ordered:
            current = normalized(str(row["text"]))
            if previous and min(len(previous), len(current)) >= 12 and SequenceMatcher(None, previous, current).ratio() >= 0.94:
                removed += 1
                continue
            kept.append(row)
            previous = current
    return kept, removed


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
    sources = manifest["sources"]
    cache = args.cache_dir or Path(tempfile.gettempdir()) / "amadeus-semantic-data"
    cache.mkdir(parents=True, exist_ok=True)
    massive = massive_rows(fetch_member(sources["massive-1.1-ko-KR"], cache), sources["massive-1.1-ko-KR"])
    hwu_path = fetch_file(sources["hwu64-2019-en"], cache, "hwu64-2019.csv")
    hwu = hwu_rows(hwu_path, sources["hwu64-2019-en"])
    clinc_path = fetch_zip_member(sources["clinc150-uci-full"], cache, "clinc150-uci.zip")
    clinc = clinc_rows(clinc_path, sources["clinc150-uci-full"])
    splits = {
        "train": massive["train"] + hwu["train"] + clinc["train"]
        + composed_multilabel(massive["train"], "train", 180) + amadeus_boundary_rows("train"),
        "validation": massive["validation"] + hwu["validation"] + clinc["validation"]
        + composed_multilabel(massive["validation"], "dev", 45) + amadeus_boundary_rows("validation"),
        "external_test": massive["external_test"] + hwu["external_test"] + clinc["external_test"],
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
    quality: dict[str, tuple[int, int, int]] = {}
    for split, rows in splits.items():
        rows, removed, conflicts = deduplicate(rows)
        rows, near_removed = remove_near_duplicates(rows)
        cleaned[split] = rows
        quality[split] = (removed, conflicts, near_removed)
    train_keys = {normalized(str(row["text"])) for row in cleaned["train"]}
    overlap = [row for row in cleaned["validation"] if normalized(str(row["text"])) in train_keys]
    cleaned["validation"] = [row for row in cleaned["validation"] if normalized(str(row["text"])) not in train_keys]
    fit_keys = train_keys | {normalized(str(row["text"])) for row in cleaned["validation"]}
    external_overlap = [row for row in cleaned["external_test"] if normalized(str(row["text"])) in fit_keys]
    cleaned["external_test"] = [row for row in cleaned["external_test"] if normalized(str(row["text"])) not in fit_keys]
    holdout_keys = {
        normalized(str(item["text"]))
        for path in sorted(args.holdout_dir.glob("*.jsonl"))
        for item in read_jsonl(path)
    }
    holdout_overlap = {
        split: [row for row in rows if normalized(str(row["text"])) in holdout_keys]
        for split, rows in cleaned.items() if split != "external_test"
    }
    for split in ("train", "validation"):
        cleaned[split] = [row for row in cleaned[split] if normalized(str(row["text"])) not in holdout_keys]
    for split, rows in cleaned.items():
        removed, conflicts, near_removed = quality[split]
        write_jsonl(args.output_dir / f"{split}.jsonl", rows)
        report["splits"][split] = {
            "rows": len(rows),
            "deduplicated": removed,
            "label_conflicts_excluded": conflicts,
            "near_duplicates_excluded": near_removed,
            "sources": Counter(str(row["source"]) for row in rows),
            "capabilities": Counter(capability for row in rows for capability in row["capabilities"]),
            "no_match": sum(not row["capabilities"] for row in rows),
            "multi_label": sum(len(row["capabilities"]) > 1 for row in rows),
            "hard_negative": sum("hard_negative" in row.get("tags", []) for row in rows),
            "capability_pair_confusion": Counter(
                tag.removeprefix("pair_")
                for row in rows for tag in row.get("tags", []) if tag.startswith("pair_")
            ),
            "multilabel_pairs": Counter(
                "+".join(sorted(row["capabilities"]))
                for row in rows if len(row["capabilities"]) > 1
            ),
        }
    report["train_validation_overlap_excluded"] = len(overlap)
    report["train_validation_overlap"] = 0
    report["fit_external_test_overlap_excluded"] = len(external_overlap)
    report["holdout_exact_overlap_excluded"] = {
        split: len(rows) for split, rows in holdout_overlap.items()
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=dict), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, default=dict))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
