"""Create the controlled multi-label-only SetFit research training variant."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from difflib import SequenceMatcher
import json
from pathlib import Path
import random
import shutil
import unicodedata


BRIDGE_ROOT = Path(__file__).resolve().parents[1]
BASE_DATA = BRIDGE_ROOT / "training" / "semantic_routing" / "prepared"
DEFAULT_OUTPUT = BRIDGE_ROOT / "research_data" / "semantic-routing-setfit-multilabel"
SEED = 1729
ADDED_PER_PAIR = 240
LABEL_PAIRS = (
    ("weather", "music_control"),
    ("weather", "pc_control"),
    ("music_control", "pc_control"),
)
COMPOSITION_TYPES = (
    "explicit_explicit",
    "implicit_explicit",
    "explicit_implicit",
    "implicit_implicit",
)
CONNECTORS = (
    " 그리고 ",
    ". 그다음에는 ",
    " 먼저 해주고, 이어서 ",
    "도 부탁해. 또 ",
    ", 그리고 ",
    " 한 다음에 ",
    ". 가능하면 이어서 ",
    "부터 해주고 그다음 ",
)
FROZEN_MANUAL_UTTERANCES = (
    "오늘 밖에서 공부해도 괜찮을까?",
    "오늘 반팔만 입어도 괜찮을까?",
    "나가려는데 겉옷 챙기는 게 좋을까?",
    "오늘 반팔만 입어도 될지 알려주고 신나는 노래도 하나 틀어줘",
    "오늘 반팔 입어도 될지 알려주고 컴퓨터 소리도 좀 줄여줘",
    "신나는 노래 하나 틀어주고 컴퓨터 소리도 조금 줄여줘",
)
IMPLICIT_INTENTS = {
    "weather": {
        "weather_dependent_advice",
        "weather_dependent_clothing",
        "weather_dependent_drying",
        "weather_dependent_rain",
        "weather_dependent_temperature",
    },
    "music_control": {"music_query", "music/query", "what_song"},
    "pc_control": {
        "audio_volume_down",
        "audio/volume_down",
        "audio_volume_other",
        "audio/volume_other",
    },
}


def normalize(text: str) -> str:
    return "".join(unicodedata.normalize("NFKC", text).casefold().split())


def load_rows(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def parent_pools(rows: list[dict[str, object]]) -> dict[str, dict[str, list[dict[str, object]]]]:
    pools = defaultdict(lambda: {"explicit": [], "implicit": []})
    for row in rows:
        capabilities = list(row["capabilities"])
        is_korean = "korean" in row.get("tags", []) or str(row.get("source", "")).startswith("amadeus-")
        if len(capabilities) != 1 or not is_korean:
            continue
        label = str(capabilities[0])
        kind = "implicit" if str(row.get("source_intent", "")) in IMPLICIT_INTENTS[label] else "explicit"
        pools[label][kind].append(row)
    return pools


def is_too_close_to_manual(text: str) -> bool:
    candidate = normalize(text)
    return any(
        SequenceMatcher(None, candidate, normalize(manual)).ratio() >= 0.65
        for manual in FROZEN_MANUAL_UTTERANCES
    )


def augment(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    pools = parent_pools(rows)
    rng = random.Random(SEED)
    for label in pools:
        for kind in pools[label]:
            rng.shuffle(pools[label][kind])
            if not pools[label][kind]:
                raise ValueError(f"missing {kind} Korean parent pool for {label}")
    existing = {normalize(str(row["text"])) for row in rows}
    result = []
    per_type = ADDED_PER_PAIR // len(COMPOSITION_TYPES)
    for pair_index, (left_label, right_label) in enumerate(LABEL_PAIRS):
        pair_added = 0
        for type_index, composition_type in enumerate(COMPOSITION_TYPES):
            left_kind, right_kind = composition_type.split("_")
            left_pool = pools[left_label][left_kind]
            right_pool = pools[right_label][right_kind]
            made = attempts = 0
            while made < per_type:
                attempts += 1
                if attempts > per_type * 100:
                    raise RuntimeError(f"unable to compose enough unique rows for {pair_index}:{composition_type}")
                left = left_pool[(attempts * 7 + pair_index * 11) % len(left_pool)]
                right = right_pool[(attempts * 13 + type_index * 17) % len(right_pool)]
                if (attempts + pair_index + type_index) % 2:
                    left, right = right, left
                connector = CONNECTORS[(attempts + pair_index * 3 + type_index) % len(CONNECTORS)]
                text = str(left["text"]).rstrip(" .?!") + connector + str(right["text"]).lstrip()
                key = normalize(text)
                if key in existing or is_too_close_to_manual(text):
                    continue
                existing.add(key)
                pair_added += 1
                made += 1
                result.append({
                    "id": f"setfit-multilabel-aug-{pair_index + 1}-{pair_added:04d}",
                    "text": text,
                    "capabilities": sorted((left_label, right_label)),
                    "source": "amadeus-setfit-multilabel-augmentation-v1",
                    "source_split": "train",
                    "source_intent": f"{left_label}+{right_label}",
                    "adaptation": "deterministic-diverse-clause-composition",
                    "parent_ids": [left["id"], right["id"]],
                    "generation": {
                        "composition_type": composition_type,
                        "connector": connector,
                        "order_swapped": left["capabilities"][0] != left_label,
                    },
                    "tags": ["generated", "korean", "multi_label", "setfit_ablation"],
                })
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    train = load_rows(BASE_DATA / "train.jsonl")
    added = augment(train)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_rows(args.output_dir / "train.jsonl", train + added)
    shutil.copyfile(BASE_DATA / "validation.jsonl", args.output_dir / "validation.jsonl")
    shutil.copyfile(BASE_DATA / "external_test.jsonl", args.output_dir / "external_test.jsonl")
    report = {
        "seed": SEED,
        "baseline_train_rows": len(train),
        "baseline_multilabel_rows": sum(len(row["capabilities"]) > 1 for row in train),
        "added_rows": len(added),
        "augmented_train_rows": len(train) + len(added),
        "augmented_multilabel_rows": sum(len(row["capabilities"]) > 1 for row in train) + len(added),
        "added_pairs": Counter("+".join(row["capabilities"]) for row in added),
        "added_composition_types": Counter(row["generation"]["composition_type"] for row in added),
        "source": "amadeus-setfit-multilabel-augmentation-v1",
        "manual_regression_overlap": sum(
            normalize(str(row["text"])) in {normalize(text) for text in FROZEN_MANUAL_UTTERANCES}
            for row in added
        ),
        "validation_unchanged": (BASE_DATA / "validation.jsonl").read_bytes()
        == (args.output_dir / "validation.jsonl").read_bytes(),
        "external_test_unchanged": (BASE_DATA / "external_test.jsonl").read_bytes()
        == (args.output_dir / "external_test.jsonl").read_bytes(),
    }
    (args.output_dir / "augmentation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
