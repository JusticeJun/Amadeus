from __future__ import annotations

import json
from pathlib import Path

from tools.augment_setfit_multilabel_data import (
    ADDED_PER_PAIR,
    COMPOSITION_TYPES,
    FROZEN_MANUAL_UTTERANCES,
    LABEL_PAIRS,
    augment,
    normalize,
)


DATA = Path(__file__).resolve().parents[1] / "training" / "semantic_routing" / "prepared"


def test_multilabel_augmentation_is_balanced_and_excludes_manual_cases() -> None:
    rows = [json.loads(line) for line in (DATA / "train.jsonl").read_text(encoding="utf-8").splitlines()]
    added = augment(rows)

    assert len(added) == ADDED_PER_PAIR * len(LABEL_PAIRS)
    assert {normalize(str(row["text"])) for row in added}.isdisjoint(
        normalize(text) for text in FROZEN_MANUAL_UTTERANCES
    )
    for pair in LABEL_PAIRS:
        pair_rows = [row for row in added if set(row["capabilities"]) == set(pair)]
        assert len(pair_rows) == ADDED_PER_PAIR
        assert {row["generation"]["composition_type"] for row in pair_rows} == set(COMPOSITION_TYPES)
