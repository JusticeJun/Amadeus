from __future__ import annotations

from collections import Counter

from tools.build_setfit_balanced_corpus import (
    CONVERSATION_COUNTS,
    PAIR_COUNTS,
    PAIR_SPECS,
    SINGLE_COUNTS,
    build,
    normalize,
)
from tools.audit_setfit_balanced_corpus import has_blocking_failure


def test_balanced_corpus_has_planned_slices_and_unique_text() -> None:
    for split in ("train", "validation"):
        rows = build(split)
        expected = (
            CONVERSATION_COUNTS[split]
            + sum(SINGLE_COUNTS[split].values())
            + len(PAIR_SPECS) * sum(PAIR_COUNTS[split].values())
        )
        assert len(rows) == expected
        assert len({normalize(str(row["text"])) for row in rows}) == expected
        assert all(row["semantic"]["domains"] is not None for row in rows)


def test_pair_families_include_full_partial_neither_and_ambiguous() -> None:
    for split in ("train", "validation"):
        rows = [row for row in build(split) if row["semantic"]["composition"] != "single"]
        counts = Counter(row["semantic"]["composition"] for row in rows)
        assert counts == Counter({
            role: count * len(PAIR_SPECS) for role, count in PAIR_COUNTS[split].items()
        })
        assert any(len(row["capabilities"]) == 2 for row in rows)
        assert any(len(row["capabilities"]) == 1 for row in rows)
        assert any(not row["capabilities"] for row in rows)


def test_audit_failure_detects_overlap() -> None:
    report = {
        "train_validation_normalized_overlap": 1,
        "train_validation_near_overlap": 0,
        "splits": {},
    }
    assert has_blocking_failure(report)
