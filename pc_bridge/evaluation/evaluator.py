from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from statistics import fmean
import time
from typing import Callable, Iterable


ToolPredictor = Callable[[str], set[str]]


@dataclass(frozen=True)
class RoutingCase:
    case_id: str
    text: str
    expected_tools: frozenset[str] | None
    category: str
    note: str = ""


@dataclass(frozen=True)
class ToolMetrics:
    true_positive: int
    false_positive: int
    false_negative: int
    true_negative: int
    precision: float
    recall: float
    f1: float


@dataclass(frozen=True)
class RoutingMismatch:
    case_id: str
    text: str
    category: str
    expected_tools: tuple[str, ...]
    predicted_tools: tuple[str, ...]


@dataclass(frozen=True)
class LatencyMetrics:
    samples: int
    mean_ms: float
    p95_ms: float
    max_ms: float


@dataclass(frozen=True)
class RoutingReport:
    scored_cases: int
    ambiguous_cases: int
    tool_metrics: dict[str, ToolMetrics]
    false_positives: tuple[RoutingMismatch, ...]
    false_negatives: tuple[RoutingMismatch, ...]
    latency: LatencyMetrics

    @property
    def mismatch_count(self) -> int:
        return len(self.false_positives) + len(self.false_negatives)

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def load_corpus(path: Path) -> list[RoutingCase]:
    cases: list[RoutingCase] = []
    seen_ids: set[str] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
            case_id = str(item["id"]).strip()
            text = str(item["text"]).strip()
            category = str(item["category"]).strip()
            raw_tools = item.get("expected_tools")
            expected = None if raw_tools is None else frozenset(str(tool) for tool in raw_tools)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid corpus entry at line {line_number}") from exc
        if not case_id or not text or not category:
            raise ValueError(f"empty required field at line {line_number}")
        if case_id in seen_ids:
            raise ValueError(f"duplicate corpus id: {case_id}")
        if raw_tools is not None and not isinstance(raw_tools, list):
            raise ValueError(f"expected_tools must be a list or null at line {line_number}")
        seen_ids.add(case_id)
        cases.append(RoutingCase(
            case_id=case_id,
            text=text,
            expected_tools=expected,
            category=category,
            note=str(item.get("note") or "").strip(),
        ))
    if not cases:
        raise ValueError("routing corpus is empty")
    return cases


def evaluate_routing(
    cases: Iterable[RoutingCase],
    predictor: ToolPredictor,
    *,
    latency_iterations: int = 100,
) -> RoutingReport:
    case_list = list(cases)
    if latency_iterations < 1:
        raise ValueError("latency_iterations must be positive")

    predictions = [frozenset(predictor(case.text)) for case in case_list]
    scored = [
        (case, prediction)
        for case, prediction in zip(case_list, predictions)
        if case.expected_tools is not None
    ]
    tools = sorted({
        tool
        for case, prediction in scored
        for tool in (case.expected_tools or frozenset()) | prediction
    })
    metrics = {
        tool: _metrics_for_tool(tool, scored)
        for tool in tools
    }

    false_positives: list[RoutingMismatch] = []
    false_negatives: list[RoutingMismatch] = []
    for case, prediction in scored:
        expected = case.expected_tools or frozenset()
        mismatch = RoutingMismatch(
            case_id=case.case_id,
            text=case.text,
            category=case.category,
            expected_tools=tuple(sorted(expected)),
            predicted_tools=tuple(sorted(prediction)),
        )
        if prediction - expected:
            false_positives.append(mismatch)
        if expected - prediction:
            false_negatives.append(mismatch)

    latency_samples = _measure_latency(case_list, predictor, latency_iterations)
    return RoutingReport(
        scored_cases=len(scored),
        ambiguous_cases=len(case_list) - len(scored),
        tool_metrics=metrics,
        false_positives=tuple(false_positives),
        false_negatives=tuple(false_negatives),
        latency=_latency_metrics(latency_samples),
    )


def _metrics_for_tool(
    tool: str,
    scored: list[tuple[RoutingCase, frozenset[str]]],
) -> ToolMetrics:
    tp = fp = fn = tn = 0
    for case, prediction in scored:
        expected = tool in (case.expected_tools or frozenset())
        actual = tool in prediction
        if expected and actual:
            tp += 1
        elif actual:
            fp += 1
        elif expected:
            fn += 1
        else:
            tn += 1
    precision = _ratio(tp, tp + fp)
    recall = _ratio(tp, tp + fn)
    return ToolMetrics(tp, fp, fn, tn, precision, recall, _ratio(2 * precision * recall, precision + recall))


def _measure_latency(
    cases: list[RoutingCase],
    predictor: ToolPredictor,
    iterations: int,
) -> list[float]:
    samples: list[float] = []
    for _ in range(iterations):
        for case in cases:
            started = time.perf_counter_ns()
            predictor(case.text)
            samples.append((time.perf_counter_ns() - started) / 1_000_000)
    return samples


def _latency_metrics(samples: list[float]) -> LatencyMetrics:
    ordered = sorted(samples)
    p95_index = max(0, int(len(ordered) * 0.95 + 0.999999) - 1)
    return LatencyMetrics(
        samples=len(samples),
        mean_ms=fmean(samples),
        p95_ms=ordered[p95_index],
        max_ms=ordered[-1],
    )


def _ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0
