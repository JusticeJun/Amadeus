"""Evaluate current tool matching against the semantic routing corpus."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


BRIDGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BRIDGE_ROOT))

from app.tools.weather import KmaWeatherTool  # noqa: E402
from evaluation import evaluate_routing, load_corpus  # noqa: E402


DEFAULT_CORPUS = BRIDGE_ROOT / "evaluation" / "intent_routing.jsonl"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate semantic tool routing offline")
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--json", action="store_true", dest="as_json")
    parser.add_argument("--fail-on-mismatch", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    weather = KmaWeatherTool("", "evaluation", None, None)

    def predict(text: str) -> set[str]:
        return {"weather"} if weather.matches(text) else set()

    report = evaluate_routing(
        load_corpus(args.corpus),
        predict,
        latency_iterations=args.iterations,
    )
    if args.as_json:
        print(json.dumps(report.as_dict(), ensure_ascii=False, indent=2))
    else:
        _print_report(report)
    return 1 if args.fail_on_mismatch and report.mismatch_count else 0


def _print_report(report) -> None:
    print(f"scored={report.scored_cases} ambiguous={report.ambiguous_cases}")
    for tool, metrics in report.tool_metrics.items():
        print(
            f"{tool}: precision={metrics.precision:.3f} recall={metrics.recall:.3f} "
            f"f1={metrics.f1:.3f} tp={metrics.true_positive} "
            f"fp={metrics.false_positive} fn={metrics.false_negative}"
        )
    print(
        f"latency: mean={report.latency.mean_ms:.4f}ms "
        f"p95={report.latency.p95_ms:.4f}ms max={report.latency.max_ms:.4f}ms "
        f"samples={report.latency.samples}"
    )
    for label, mismatches in (
        ("false_positive", report.false_positives),
        ("false_negative", report.false_negatives),
    ):
        for item in mismatches:
            print(f"{label}: {item.case_id} | {item.text}")


if __name__ == "__main__":
    raise SystemExit(main())
