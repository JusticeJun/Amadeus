"""Evaluate Music-internal structured interpretation without playback."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


BRIDGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BRIDGE_ROOT))

from app.models import ChatMessage  # noqa: E402
from app.config import Settings  # noqa: E402
from app.groq_semantic import GroqSemanticLlmClient  # noqa: E402
from app.music_control import MusicSemanticInterpreter, RuleBasedMusicActionParser  # noqa: E402
from app.pc_control import default_app_registry  # noqa: E402
from app.routing import RoutingRequest, create_default_semantic_router  # noqa: E402
from evaluation.music_interpretation import (  # noqa: E402
    MusicInterpretationCase,
    MusicInterpretationPrediction,
    evaluate_music_interpretation,
    load_music_interpretation_corpus,
)


DEFAULT_CORPUS = (
    BRIDGE_ROOT / "evaluation" / "music_interpretation" / "holdout.jsonl"
)
DEVELOPMENT_CORPUS = DEFAULT_CORPUS.with_name("development.jsonl")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate structured Music interpretation offline",
    )
    parser.add_argument("--corpus", type=Path)
    parser.add_argument("--json", action="store_true", dest="as_json")
    parser.add_argument("--hybrid", action="store_true")
    args = parser.parse_args()
    rule_parser = RuleBasedMusicActionParser()
    interpreter = None
    if args.hybrid:
        interpreter = MusicSemanticInterpreter(
            rule_parser, GroqSemanticLlmClient(Settings.from_env()),
        )
    router = create_default_semantic_router(default_app_registry(), interpreter)

    def predict(case: MusicInterpretationCase) -> MusicInterpretationPrediction:
        history = tuple(ChatMessage(turn.role, turn.content) for turn in case.context)
        decision = router.route(RoutingRequest(case.text, history))
        if "music_control" not in decision.required_capabilities:
            return MusicInterpretationPrediction("rejected")
        parsed = interpreter.interpret(case.text, history) if interpreter else rule_parser.parse(
            case.text
        )
        if not parsed.ok:
            status = parsed.error_code if parsed.error_code in {
                "ambiguous", "unsupported", "not_music",
            } else "unsupported"
            return MusicInterpretationPrediction(status)
        return MusicInterpretationPrediction("parsed", parsed.actions)

    report = evaluate_music_interpretation(
        load_music_interpretation_corpus(
            args.corpus or (DEVELOPMENT_CORPUS if args.hybrid else DEFAULT_CORPUS)
        ), predict,
    )
    if args.as_json:
        payload = {"report": report.as_dict()}
        if interpreter:
            payload["runtime"] = interpreter.metrics.__dict__
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        _print_report(report)
        if interpreter:
            metrics = interpreter.metrics
            print(
                f"runtime: provider={metrics.provider} model={metrics.model} "
                f"requests={metrics.requests} fast_path={metrics.fast_path_hits} "
                f"llm_calls={metrics.llm_calls} llm_fallbacks={metrics.llm_fallbacks} "
                f"input_tokens={metrics.input_tokens} output_tokens={metrics.output_tokens} "
                f"latency={metrics.latency_seconds:.3f}s timeouts={metrics.timeouts} "
                f"rate_limits={metrics.rate_limits} errors={metrics.errors}"
            )
    return 0


def _print_report(report) -> None:
    print(f"cases={report.cases}")
    for name in (
        "action", "entity", "full_structured_request", "action_sequence_exact",
        "ambiguity_handling", "context_dependent", "alternate_query_coverage",
        "alternate_query_boundedness",
    ):
        metric = getattr(report, name)
        print(f"{name}: {metric.correct}/{metric.total} ({metric.accuracy:.3f})")
    for category, result in report.categories.items():
        accuracy = result.full_structured_correct / result.cases
        print(
            f"category {category}: {result.full_structured_correct}/{result.cases} "
            f"({accuracy:.3f})"
        )
    for mismatch in report.mismatches:
        print(
            f"mismatch: {mismatch.case_id} | {mismatch.expected_status} -> "
            f"{mismatch.predicted_status} | {mismatch.text}"
        )


if __name__ == "__main__":
    raise SystemExit(main())
