from __future__ import annotations

from pathlib import Path

from app.models import ChatMessage
from app.music_control import RuleBasedMusicActionParser
from app.pc_control import default_app_registry
from app.routing import RoutingRequest, create_default_semantic_router
from evaluation.music_interpretation import (
    MusicInterpretationCase,
    MusicInterpretationPrediction,
    evaluate_music_interpretation,
    load_music_interpretation_corpus,
)


CORPUS_DIR = Path(__file__).resolve().parents[1] / "evaluation" / "music_interpretation"


def _rule_prediction(case: MusicInterpretationCase) -> MusicInterpretationPrediction:
    history = tuple(ChatMessage(turn.role, turn.content) for turn in case.context)
    decision = create_default_semantic_router(default_app_registry()).route(
        RoutingRequest(case.text, history),
    )
    if "music_control" not in decision.required_capabilities:
        return MusicInterpretationPrediction("rejected")
    parsed = RuleBasedMusicActionParser().parse(case.text)
    if not parsed.ok:
        return MusicInterpretationPrediction("unsupported")
    return MusicInterpretationPrediction("parsed", parsed.actions)


def test_holdout_corpus_covers_required_interpretation_boundaries() -> None:
    cases = load_music_interpretation_corpus(CORPUS_DIR / "holdout.jsonl")
    categories = {case.category for case in cases}

    assert len(cases) == 55
    assert {
        "simple_transport",
        "song_request",
        "artist_qualified_song",
        "artist_playback",
        "playlist_playback",
        "playlist_track_playback",
        "same_tool_multi_action_sequence",
        "conversational_wrapper",
        "cross_script_entity",
        "translated_title",
        "artist_alias_variation",
        "context_dependent",
        "ambiguous_request",
        "hard_negative",
        "unsupported_request",
        "malformed_request",
    } <= categories
    field_failures = {case.text for case in cases if "field_failure" in case.tags}
    assert {
        "Backnumber 플레이리스트 있잖아 틀어",
        "안녕 크리스 적적한데 백넘버 플레이리스트 틀어줘",
        "내 굿나잇 플레이리스트에서 아무 노래나 틀어줘",
        "즛토마요 하나이치몬메 틀어줘",
        "높은 산의 하나코상 틀어줄래",
        "멈춰줘",
    } <= field_failures


def test_rule_music_interpretation_holdout_baseline_is_preserved() -> None:
    report = evaluate_music_interpretation(
        load_music_interpretation_corpus(CORPUS_DIR / "holdout.jsonl"),
        _rule_prediction,
    )

    assert (report.action.correct, report.action.total) == (34, 51)
    assert (report.entity.correct, report.entity.total) == (35, 65)
    assert (
        report.full_structured_request.correct,
        report.full_structured_request.total,
    ) == (26, 55)
    assert (report.action_sequence_exact.correct, report.action_sequence_exact.total) == (
        36, 55,
    )
    assert (report.ambiguity_handling.correct, report.ambiguity_handling.total) == (0, 3)
    assert (report.context_dependent.correct, report.context_dependent.total) == (0, 5)
    assert (report.alternate_query_coverage.correct, report.alternate_query_coverage.total) == (
        0, 9,
    )


def test_development_and_holdout_ids_are_disjoint() -> None:
    development = load_music_interpretation_corpus(CORPUS_DIR / "development.jsonl")
    holdout = load_music_interpretation_corpus(CORPUS_DIR / "holdout.jsonl")

    assert {case.case_id for case in development}.isdisjoint(
        case.case_id for case in holdout
    )
