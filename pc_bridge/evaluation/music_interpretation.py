from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
from typing import Callable, Iterable
import unicodedata

from app.music_control import MusicAction, MusicActionType


_ENTITY_FIELDS = ("title", "artist", "playlist")


@dataclass(frozen=True)
class MusicContext:
    role: str
    content: str


@dataclass(frozen=True)
class ExpectedMusicAction:
    action_type: MusicActionType
    title: str = ""
    artist: str = ""
    playlist: str = ""
    required_alternate_queries: tuple[str, ...] = ()


@dataclass(frozen=True)
class MusicInterpretationCase:
    case_id: str
    text: str
    category: str
    expected_status: str
    expected_actions: tuple[ExpectedMusicAction, ...] = ()
    tags: frozenset[str] = frozenset()
    context: tuple[MusicContext, ...] = ()


@dataclass(frozen=True)
class MusicInterpretationPrediction:
    status: str
    actions: tuple[MusicAction, ...] = ()


@dataclass(frozen=True)
class Metric:
    correct: int
    total: int

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total else 0.0


@dataclass(frozen=True)
class CategoryResult:
    cases: int
    full_structured_correct: int


@dataclass(frozen=True)
class MusicInterpretationMismatch:
    case_id: str
    text: str
    category: str
    expected_status: str
    predicted_status: str
    expected_actions: tuple[dict[str, str], ...]
    predicted_actions: tuple[dict[str, str], ...]


@dataclass(frozen=True)
class MusicInterpretationReport:
    cases: int
    action: Metric
    entity: Metric
    full_structured_request: Metric
    action_sequence_exact: Metric
    ambiguity_handling: Metric
    context_dependent: Metric
    alternate_query_coverage: Metric
    alternate_query_boundedness: Metric
    categories: dict[str, CategoryResult]
    mismatches: tuple[MusicInterpretationMismatch, ...]

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


MusicInterpretationPredictor = Callable[
    [MusicInterpretationCase], MusicInterpretationPrediction
]


def load_music_interpretation_corpus(path: Path) -> list[MusicInterpretationCase]:
    cases: list[MusicInterpretationCase] = []
    seen_ids: set[str] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
            expected = item["expected"]
            actions = tuple(
                ExpectedMusicAction(
                    MusicActionType(action["type"]),
                    str(action.get("title") or ""),
                    str(action.get("artist") or ""),
                    str(action.get("playlist") or ""),
                    tuple(str(query) for query in action.get(
                        "required_alternate_queries", [],
                    )),
                )
                for action in expected.get("actions", [])
            )
            context = tuple(
                MusicContext(str(turn["role"]), str(turn["content"]))
                for turn in item.get("context", [])
            )
            case = MusicInterpretationCase(
                case_id=str(item["id"]),
                text=str(item["text"]),
                category=str(item["category"]),
                expected_status=str(expected["status"]),
                expected_actions=actions,
                tags=frozenset(str(tag) for tag in item.get("tags", [])),
                context=context,
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid music corpus entry at line {line_number}") from exc
        if case.case_id in seen_ids:
            raise ValueError(f"duplicate music corpus id: {case.case_id}")
        if case.expected_status == "parsed" and not case.expected_actions:
            raise ValueError(f"parsed case has no actions at line {line_number}")
        if case.expected_status != "parsed" and case.expected_actions:
            raise ValueError(f"non-parsed case has actions at line {line_number}")
        seen_ids.add(case.case_id)
        cases.append(case)
    if not cases:
        raise ValueError("music interpretation corpus is empty")
    return cases


def evaluate_music_interpretation(
    cases: Iterable[MusicInterpretationCase],
    predictor: MusicInterpretationPredictor,
    *,
    max_alternate_queries: int = 4,
) -> MusicInterpretationReport:
    case_list = list(cases)
    predictions = [predictor(case) for case in case_list]
    action_correct = action_total = 0
    entity_correct = entity_total = 0
    sequence_correct = 0
    full_correct = 0
    ambiguity_correct = ambiguity_total = 0
    context_correct = context_total = 0
    alternate_correct = alternate_total = 0
    bounded_correct = bounded_total = 0
    category_results: dict[str, list[bool]] = {}
    mismatches: list[MusicInterpretationMismatch] = []

    for case, prediction in zip(case_list, predictions):
        expected_types = tuple(action.action_type for action in case.expected_actions)
        predicted_types = tuple(action.action_type for action in prediction.actions)
        sequence_match = (
            case.expected_status == prediction.status
            and expected_types == predicted_types
        )
        sequence_correct += sequence_match

        if case.expected_status == "parsed":
            width = max(len(case.expected_actions), len(prediction.actions))
            action_total += width
            for index in range(width):
                expected_action = (
                    case.expected_actions[index]
                    if index < len(case.expected_actions) else None
                )
                predicted_action = (
                    prediction.actions[index]
                    if index < len(prediction.actions) else None
                )
                action_correct += bool(
                    expected_action and predicted_action
                    and expected_action.action_type == predicted_action.action_type
                )
                if expected_action:
                    for field in _ENTITY_FIELDS:
                        expected_value = getattr(expected_action, field)
                        predicted_value = (
                            getattr(predicted_action, field) if predicted_action else ""
                        )
                        if expected_value or predicted_value:
                            entity_total += 1
                            entity_correct += _normalize(expected_value) == _normalize(
                                predicted_value
                            )
                    for required in expected_action.required_alternate_queries:
                        alternate_total += 1
                        alternate_correct += bool(predicted_action) and any(
                            _normalize(query) == _normalize(required)
                            for query in predicted_action.alternate_queries
                        )
                if predicted_action:
                    bounded_total += 1
                    alternates = predicted_action.alternate_queries
                    bounded_correct += (
                        len(alternates) <= max_alternate_queries
                        and all(query.strip() for query in alternates)
                        and len({_normalize(query) for query in alternates}) == len(alternates)
                    )

        structured_match = sequence_match and all(
            _action_matches(expected, predicted)
            for expected, predicted in zip(case.expected_actions, prediction.actions)
        )
        full_correct += structured_match
        category_results.setdefault(case.category, []).append(structured_match)
        if case.expected_status == "ambiguous":
            ambiguity_total += 1
            ambiguity_correct += prediction.status == "ambiguous"
        if case.context:
            context_total += 1
            context_correct += structured_match
        if not structured_match:
            mismatches.append(MusicInterpretationMismatch(
                case.case_id,
                case.text,
                case.category,
                case.expected_status,
                prediction.status,
                tuple(_expected_action_data(action) for action in case.expected_actions),
                tuple(_action_data(action) for action in prediction.actions),
            ))

    return MusicInterpretationReport(
        cases=len(case_list),
        action=Metric(action_correct, action_total),
        entity=Metric(entity_correct, entity_total),
        full_structured_request=Metric(full_correct, len(case_list)),
        action_sequence_exact=Metric(sequence_correct, len(case_list)),
        ambiguity_handling=Metric(ambiguity_correct, ambiguity_total),
        context_dependent=Metric(context_correct, context_total),
        alternate_query_coverage=Metric(alternate_correct, alternate_total),
        alternate_query_boundedness=Metric(bounded_correct, bounded_total),
        categories={
            category: CategoryResult(len(results), sum(results))
            for category, results in sorted(category_results.items())
        },
        mismatches=tuple(mismatches),
    )


def _action_matches(expected: ExpectedMusicAction, predicted: MusicAction) -> bool:
    return expected.action_type == predicted.action_type and all(
        _normalize(getattr(expected, field)) == _normalize(getattr(predicted, field))
        for field in _ENTITY_FIELDS
    )


def _normalize(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).casefold()
    return re.sub(r"\s+", "", value)


def _expected_action_data(action: ExpectedMusicAction) -> dict[str, str]:
    return {
        "type": action.action_type.value,
        "title": action.title,
        "artist": action.artist,
        "playlist": action.playlist,
    }


def _action_data(action: MusicAction) -> dict[str, str]:
    return {
        "type": action.action_type.value,
        "title": action.title,
        "artist": action.artist,
        "playlist": action.playlist,
    }
