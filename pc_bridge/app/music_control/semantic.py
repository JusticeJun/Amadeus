from __future__ import annotations

from dataclasses import dataclass
import re

from ..models import ChatMessage
from ..semantic_llm import SemanticLlmClient, SemanticLlmError, SemanticLlmRequest
from .actions import MusicAction, MusicActionParseResult, MusicActionSequence, MusicActionType

_MAX_ACTIONS = 4
_MAX_ALTERNATES = 4
_MUSIC_TERMS = ("노래", "곡", "음악", "플레이리스트", "재생", "일시정지", "앨범")
_MUSIC_VERBS = ("틀어", "들려", "멈춰", "정지", "넘겨", "돌아가", "재생")
_SYSTEM_PROMPT = """Convert only Apple Music requests into structured actions.
Return parsed for supported requests, ambiguous when a referenced entity cannot be
resolved from the supplied context, unsupported for malformed or unsupported Apple
Music requests, and not_music for unrelated requests.

Action rules:
- play_song requires title; artist is optional.
- play_artist means play any music by an artist and requires artist, with empty title.
- play_playlist means play a named playlist and requires playlist, with empty title
  and artist.
- play_playlist_track requires both playlist and title; artist is optional.
- transport, list, and now-playing actions have all entity fields empty.
- Preserve the complete user-facing entity spelling, including multi-word names.
- Ignore greetings, assistant vocatives, mood statements, politeness, and request
  filler surrounding the actual music request; never include them in entities.
- Every unused entity field must be an empty string.

Add at most four concise alternate_queries only when a cross-script, translated,
alias, spacing, punctuation, or playlist-name variant helps deterministic search.
Never output catalog IDs, playlist IDs, execution results, current playback facts,
tool calls, conditional plans, or cross-capability actions.

Korean structural examples with placeholders:
- "도우미야 부탁 하나만 가수명 곡명 좀 틀어줄래" means play_song with
  artist="가수명" and title="곡명"; "도우미야 부탁 하나만" and "좀" are filler.
- "오늘 울적한데 목록명 플레이리스트 틀어줘" means play_playlist with
  playlist="목록명"; the mood statement is not part of the playlist name.
These examples describe grammar only; never copy placeholder values into output."""

_ACTION_SCHEMA = {"type": "object", "properties": {
    "type": {"type": "string", "enum": [item.value for item in MusicActionType]},
    "title": {"type": "string", "maxLength": 200},
    "artist": {"type": "string", "maxLength": 200},
    "playlist": {"type": "string", "maxLength": 200},
    "alternate_queries": {"type": "array", "maxItems": _MAX_ALTERNATES,
                          "items": {"type": "string", "minLength": 1, "maxLength": 240}},
    "artist_explicit": {"type": "boolean"},
}, "required": ["type", "title", "artist", "playlist", "alternate_queries",
                 "artist_explicit"], "additionalProperties": False}
MUSIC_SEMANTIC_SCHEMA = {"type": "object", "properties": {
    "status": {"type": "string", "enum": ["parsed", "ambiguous", "unsupported", "not_music"]},
    "actions": {"type": "array", "maxItems": _MAX_ACTIONS, "items": _ACTION_SCHEMA},
}, "required": ["status", "actions"], "additionalProperties": False}


@dataclass
class MusicSemanticMetrics:
    requests: int = 0
    fast_path_hits: int = 0
    llm_calls: int = 0
    llm_fallbacks: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    latency_seconds: float = 0.0
    timeouts: int = 0
    rate_limits: int = 0
    errors: int = 0
    provider: str = ""
    model: str = ""


class MusicSemanticInterpreter:
    """Hybrid interpreter that has no candidate-selection or execution authority."""

    def __init__(self, rule_parser, semantic_client: SemanticLlmClient | None) -> None:
        self._rule_parser = rule_parser
        self._client = semantic_client
        self.metrics = MusicSemanticMetrics()
        self._cached_key: tuple[object, ...] | None = None
        self._cached_result: MusicActionParseResult | None = None

    def is_candidate(self, text: str, history: tuple[ChatMessage, ...] = ()) -> bool:
        compact = "".join(text.lower().split())
        if any(term in compact for term in _MUSIC_TERMS):
            return True
        if any(verb in compact for verb in _MUSIC_VERBS):
            return not any(term in compact for term in ("조명", "불", "영상", "유튜브"))
        return bool(history and _is_deictic(text) and any(
            any(term in item.content.lower() for term in _MUSIC_TERMS)
            for item in history[-4:]
        ))

    def interpret(self, text: str, history: tuple[ChatMessage, ...] = ()) -> MusicActionParseResult:
        context = _minimal_music_context(history)
        key = (text, context)
        if key == self._cached_key and self._cached_result is not None:
            return self._cached_result
        self.metrics.requests += 1
        rule = self._rule_parser.parse(text)
        if _is_safe_fast_path(rule):
            self.metrics.fast_path_hits += 1
            return self._remember(key, rule)
        if self._client is None:
            return self._remember(key, _safe_rule_fallback(text, rule, "provider_unavailable"))
        self.metrics.llm_fallbacks += 1
        try:
            response = self._client.complete(SemanticLlmRequest(
                task="music_interpretation", system_prompt=_SYSTEM_PROMPT,
                input={"utterance": text, "music_context": list(context)},
                schema_name="music_action_sequence", schema=MUSIC_SEMANTIC_SCHEMA,
            ))
            self.metrics.llm_calls += 1
            metrics = response.metrics
            self.metrics.provider, self.metrics.model = metrics.provider, metrics.model
            self.metrics.input_tokens += metrics.input_tokens
            self.metrics.output_tokens += metrics.output_tokens
            self.metrics.latency_seconds += metrics.elapsed_seconds
            result = _parse_semantic_data(response.data)
            if result.error_code == "schema_violation":
                self.metrics.errors += 1
        except SemanticLlmError as exc:
            self.metrics.llm_calls += 1
            if exc.code == "timeout":
                self.metrics.timeouts += 1
            elif exc.code == "rate_limit":
                self.metrics.rate_limits += 1
            else:
                self.metrics.errors += 1
            result = _safe_rule_fallback(text, rule, exc.code)
        return self._remember(key, result)

    def _remember(self, key, result):
        self._cached_key, self._cached_result = key, result
        return result


def _parse_semantic_data(data: dict[str, object]) -> MusicActionParseResult:
    status, raw_actions = data.get("status"), data.get("actions")
    if status not in {"parsed", "ambiguous", "unsupported", "not_music"}:
        return MusicActionParseResult(error_code="schema_violation")
    if not isinstance(raw_actions, list) or len(raw_actions) > _MAX_ACTIONS:
        return MusicActionParseResult(error_code="schema_violation")
    if status != "parsed":
        return MusicActionParseResult(error_code=str(status)) if not raw_actions else MusicActionParseResult(error_code="schema_violation")
    if not raw_actions:
        return MusicActionParseResult(error_code="schema_violation")
    actions: list[MusicAction] = []
    try:
        for raw in raw_actions:
            if not isinstance(raw, dict) or set(raw) != {"type", "title", "artist", "playlist", "alternate_queries", "artist_explicit"}:
                raise ValueError
            if not isinstance(raw["artist_explicit"], bool):
                raise ValueError
            alternates = raw["alternate_queries"]
            if not isinstance(alternates, list) or len(alternates) > _MAX_ALTERNATES:
                raise ValueError
            normalized = ["".join(str(item).casefold().split()) for item in alternates]
            if any(not isinstance(item, str) or not item.strip() for item in alternates) or len(set(normalized)) != len(normalized):
                raise ValueError
            action = MusicAction(
                MusicActionType(str(raw["type"])), _bounded(raw["title"]),
                _bounded(raw["artist"]), _bounded(raw["playlist"]),
                source_query=" ".join(filter(None, (_bounded(raw["artist"]), _bounded(raw["title"])))),
                alternate_queries=tuple(item.strip() for item in alternates),
                artist_explicit=raw["artist_explicit"] is True,
            )
            _validate_action(action)
            actions.append(action)
    except (KeyError, TypeError, ValueError):
        return MusicActionParseResult(error_code="schema_violation")
    return MusicActionParseResult(MusicActionSequence(tuple(actions)))


def _validate_action(action: MusicAction) -> None:
    required = {
        MusicActionType.PLAY_SONG: bool(action.title),
        MusicActionType.PLAY_ARTIST: bool(action.artist),
        MusicActionType.PLAY_PLAYLIST: bool(action.playlist),
        MusicActionType.PLAY_PLAYLIST_TRACK: bool(action.playlist and action.title),
    }.get(action.action_type, True)
    if not required:
        raise ValueError
    allowed = {
        MusicActionType.PLAY_SONG: {"title", "artist"},
        MusicActionType.PLAY_ARTIST: {"artist"},
        MusicActionType.PLAY_PLAYLIST: {"playlist"},
        MusicActionType.PLAY_PLAYLIST_TRACK: {"title", "artist", "playlist"},
    }.get(action.action_type, set())
    if any(getattr(action, field) for field in {"title", "artist", "playlist"} - allowed):
        raise ValueError


def _bounded(value: object) -> str:
    if not isinstance(value, str) or len(value) > 200:
        raise ValueError
    return value.strip()


def _minimal_music_context(history: tuple[ChatMessage, ...]) -> tuple[str, ...]:
    relevant = [f"{item.role}: {item.content[:300]}" for item in history[-6:]
                if any(term in item.content.lower() for term in _MUSIC_TERMS + _MUSIC_VERBS)]
    return tuple(relevant[-2:])


def _is_safe_fast_path(result: MusicActionParseResult) -> bool:
    safe = {MusicActionType.PLAY, MusicActionType.PAUSE, MusicActionType.NEXT,
            MusicActionType.PREVIOUS, MusicActionType.GET_NOW_PLAYING,
            MusicActionType.LIST_PLAYLISTS}
    return result.ok and all(action.action_type in safe for action in result.actions)


def _safe_rule_fallback(text: str, rule: MusicActionParseResult, failure: str) -> MusicActionParseResult:
    compact = "".join(text.strip().rstrip(" ?!.,~").split())
    simple = len(text.split()) <= 4 and bool(re.fullmatch(
        r"[\w가-힣ぁ-んァ-ヶ一-龯.-]{1,100}(?:틀어줘|재생해줘)", compact,
    ))
    return rule if simple and rule.ok and len(rule.actions) == 1 else MusicActionParseResult(error_code=f"semantic_{failure}")


def _is_deictic(text: str) -> bool:
    compact = "".join(text.lower().split())
    return any(term in compact for term in ("그거", "그노래", "그플레이리스트", "아까"))
