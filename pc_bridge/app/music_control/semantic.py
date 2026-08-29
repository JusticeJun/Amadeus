from __future__ import annotations

from dataclasses import dataclass
import re

from ..models import ChatMessage
from ..semantic_llm import SemanticLlmClient, SemanticLlmError, SemanticLlmRequest
from .actions import MusicAction, MusicActionParseResult, MusicActionSequence, MusicActionType
from .controller import CatalogQueryVariant, SongCandidateJudgment

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
  and artist. A request for any/random song from a named playlist is play_playlist;
  "any song" is not a missing or ambiguous track entity.
- play_playlist_track requires both playlist and title; artist is optional.
- transport, list, and now-playing actions have all entity fields empty.
- Preserve the complete user-facing entity spelling, including multi-word names.
- Separate only the final playback command from the preceding entity span. In a
  "<artist surface> <title surface> <playback command>" request, preserve both entity
  slots even when the title surface itself looks imperative or ends in a Korean
  request-like suffix such as "...해줘". Do not reinterpret a verb-like title as a
  second command when a distinct final playback command follows it.
- Ignore greetings, assistant vocatives, mood statements, politeness, and request
  filler surrounding the actual music request; never include them in entities.
- The assistant is named Chris (“크리스”). Treat that name as a vocative when it
  follows a greeting or directly addresses the assistant; do not prepend it to a
  song, artist, or playlist entity.
- Every unused entity field must be an empty string.

Add at most four concise alternate_queries only when a cross-script, translated,
alias, spacing, punctuation, or playlist-name variant helps deterministic search.
When an artist or title is a spoken transliteration or translation rather than the
likely catalog spelling, include a combined artist/title query in likely canonical
script. Do this generically from semantic equivalence; do not rely on a fixed entity
dictionary. If no reliable equivalent is known, leave it out instead of guessing.
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
    "title": {"type": "string"},
    "artist": {"type": "string"},
    "playlist": {"type": "string"},
    "alternate_queries": {"type": "array", "items": {"type": "string"}},
    "artist_explicit": {"type": "boolean"},
}, "required": ["type", "title", "artist", "playlist", "alternate_queries",
                 "artist_explicit"], "additionalProperties": False}
MUSIC_SEMANTIC_SCHEMA = {"type": "object", "properties": {
    "status": {"type": "string", "enum": ["parsed", "ambiguous", "unsupported", "not_music"]},
    "actions": {"type": "array", "items": _ACTION_SCHEMA},
}, "required": ["status", "actions"], "additionalProperties": False}
_QUERY_REWRITE_SCHEMA = {"type": "object", "properties": {
    "variants": {"type": "array", "items": {
        "type": "object", "properties": {
            "artist": {"type": "string"}, "title": {"type": "string"},
        }, "required": ["artist", "title"], "additionalProperties": False,
    }},
}, "required": ["variants"], "additionalProperties": False}
_QUERY_REWRITE_PROMPT = """Act only as a language adapter for Apple Music catalog
search. Convert the supplied structured artist_surface and title_surface into at most
four structured {artist,title} variants likely to be searchable catalog spellings.
Prefer canonical artist orthography and canonical or phonetic title orthography over
word-by-word free-form guesses. When Korean surfaces appear to be localized readings
of foreign entities, prioritize useful cross-script discovery: likely canonical
artist spelling, original-language artist/title, romanized forms, and official/common
English forms when reliably known. A Korean spelling correction alone is not a useful
cross-language variant. Make variants materially distinct discovery strategies rather
than repeating nearby spellings in one script. Variants may bridge localized spellings,
scripts, translations, transliterations, spacing, or punctuation. Preserve the identity of
both supplied entities; use an empty artist only when no artist was supplied. Do not
create catalog IDs, choose a track, claim a match, or report execution. Do not repeat
the original pair. Return an empty list when no reliable adapter variant is known.
Every variant is only a search query; native Apple Music results remain authoritative."""
_CANDIDATE_JUDGE_SCHEMA = {"type": "object", "properties": {
    "status": {"type": "string", "enum": ["match", "uncertain", "none"]},
    "candidate_index": {"type": "integer"},
    "title_equivalent": {"type": "boolean"},
    "artist_equivalent": {"type": "boolean"},
}, "required": ["status", "candidate_index", "title_equivalent", "artist_equivalent"],
   "additionalProperties": False}
_PLAYLIST_CANDIDATE_SCHEMA = {"type": "object", "properties": {
    "status": {"type": "string", "enum": ["match", "ambiguous", "no_match"]},
    "candidate_index": {"type": "integer"},
}, "required": ["status", "candidate_index"], "additionalProperties": False}
_SONG_CANDIDATE_PROMPT = """Judge semantic equivalence only among supplied Apple
Music song candidates. Compare the user's artist/title surface with candidate
metadata across scripts, translations, and transliterations. A Korean surface may be
the phonetic reading of Japanese kanji or kana with no shared characters; reason from
the candidate title's pronunciation rather than requiring string overlap. Artist
equivalence must also hold. Report title and artist equivalence independently. Return
match only when both are true and one supplied candidate is clearly
the same song and artist; otherwise return uncertain or none with candidate_index=-1.
Never create an entity, catalog ID, execution result, or candidate outside the
supplied list."""
_PLAYLIST_CANDIDATE_PROMPT = """Judge semantic fit only among supplied personal
playlist names. Return match with the supplied candidate_index only when one existing
name is clearly preferred by the user surface. Return ambiguous when multiple supplied
names remain plausible, or no_match when none fit, both with candidate_index=-1.
Never create or rename a playlist and never claim execution."""


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
    last_error: str = ""


@dataclass(frozen=True)
class PlaylistCandidateJudgment:
    outcome: str
    index: int = -1
    error_category: str = ""

    @property
    def approved(self) -> bool:
        return self.outcome == "match" and self.index >= 0 and not self.error_category


class MusicSemanticInterpreter:
    """Hybrid interpreter that has no candidate-selection or execution authority."""

    def __init__(self, rule_parser, semantic_client: SemanticLlmClient | None) -> None:
        self._rule_parser = rule_parser
        self._client = semantic_client
        self.metrics = MusicSemanticMetrics()
        self._cached_key: tuple[object, ...] | None = None
        self._cached_result: MusicActionParseResult | None = None
        self.last_song_judgment: SongCandidateJudgment | None = None

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
        self.metrics.last_error = ""
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
            self.metrics.last_error = str(exc)
            result = _safe_rule_fallback(text, rule, exc.code)
        return self._remember(key, result)

    def rewrite_track_queries(
        self, artist_surface: str, title_surface: str,
    ) -> tuple[CatalogQueryVariant, ...]:
        if self._client is None:
            return ()
        try:
            response = self._client.complete(SemanticLlmRequest(
                task="music_catalog_query_rewrite", system_prompt=_QUERY_REWRITE_PROMPT,
                input={
                    "artist_surface": artist_surface, "title_surface": title_surface,
                },
                schema_name="music_catalog_query_rewrite", schema=_QUERY_REWRITE_SCHEMA,
            ))
            self.metrics.llm_calls += 1
            metrics = response.metrics
            self.metrics.input_tokens += metrics.input_tokens
            self.metrics.output_tokens += metrics.output_tokens
            self.metrics.latency_seconds += metrics.elapsed_seconds
            raw = response.data.get("variants")
            if not isinstance(raw, list) or len(raw) > _MAX_ALTERNATES:
                return ()
            source_key = (_query_key(artist_surface), _query_key(title_surface))
            seen = {source_key}
            variants = []
            for item in raw:
                if not isinstance(item, dict) or set(item) != {"artist", "title"}:
                    return ()
                artist, title = item["artist"], item["title"]
                if not isinstance(artist, str) or not isinstance(title, str):
                    return ()
                artist, title = artist.strip(), title.strip()
                if not title or len(artist) > 200 or len(title) > 200:
                    return ()
                key = (_query_key(artist), _query_key(title))
                if key not in seen:
                    seen.add(key)
                    variants.append(CatalogQueryVariant(artist, title))
            return tuple(variants)
        except SemanticLlmError:
            self.metrics.llm_calls += 1
            return ()

    def judge_song_candidate(
        self, action: MusicAction, candidates: tuple[object, ...],
    ) -> SongCandidateJudgment | None:
        candidate_data = [{
            "index": index,
            "title": str(getattr(item, "title", "")),
            "artist": str(getattr(item, "artist", "")),
            "album": str(getattr(item, "album", "")),
            "rank": int(getattr(item, "search_rank", index)),
        } for index, item in enumerate(candidates[:10])]
        if not candidate_data:
            return self._remember_song_judgment(SongCandidateJudgment(
                "none", rejection_reason="no_match",
            ))
        if self._client is None:
            return self._remember_song_judgment(SongCandidateJudgment(
                "error", error_category="provider_error",
            ))
        self.metrics.last_error = ""
        try:
            response = self._client.complete(SemanticLlmRequest(
                task="music_song_candidate_judgment", system_prompt=_SONG_CANDIDATE_PROMPT,
                input={"artist_surface": action.artist, "title_surface": action.title,
                       "candidates": candidate_data},
                schema_name="music_candidate_judgment", schema=_CANDIDATE_JUDGE_SCHEMA,
            ))
            self.metrics.llm_calls += 1
            index = response.data.get("candidate_index")
            title_equivalent = response.data.get("title_equivalent")
            artist_equivalent = response.data.get("artist_equivalent")
            status = response.data.get("status")
            if not isinstance(index, int) or not 0 <= index < len(candidate_data):
                reason = "no_match" if status in {"none", "uncertain"} and index == -1 else "invalid_index"
            elif title_equivalent is not True:
                reason = "title_not_equivalent"
            elif action.artist and artist_equivalent is not True:
                reason = "artist_not_equivalent"
            elif status != "match":
                reason = "no_match"
            else:
                reason = ""
            return self._remember_song_judgment(SongCandidateJudgment(
                str(status or "invalid"), index if isinstance(index, int) else -1,
                title_equivalent is True, artist_equivalent is True,
                rejection_reason=reason,
            ))
        except SemanticLlmError as exc:
            self.metrics.llm_calls += 1
            if exc.code == "timeout":
                self.metrics.timeouts += 1
            elif exc.code == "rate_limit":
                self.metrics.rate_limits += 1
            else:
                self.metrics.errors += 1
            self.metrics.last_error = str(exc)
            return self._remember_song_judgment(SongCandidateJudgment(
                "error", error_category=_candidate_error_category(exc),
            ))

    def _remember_song_judgment(
        self, judgment: SongCandidateJudgment,
    ) -> SongCandidateJudgment:
        self.last_song_judgment = judgment
        return judgment

    def judge_playlist_candidate(
        self, surface: str, candidates: tuple[object, ...],
    ) -> int | None:
        judgment = self.select_playlist_candidate(
            surface, tuple(str(getattr(item, "name", "")) for item in candidates[:10]),
        )
        return judgment.index if judgment.approved else None

    def select_playlist_candidate(
        self, surface: str, options: tuple[str, ...],
    ) -> PlaylistCandidateJudgment:
        candidate_data = [
            {"index": index, "name": name}
            for index, name in enumerate(options[:10])
        ]
        return self._judge_candidate(
            "music_playlist_candidate_judgment", _PLAYLIST_CANDIDATE_PROMPT,
            {"playlist_surface": surface, "candidates": candidate_data},
            len(candidate_data),
        )

    def _judge_candidate(
        self, task: str, prompt: str, input_data: dict[str, object], count: int,
    ) -> PlaylistCandidateJudgment:
        if self._client is None or not count:
            return PlaylistCandidateJudgment("error", error_category="provider_error")
        try:
            response = self._client.complete(SemanticLlmRequest(
                task=task, system_prompt=prompt, input=input_data,
                schema_name="music_playlist_candidate_judgment",
                schema=_PLAYLIST_CANDIDATE_SCHEMA,
            ))
            self.metrics.llm_calls += 1
            metrics = response.metrics
            self.metrics.input_tokens += metrics.input_tokens
            self.metrics.output_tokens += metrics.output_tokens
            self.metrics.latency_seconds += metrics.elapsed_seconds
            status = response.data.get("status")
            index = response.data.get("candidate_index")
            if status not in {"match", "ambiguous", "no_match"}:
                return PlaylistCandidateJudgment("invalid", error_category="schema_error")
            if status == "match" and isinstance(index, int) and 0 <= index < count:
                return PlaylistCandidateJudgment("match", index)
            if status in {"ambiguous", "no_match"} and index == -1:
                return PlaylistCandidateJudgment(status)
            return PlaylistCandidateJudgment("invalid", error_category="schema_error")
        except SemanticLlmError as exc:
            self.metrics.llm_calls += 1
            self.metrics.last_error = str(exc)
            return PlaylistCandidateJudgment(
                "error", error_category=_candidate_error_category(exc),
            )

    def continue_clarification(
        self, pending: dict[str, object], text: str,
        history: tuple[ChatMessage, ...] = (),
    ) -> MusicActionParseResult:
        """Extract a missing slot through the same bounded semantic contract."""
        action = pending.get("action")
        if (
            pending.get("kind") != "missing_entity"
            or pending.get("field") != "artist"
            or not isinstance(action, dict)
            or action.get("type") != MusicActionType.PLAY_SONG.value
            or not isinstance(action.get("title"), str)
        ):
            return MusicActionParseResult(error_code="unsupported")
        title = action["title"].strip()
        if not title:
            return MusicActionParseResult(error_code="unsupported")
        request = (
            "Complete this unresolved Apple Music request. The song title is "
            f"{title!r}; the user was asked only for its artist and answered "
            f"{text!r}. Remove conversational agreement, discourse wrappers, "
            "pronouns, copulas, and sentence endings from the artist slot."
        )
        result = self.interpret(request, history)
        candidate = result.action
        if (
            candidate is None
            or candidate.action_type is not MusicActionType.PLAY_SONG
            or candidate.title != title
            or not candidate.artist
            or not candidate.artist_explicit
        ):
            return MusicActionParseResult(error_code="ambiguous")
        return result

    def continue_track_correction(
        self, pending: dict[str, object], text: str,
        history: tuple[ChatMessage, ...] = (),
    ) -> MusicActionParseResult:
        action = pending.get("action")
        if (
            pending.get("kind") != "track_correction"
            or not isinstance(action, dict)
            or action.get("type") != MusicActionType.PLAY_SONG.value
        ):
            return MusicActionParseResult(error_code="unsupported")
        request = (
            "The immediately previous Apple Music track request failed without playback. "
            f"The user now said {text!r}. Interpret it as a replacement play_song only "
            "when it clearly supplies a corrected artist/title pair. Remove discourse "
            "wrappers and do not reuse an uncertain old entity. Otherwise return ambiguous."
        )
        result = self.interpret(request, history)
        candidate = result.action
        if (
            candidate is None
            or candidate.action_type is not MusicActionType.PLAY_SONG
            or not candidate.title
            or not candidate.artist
            or not candidate.artist_explicit
        ):
            return MusicActionParseResult(error_code="ambiguous")
        return result

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


def _query_key(value: str) -> str:
    return "".join(value.casefold().split())


def _candidate_error_category(exc: SemanticLlmError) -> str:
    if exc.code == "malformed_response":
        return "malformed_response"
    if exc.code in {"schema_error", "schema_violation", "json_validate_failed"} \
            or "schema_mismatch=" in str(exc):
        return "schema_error"
    return "provider_error"


def _safe_rule_fallback(text: str, rule: MusicActionParseResult, failure: str) -> MusicActionParseResult:
    compact = "".join(text.strip().rstrip(" ?!.,~").split())
    simple = len(text.split()) <= 4 and bool(re.fullmatch(
        r"[\w가-힣ぁ-んァ-ヶ一-龯.-]{1,100}(?:틀어줘|재생해줘)", compact,
    ))
    return rule if simple and rule.ok and len(rule.actions) == 1 else MusicActionParseResult(error_code=f"semantic_{failure}")


def _is_deictic(text: str) -> bool:
    compact = "".join(text.lower().split())
    return any(term in compact for term in ("그거", "그노래", "그플레이리스트", "아까"))
