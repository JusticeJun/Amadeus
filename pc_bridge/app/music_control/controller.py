from __future__ import annotations

from dataclasses import dataclass
import json
import os
import re
import sys
import time
import unicodedata
from typing import Protocol

from .actions import MusicAction, MusicActionResult, MusicActionType


_MAX_TRACK_QUERY_VARIANTS = 4


def _retrieval_diagnostic(stage: str, **data: object) -> None:
    if os.environ.get("AMADEUS_MUSIC_DIAGNOSTICS") == "1":
        print(
            f"[music_retrieval_diagnostic] {stage}:" + json.dumps(
                data, ensure_ascii=False, separators=(",", ":"),
            ),
            file=sys.stderr,
        )


_MIN_RELEVANCE_RANK_GAP = 4


@dataclass(frozen=True)
class MusicItem:
    item_id: str
    title: str
    artist: str = ""
    album: str = ""
    recording_id: str = ""
    search_rank: int = 0


@dataclass(frozen=True)
class PlaylistItem:
    playlist_id: str
    name: str


@dataclass(frozen=True)
class PlaylistSnapshot:
    items: tuple[PlaylistItem, ...]
    partial: bool = False
    warning: str = ""


@dataclass(frozen=True)
class PersonalMusicItem:
    item: MusicItem
    in_library: bool = False
    playlist_count: int = 0


@dataclass(frozen=True)
class PersonalMusicSnapshot:
    items: tuple[PersonalMusicItem, ...]
    partial: bool = False
    warning: str = ""


@dataclass(frozen=True)
class SongCandidateJudgment:
    outcome: str
    index: int = -1
    title_equivalent: bool = False
    artist_equivalent: bool = False
    rejection_reason: str = ""
    error_category: str = ""

    @property
    def approved(self) -> bool:
        return (
            self.outcome == "match"
            and self.index >= 0
            and self.title_equivalent
            and not self.rejection_reason
            and not self.error_category
        )


@dataclass(frozen=True)
class CatalogQueryVariant:
    artist: str = ""
    title: str = ""

    @property
    def query(self) -> str:
        return " ".join(filter(None, (self.artist.strip(), self.title.strip())))


class AppleMusicBackend(Protocol):
    def selector_health(self) -> dict[str, bool]: ...
    def search_songs(self, query: str) -> tuple[MusicItem, ...]: ...
    def search_artists(self, query: str) -> tuple[MusicItem, ...]: ...
    def search_playlists(self, query: str) -> tuple[PlaylistItem, ...]: ...
    def list_playlists(self) -> PlaylistSnapshot: ...
    def personal_songs(self) -> PersonalMusicSnapshot: ...
    def play_song(self, item_id: str) -> MusicItem: ...
    def play_artist(self, item_id: str) -> MusicItem: ...
    def load_playlist(self, playlist_id: str) -> tuple[MusicItem, ...]: ...
    def playlist_tracks(self, playlist_id: str) -> tuple[MusicItem, ...]: ...
    def play_queue_item(self, index: int) -> MusicItem: ...
    def play(self) -> MusicItem: ...
    def pause(self) -> MusicItem: ...
    def next(self) -> MusicItem: ...
    def previous(self) -> MusicItem: ...
    def now_playing(self) -> MusicItem: ...


class MusicCandidateSemantics(Protocol):
    def judge_song_candidate(
        self, action: MusicAction, candidates: tuple[MusicItem, ...],
    ) -> SongCandidateJudgment | None: ...
    def judge_playlist_candidate(
        self, surface: str, candidates: tuple[PlaylistItem, ...],
    ) -> int | None: ...
    def rewrite_track_queries(
        self, artist_surface: str, title_surface: str,
    ) -> tuple[CatalogQueryVariant, ...]: ...


class TrackResolver:
    """Resolve one request against actual catalog/library candidates."""

    def __init__(self, semantics: MusicCandidateSemantics | None = None) -> None:
        self._semantics = semantics

    def resolve(
        self, action: MusicAction, candidates: tuple[MusicItem, ...],
    ) -> MusicItem:
        deterministic = _deterministic_track_matches(action, candidates)
        if len(deterministic) == 1:
            return deterministic[0]
        if len(deterministic) > 1:
            raise MusicControlError("ambiguous", "multiple songs matched")
        if not candidates:
            raise MusicControlError("no_match", "no song candidates were retrieved")
        if self._semantics is None:
            raise MusicControlError("no_match", "no reasonable song candidate")
        bounded = candidates[:10]
        judgment = self._semantics.judge_song_candidate(action, bounded)
        if (
            judgment is None
            or not judgment.approved
            or not judgment.title_equivalent
            or (action.artist and not judgment.artist_equivalent)
            or not 0 <= judgment.index < len(bounded)
        ):
            raise MusicControlError("no_match", "no reasonable song candidate")
        return bounded[judgment.index]


class AppleMusicPwaController:
    def __init__(
        self,
        backend: AppleMusicBackend,
        *,
        playlist_cache_seconds: float = 60.0,
        personal_music_cache_seconds: float = 300.0,
        health_cache_seconds: float = 30.0,
        candidate_semantics: MusicCandidateSemantics | None = None,
        clock=time.monotonic,
    ) -> None:
        self._backend = backend
        self._cache_seconds = playlist_cache_seconds
        self._health_cache_seconds = health_cache_seconds
        self._personal_cache_seconds = personal_music_cache_seconds
        self._candidate_semantics = candidate_semantics
        self._track_resolver = TrackResolver(candidate_semantics)
        self._clock = clock
        self._playlist_cache: tuple[float, PlaylistSnapshot] | None = None
        self._personal_cache: tuple[float, PersonalMusicSnapshot] | None = None
        self._last_healthy_at: float | None = None

    def execute(self, action: MusicAction) -> MusicActionResult:
        try:
            self._ensure_backend_health()
            data = self._execute(action)
            return MusicActionResult(action, True, data)
        except MusicControlError as exc:
            return MusicActionResult(action, False, {
                "reason": exc.code,
                **({"candidate_options": list(exc.candidate_options)}
                   if exc.candidate_options else {}),
            }, str(exc))
        except (OSError, TimeoutError) as exc:
            return MusicActionResult(action, False, {"reason": "backend_unavailable"}, str(exc))

    def _ensure_backend_health(self) -> None:
        now = self._clock()
        if (
            self._last_healthy_at is not None
            and now - self._last_healthy_at < self._health_cache_seconds
        ):
            return
        health = self._backend.selector_health()
        if not health.get("authorized") or not health.get("player"):
            raise MusicControlError(
                "pwa_unavailable", "Apple Music session or player is unavailable",
            )
        self._last_healthy_at = now

    def _execute(self, action: MusicAction) -> dict[str, object]:
        kind = action.action_type
        if kind is MusicActionType.LIST_PLAYLISTS:
            snapshot = self._playlists()
            return {
                "playlists": [item.name for item in snapshot.items],
                "partial": snapshot.partial,
                **({"warning": snapshot.warning} if snapshot.warning else {}),
            }
        if kind is MusicActionType.PLAY_PLAYLIST:
            playlist = self._playlist(action.playlist, action.alternate_queries)
            queue = self._backend.load_playlist(playlist.playlist_id)
            if not queue:
                raise MusicControlError("not_found", "playlist is empty")
            actual = self._backend.play_queue_item(0)
            return _verified_data(actual, queue[0]) | {
                "playlist": playlist.name, "queue_length": len(queue),
            }
        if kind is MusicActionType.PLAY_PLAYLIST_TRACK:
            playlist = self._playlist(action.playlist, action.alternate_queries)
            queue = self._backend.load_playlist(playlist.playlist_id)
            expected, index = _unique_music_match(
                queue, action.title, action.artist, action.alternate_queries,
            )
            actual = self._backend.play_queue_item(index)
            return _verified_data(actual, expected) | {
                "playlist": playlist.name, "queue_position": index,
                "queue_length": len(queue),
            }
        if kind is MusicActionType.PLAY_SONG:
            query = action.source_query or " ".join(filter(None, (action.artist, action.title)))
            _retrieval_diagnostic(
                "parsed", action=kind.value, artist=action.artist, title=action.title,
                source_query=action.source_query,
            )
            _retrieval_diagnostic("search", query=query, backend="apple_music_catalog")
            native_results = self._backend.search_songs(query)
            rewritten_results: tuple[tuple[MusicItem, ...], ...] = ()
            rewrite_variants: tuple[CatalogQueryVariant, ...] = ()
            if not native_results and self._candidate_semantics is not None:
                rewrite_variants = self._candidate_semantics.rewrite_track_queries(
                    action.artist, action.title,
                )[:_MAX_TRACK_QUERY_VARIANTS]
                _retrieval_diagnostic(
                    "query_rewrite", invoked="yes", reason="initial_catalog_empty",
                    variants=[{
                        "artist": variant.artist, "title": variant.title,
                    } for variant in rewrite_variants],
                )
                rewritten_results = tuple(
                    self._backend.search_songs(variant.query)
                    for variant in rewrite_variants if variant.query
                )
            else:
                _retrieval_diagnostic(
                    "query_rewrite", invoked="no",
                    reason=(
                        "initial_catalog_results_present"
                        if native_results else "rewriter_unavailable"
                    ),
                )
            catalog_candidates = _merge_search_results((
                native_results, *rewritten_results,
            ))
            personal = self._personal_music()
            candidates = _merge_search_results((
                catalog_candidates, tuple(entry.item for entry in personal.items),
            ))
            _retrieval_diagnostic(
                "candidate_pool", native_catalog_count=len(native_results),
                rewritten_catalog_count=sum(map(len, rewritten_results)),
                personal_index_count=len(personal.items),
                merged_candidate_count=len(candidates),
                rejection_reason=("no_actual_candidates" if not candidates else ""),
            )
            expected = self._track_resolver.resolve(action, candidates)
            return _verified_data(self._backend.play_song(expected.item_id), expected)
        if kind is MusicActionType.PLAY_ARTIST:
            artist = self._resolve_artist(action.artist, action.alternate_queries)
            candidates = self._backend.search_artists(artist)
            expected = _unique_match(candidates, artist, lambda item: item.artist)
            actual = self._backend.play_artist(expected.item_id)
            if _normalize(actual.artist) != _normalize(expected.artist):
                raise MusicControlError("metadata_mismatch", "now-playing artist did not match")
            return {"artist": expected.artist, "now_playing": _item_data(actual)}
        operations = {
            MusicActionType.PLAY: self._backend.play,
            MusicActionType.PAUSE: self._backend.pause,
            MusicActionType.NEXT: self._backend.next,
            MusicActionType.PREVIOUS: self._backend.previous,
            MusicActionType.GET_NOW_PLAYING: self._backend.now_playing,
        }
        operation = operations.get(kind)
        if operation is None:
            raise MusicControlError("unsupported_action", "unsupported music action")
        return {"now_playing": _item_data(operation())}

    def _playlists(self) -> PlaylistSnapshot:
        now = self._clock()
        if self._playlist_cache and now - self._playlist_cache[0] < self._cache_seconds:
            return self._playlist_cache[1]
        try:
            snapshot = self._backend.list_playlists()
        except (MusicControlError, OSError, TimeoutError):
            if not self._playlist_cache:
                raise
            stale = self._playlist_cache[1]
            return PlaylistSnapshot(
                stale.items, partial=True, warning="playlist_refresh_failed",
            )
        self._playlist_cache = (now, snapshot)
        return snapshot

    def _personal_music(self) -> PersonalMusicSnapshot:
        now = self._clock()
        if self._personal_cache and now - self._personal_cache[0] < self._personal_cache_seconds:
            return self._personal_cache[1]
        try:
            snapshot = self._backend.personal_songs()
        except (MusicControlError, OSError, TimeoutError):
            if self._personal_cache:
                return self._personal_cache[1]
            return PersonalMusicSnapshot((), partial=True, warning="personal_index_unavailable")
        self._personal_cache = (now, snapshot)
        return snapshot

    def _playlist(self, name: str, alternates: tuple[str, ...] = ()) -> PlaylistItem:
        names = tuple(dict.fromkeys((name, *alternates)))
        library = self._playlists().items
        wanted = {_normalize(query) for query in names}
        library_matches = [item for item in library if _normalize(item.name) in wanted]
        if len(library_matches) > 1:
            raise MusicControlError("ambiguous", "multiple library playlists matched")
        if library_matches:
            return library_matches[0]
        membership_matches = self._playlist_membership_matches(library, names)
        if len(membership_matches) == 1:
            return membership_matches[0]
        if len(membership_matches) > 1:
            raise MusicControlError(
                "ambiguous", "multiple personal playlists matched",
                candidate_options=tuple(item.name for item in membership_matches),
            )
        raise MusicControlError("not_found", "personal playlist could not be resolved")

    def _playlist_membership_matches(
        self, library: tuple[PlaylistItem, ...], names: tuple[str, ...],
    ) -> list[PlaylistItem]:
        variants = tuple(dict.fromkeys(
            variant for name in names for variant in _entity_query_variants(name)
        ))
        search = _merge_search_results(
            self._backend.search_songs(query) for query in variants
        )
        leading_artists = [_normalize(item.artist) for item in search[:5] if item.artist]
        if not leading_artists:
            return []
        artist = max(set(leading_artists), key=leading_artists.count)
        if leading_artists.count(artist) < min(4, len(leading_artists)):
            return []
        matches = []
        read_tracks = getattr(self._backend, "playlist_tracks", None)
        if read_tracks is None:
            return []
        for playlist in library:
            try:
                tracks = read_tracks(playlist.playlist_id)
            except (MusicControlError, OSError, TimeoutError):
                continue
            same_artist = sum(_normalize(track.artist) == artist for track in tracks)
            if same_artist >= 2 and same_artist / len(tracks) >= 0.6:
                matches.append(playlist)
        return matches

    def _resolve_artist(self, query: str, alternates: tuple[str, ...] = ()) -> str:
        queries = tuple(dict.fromkeys((query, *alternates)))
        candidates = _merge_named_results(
            self._backend.search_artists(item) for item in queries
        )
        wanted = {_normalize(item) for item in queries}
        exact = [item for item in candidates if _normalize(item.artist) in wanted]
        if len(exact) == 1:
            return exact[0].artist
        if len(exact) > 1:
            raise MusicControlError("ambiguous", "artist could not be resolved uniquely")
        songs = sorted(_merge_search_results(
            self._backend.search_songs(item) for item in queries
        ), key=lambda item: item.search_rank)
        leading_artists = {_normalize(item.artist) for item in songs[:5] if item.artist}
        if len(leading_artists) == 1 and songs:
            return songs[0].artist
        raise MusicControlError("ambiguous", "artist could not be resolved uniquely")


class MusicControlError(RuntimeError):
    def __init__(
        self, code: str, message: str, *, candidate_options: tuple[str, ...] = (),
    ) -> None:
        super().__init__(message)
        self.code = code
        self.candidate_options = candidate_options


def _normalize(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).casefold()
    return re.sub(r"[^0-9a-z가-힣ぁ-んァ-ヶ一-龯]+", "", value)


def _unique_match(items, query: str, value_getter):
    wanted = _normalize(query)
    matches = [item for item in items if _normalize(value_getter(item)) == wanted]
    if not matches:
        raise MusicControlError("not_found", "no matching Apple Music item")
    if len(matches) > 1:
        raise MusicControlError("ambiguous", "multiple Apple Music items matched")
    return matches[0]


def _unique_music_match(
    items: tuple[MusicItem, ...], title: str, artist: str,
    alternate_queries: tuple[str, ...] = (),
) -> tuple[MusicItem, int]:
    title_key = _normalize(title)
    artist_key = _normalize(artist)
    query_forms = {_normalize(query) for query in alternate_queries}
    matches = [
        (item, index) for index, item in enumerate(items)
        if (
            _normalize(item.title) == title_key
            and (not artist_key or _normalize(item.artist) == artist_key)
        ) or _normalize(item.artist + " " + item.title) in query_forms
    ]
    if not matches:
        raise MusicControlError("not_found", "no matching song")
    groups: dict[tuple[str, str], list[tuple[MusicItem, int]]] = {}
    for item, index in matches:
        recording_key = _normalize(item.recording_id) or f"catalog:{item.item_id}"
        groups.setdefault((_normalize(item.artist), recording_key), []).append((item, index))
    if len(groups) == 1:
        return min(matches, key=lambda match: match[0].search_rank)
    ranked_groups = sorted(
        (
            min(group, key=lambda match: match[0].search_rank)
            for group in groups.values()
        ),
        key=lambda match: match[0].search_rank,
    )
    leader, runner_up = ranked_groups[:2]
    if (
        leader[0].search_rank == 0
        and runner_up[0].search_rank - leader[0].search_rank
        >= _MIN_RELEVANCE_RANK_GAP
    ):
        return leader
    raise MusicControlError("ambiguous", "multiple songs matched")


def _deterministic_track_matches(
    action: MusicAction, candidates: tuple[MusicItem, ...],
) -> tuple[MusicItem, ...]:
    """Return canonical recordings supported by explicit lexical evidence."""
    requested_title = _normalize(action.title)
    requested_artist = _normalize(action.artist)
    groups: dict[tuple[str, str], MusicItem] = {}
    for item in candidates:
        if not requested_title or _normalize(item.title) != requested_title:
            continue
        if requested_artist and _normalize(item.artist) != requested_artist:
            continue
        recording_key = _normalize(item.recording_id) or f"catalog:{item.item_id}"
        key = (_normalize(item.artist), recording_key)
        current = groups.get(key)
        if current is None or item.search_rank < current.search_rank:
            groups[key] = item
    return tuple(sorted(groups.values(), key=lambda item: item.search_rank))


def _entity_query_variants(value: str) -> tuple[str, ...]:
    value = value.strip()
    return (value,) if value else ()


def _merge_search_results(
    result_sets,
) -> tuple[MusicItem, ...]:
    by_id: dict[str, MusicItem] = {}
    for results in result_sets:
        for item in results:
            current = by_id.get(item.item_id)
            if current is None or item.search_rank < current.search_rank:
                by_id[item.item_id] = item
    return tuple(sorted(by_id.values(), key=lambda item: item.search_rank))


def _merge_named_results(result_sets):
    by_id = {}
    for results in result_sets:
        for item in results:
            key = getattr(item, "item_id", "") or getattr(item, "playlist_id", "")
            if not key:
                key = _normalize(getattr(item, "artist", "") or getattr(item, "name", ""))
            by_id.setdefault(key, item)
    return tuple(by_id.values())


def _verified_data(actual: MusicItem, expected: MusicItem) -> dict[str, object]:
    if _normalize(actual.title) != _normalize(expected.title) or (
        expected.artist and _normalize(actual.artist) != _normalize(expected.artist)
    ):
        raise MusicControlError("metadata_mismatch", "now-playing metadata did not match")
    return {"now_playing": _item_data(actual)}


def _item_data(item: MusicItem) -> dict[str, str]:
    return {
        "id": item.item_id, "title": item.title,
        "artist": item.artist, "album": item.album,
        **({"recording_id": item.recording_id} if item.recording_id else {}),
    }
