from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
import re
import time
import unicodedata
from typing import Protocol

from .actions import MusicAction, MusicActionResult, MusicActionType


_MIN_RELEVANCE_RANK_GAP = 4
_MIN_SONG_SCORE = 80.0
_MIN_SONG_SCORE_MARGIN = 8.0


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
    def play_queue_item(self, index: int) -> MusicItem: ...
    def play(self) -> MusicItem: ...
    def pause(self) -> MusicItem: ...
    def next(self) -> MusicItem: ...
    def previous(self) -> MusicItem: ...
    def now_playing(self) -> MusicItem: ...


class AppleMusicPwaController:
    def __init__(
        self,
        backend: AppleMusicBackend,
        *,
        playlist_cache_seconds: float = 60.0,
        personal_music_cache_seconds: float = 300.0,
        health_cache_seconds: float = 30.0,
        clock=time.monotonic,
    ) -> None:
        self._backend = backend
        self._cache_seconds = playlist_cache_seconds
        self._health_cache_seconds = health_cache_seconds
        self._personal_cache_seconds = personal_music_cache_seconds
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
            return MusicActionResult(action, False, {"reason": exc.code}, str(exc))
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
            playlist = self._playlist(action.playlist)
            queue = self._backend.load_playlist(playlist.playlist_id)
            if not queue:
                raise MusicControlError("not_found", "playlist is empty")
            actual = self._backend.play_queue_item(0)
            return _verified_data(actual, queue[0]) | {
                "playlist": playlist.name, "queue_length": len(queue),
            }
        if kind is MusicActionType.PLAY_PLAYLIST_TRACK:
            playlist = self._playlist(action.playlist)
            queue = self._backend.load_playlist(playlist.playlist_id)
            expected, index = _unique_music_match(queue, action.title, action.artist)
            actual = self._backend.play_queue_item(index)
            return _verified_data(actual, expected) | {
                "playlist": playlist.name, "queue_position": index,
                "queue_length": len(queue),
            }
        if kind is MusicActionType.PLAY_SONG:
            query = action.source_query or " ".join(filter(None, (
                action.artist, action.title,
            )))
            queries = tuple(dict.fromkeys((query, *action.alternate_queries)))
            candidates = _merge_search_results(
                self._backend.search_songs(candidate_query)
                for candidate_query in queries if candidate_query
            )
            title_candidates = (
                self._backend.search_songs(action.title) if action.artist else ()
            )
            personal = self._personal_music()
            expected = _resolve_song_candidate(
                candidates, title_candidates, action, personal.items,
            )
            return _verified_data(self._backend.play_song(expected.item_id), expected)
        if kind is MusicActionType.PLAY_ARTIST:
            artist = self._resolve_artist(action.artist)
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

    def _playlist(self, name: str) -> PlaylistItem:
        library = self._playlists().items
        library_matches = [item for item in library if _normalize(item.name) == _normalize(name)]
        if len(library_matches) > 1:
            raise MusicControlError("ambiguous", "multiple library playlists matched")
        if library_matches:
            return library_matches[0]
        return _unique_match(
            self._backend.search_playlists(name), name, lambda item: item.name,
        )

    def _resolve_artist(self, query: str) -> str:
        candidates = self._backend.search_artists(query)
        exact = [item for item in candidates if _normalize(item.artist) == _normalize(query)]
        if len(exact) == 1:
            return exact[0].artist
        if len(exact) > 1:
            raise MusicControlError("ambiguous", "artist could not be resolved uniquely")
        songs = sorted(self._backend.search_songs(query), key=lambda item: item.search_rank)
        leading_artists = {_normalize(item.artist) for item in songs[:5] if item.artist}
        if len(leading_artists) == 1 and songs:
            return songs[0].artist
        raise MusicControlError("ambiguous", "artist could not be resolved uniquely")


class MusicControlError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


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
) -> tuple[MusicItem, int]:
    title_key = _normalize(title)
    artist_key = _normalize(artist)
    matches = [
        (item, index) for index, item in enumerate(items)
        if _normalize(item.title) == title_key
        and (not artist_key or _normalize(item.artist) == artist_key)
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


def _resolve_song_candidate(
    candidates: tuple[MusicItem, ...],
    title_candidates: tuple[MusicItem, ...],
    action: MusicAction,
    personal_items: tuple[PersonalMusicItem, ...] = (),
) -> MusicItem:
    personal_by_id = {entry.item.item_id: entry for entry in personal_items}
    candidate_ids = {item.item_id for item in candidates}
    personal_candidates = tuple(
        entry.item for entry in personal_items if entry.item.item_id not in candidate_ids
    )
    ranked = sorted(candidates + personal_candidates, key=lambda item: item.search_rank)
    if not ranked:
        raise MusicControlError("no_match", "no reasonable song candidate")
    groups: dict[tuple[str, str], MusicItem] = {}
    for item in ranked:
        recording_key = _normalize(item.recording_id) or f"catalog:{item.item_id}"
        key = (_normalize(item.artist), recording_key)
        current = groups.get(key)
        if current is None or item.search_rank < current.search_rank:
            groups[key] = item

    requested_title = _normalize(action.title)
    requested_artist = _normalize(action.artist)
    source_query = _normalize(action.source_query or " ".join(filter(None, (
        action.artist, action.title,
    ))))
    query_forms = tuple(filter(None, (
        source_query, *(_normalize(query) for query in action.alternate_queries),
    )))
    leader = ranked[0]
    exact_requested = [
        item for item in ranked if _normalize(item.title) == requested_title
    ]
    ranked_alias = (
        leader.search_rank == 0
        and _is_self_titled_release(leader)
        and _normalize(leader.title) != requested_title
        and (not exact_requested or exact_requested[0].search_rank >= 2)
    )
    title_leader_id = (
        min(title_candidates, key=lambda item: item.search_rank).item_id
        if title_candidates else ""
    )
    top_artists = [_normalize(item.artist) for item in ranked[:5] if item.artist]
    dominant_leader_artist = bool(top_artists) and (
        len(set(top_artists)) == 1
        or top_artists.count(_normalize(leader.artist)) >= 4
    )

    scored: list[tuple[float, MusicItem]] = []
    for item in groups.values():
        title = _normalize(item.title)
        artist = _normalize(item.artist)
        combined = {artist + title, title + artist}
        title_exact = bool(requested_title) and title == requested_title
        combined_exact = any(query in combined for query in query_forms)
        full_title_exact = any(title == query for query in query_forms)
        artist_exact = bool(requested_artist) and artist == requested_artist
        artist_in_query = bool(artist) and any(artist in query for query in query_forms)
        title_in_query = bool(title) and any(title in query for query in query_forms)
        alias_corroborated = (
            item.item_id == leader.item_id
            and item.item_id == title_leader_id
            and dominant_leader_artist
        )
        cross_script_leader = (
            item.item_id == leader.item_id
            and item.search_rank == 0
            and _is_self_titled_release(item)
            and dominant_leader_artist
            and (artist_exact or alias_corroborated)
        )
        rank_alias_leader = item.item_id == leader.item_id and ranked_alias

        similarity = max(
            *(_similarity(title, query) for query in query_forms),
            _similarity(title, requested_title),
        )
        if combined_exact:
            title_score = 105.0
        elif full_title_exact:
            title_score = 100.0
        elif title_exact:
            title_score = 90.0
        elif title_in_query and artist_in_query:
            title_score = 95.0
        elif similarity >= 0.92:
            title_score = 75.0
        elif similarity >= 0.82:
            title_score = 55.0
        elif rank_alias_leader:
            title_score = 105.0
        elif cross_script_leader:
            title_score = 65.0
        elif alias_corroborated:
            title_score = 75.0
        else:
            continue

        artist_hint_conflicts = (
            bool(requested_artist)
            and not artist_exact
            and not alias_corroborated
            and not full_title_exact
        )
        if artist_hint_conflicts:
            continue
        if action.artist_explicit and requested_artist and not (
            artist_exact or alias_corroborated
        ):
            continue

        artist_score = 30.0 if artist_exact else (20.0 if artist_in_query else 0.0)
        rank_score = max(0.0, 20.0 - 3.0 * item.search_rank)
        release_score = 5.0 if _is_self_titled_release(item) else 0.0
        personal = personal_by_id.get(item.item_id)
        personal_score = 0.0
        if personal:
            personal_score += 25.0 if personal.in_library else 0.0
            if personal.playlist_count:
                personal_score += min(30.0, 15.0 + 5.0 * personal.playlist_count)
        scored.append((
            title_score + artist_score + rank_score + release_score + personal_score,
            item,
        ))

    if not scored:
        raise MusicControlError("no_match", "no reasonable song candidate")
    scored.sort(key=lambda pair: (-pair[0], pair[1].search_rank))
    best_score, best = scored[0]
    if best_score < _MIN_SONG_SCORE:
        raise MusicControlError("no_match", "no reasonable song candidate")
    if len(scored) > 1 and best_score - scored[1][0] <= _MIN_SONG_SCORE_MARGIN:
        raise MusicControlError("ambiguous", "multiple songs matched")
    return best


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


def _similarity(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    return SequenceMatcher(None, left, right).ratio()


def _is_self_titled_release(item: MusicItem) -> bool:
    title = _normalize(item.title)
    return bool(title) and _normalize(item.album).startswith(title)


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
