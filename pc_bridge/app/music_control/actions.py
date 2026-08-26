from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol


class MusicActionType(str, Enum):
    PLAY_SONG = "play_song"
    PLAY_ARTIST = "play_artist"
    PLAY_PLAYLIST = "play_playlist"
    PLAY_PLAYLIST_TRACK = "play_playlist_track"
    LIST_PLAYLISTS = "list_playlists"
    PLAY = "play"
    PAUSE = "pause"
    NEXT = "next"
    PREVIOUS = "previous"
    GET_NOW_PLAYING = "get_now_playing"


@dataclass(frozen=True)
class MusicAction:
    action_type: MusicActionType
    title: str = ""
    artist: str = ""
    playlist: str = ""
    source_query: str = field(default="", compare=False, repr=False)
    alternate_queries: tuple[str, ...] = field(default=(), compare=False, repr=False)
    artist_explicit: bool = field(default=False, compare=False, repr=False)


@dataclass(frozen=True)
class MusicActionSequence:
    actions: tuple[MusicAction, ...]

    def __post_init__(self) -> None:
        if not self.actions:
            raise ValueError("music action sequence cannot be empty")


@dataclass(frozen=True)
class MusicActionParseResult:
    sequence: MusicActionSequence | None = None
    error_code: str = ""

    @property
    def ok(self) -> bool:
        return self.sequence is not None and not self.error_code

    @property
    def actions(self) -> tuple[MusicAction, ...]:
        return self.sequence.actions if self.sequence else ()

    @property
    def action(self) -> MusicAction | None:
        return self.actions[0] if len(self.actions) == 1 else None


@dataclass(frozen=True)
class MusicActionResult:
    action: MusicAction
    ok: bool
    data: dict[str, object]
    error: str = ""


@dataclass(frozen=True)
class MusicActionSequenceResult:
    sequence: MusicActionSequence
    results: tuple[MusicActionResult, ...]

    @property
    def ok(self) -> bool:
        return all(result.ok for result in self.results)

    @property
    def status(self) -> str:
        succeeded = sum(result.ok for result in self.results)
        if succeeded == len(self.results):
            return "success"
        return "partial_failure" if succeeded else "failure"


class MusicActionParser(Protocol):
    def parse(self, user_text: str) -> MusicActionParseResult: ...


class MusicController(Protocol):
    def execute(self, action: MusicAction) -> MusicActionResult: ...
