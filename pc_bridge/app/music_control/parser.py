from __future__ import annotations

import re

from .actions import (
    MusicAction, MusicActionParseResult, MusicActionSequence, MusicActionType,
)


_PLAY_COMMAND = r"(?:틀어(?:줘|줄래)?|재생(?:해줘|해줄래|해)?|들려(?:줘|줄래)?)"
_LEADING_CONVERSATIONAL_FILLER = re.compile(
    r"^(?:(?:안녕(?:하세요)?|야|저기|있잖아|잠깐만)(?:[,.!?~]+|\s+))+",
    re.IGNORECASE,
)
_SEQUENCE_CONNECTOR = re.compile(
    r"\s*(?:그리고|한\s*다음(?:에)?|(?<=일시정지)하고|(?<=재생)하고|"
    r"(?<=해줘)\s*하고|(?<=틀)고|(?<=재생하)고|(?<=일시정지하)고|"
    r"(?<=멈추)고|(?<=넘기)고)\s*(?:바로\s+)?",
)


class RuleBasedMusicActionParser:
    """Parse the representative commands supported before semantic ML routing."""

    def parse(self, user_text: str) -> MusicActionParseResult:
        text = self._normalize_input(user_text)
        clauses = tuple(filter(None, _SEQUENCE_CONNECTOR.split(text)))
        if not clauses:
            return MusicActionParseResult(error_code="unsupported_action")
        actions: list[MusicAction] = []
        for clause in clauses:
            parsed = self._parse_single(_complete_linked_clause(clause.strip()))
            if parsed is None:
                return MusicActionParseResult(error_code="unsupported_action")
            actions.append(parsed)
        return MusicActionParseResult(MusicActionSequence(tuple(actions)))

    @staticmethod
    def _normalize_input(user_text: str) -> str:
        text = " ".join(user_text.strip().split())
        text = re.split(
            r"(?:알려\s*주고|확인\s*하고|볼륨\s*[^,]+?하고|소리\s*[^,]+?하고)",
            text,
        )[-1].strip()
        text = _LEADING_CONVERSATIONAL_FILLER.sub("", text).strip()
        text = re.sub(r"^크리스(?:야|아)?[,.!?~]*\s+", "", text).strip()
        text = re.sub(r"\s+좀\s+(?=(?:틀어|재생))", " ", text).strip()
        return text.rstrip(" ?!.,~")

    @staticmethod
    def _parse_single(text: str) -> MusicAction | None:
        compact = "".join(text.lower().split())
        if compact in {"재생", "재생해", "재생해줘"}:
            return MusicAction(MusicActionType.PLAY)

        fixed = (
            (("내플레이리스트뭐있", "플레이리스트목록"), MusicActionType.LIST_PLAYLISTS),
            (("지금무슨곡", "지금무슨노래", "지금뭐재생", "현재재생곡"),
             MusicActionType.GET_NOW_PLAYING),
            (("일시정지", "잠깐멈춰", "재생멈춰"), MusicActionType.PAUSE),
            (("다시재생", "계속재생"), MusicActionType.PLAY),
            (("다음곡", "다음노래", "한곡넘겨"), MusicActionType.NEXT),
            (("이전곡", "이전노래", "아까노래"), MusicActionType.PREVIOUS),
        )
        for phrases, action_type in fixed:
            if any(phrase in compact for phrase in phrases):
                return MusicAction(action_type)

        arbitrary_playlist = re.search(
            r"(?:내\s*)?(?P<name>.+?)\s*플레이리스트(?:에서)?\s*"
            rf"아무거나(?:\s*한\s*곡)?\s*{_PLAY_COMMAND}$",
            text,
        )
        if arbitrary_playlist:
            return MusicAction(
                MusicActionType.PLAY_PLAYLIST,
                playlist=arbitrary_playlist.group("name").strip(),
            )

        playlist_track = re.search(
            r"(?:내\s*)?(?:플레이리스트\s*)?(?P<playlist>.+?)에서\s*"
            rf"(?:(?P<artist>\S+)\s+)?(?P<title>.+?)\s*{_PLAY_COMMAND}",
            text,
        )
        if playlist_track:
            return MusicAction(
                MusicActionType.PLAY_PLAYLIST_TRACK,
                title=playlist_track.group("title").strip(),
                artist=(playlist_track.group("artist") or "").strip(),
                playlist=playlist_track.group("playlist").strip(),
            )

        playlist = re.search(
            r"(?:내\s*)?(?:플레이리스트\s*)?(?P<name>.+?)(?:\s*플레이리스트)?\s*"
            rf"{_PLAY_COMMAND}$",
            text,
        )
        if playlist and ("플레이리스트" in text or "대표곡" in text):
            name = re.sub(r"^내\s*", "", playlist.group("name").strip()).strip()
            return MusicAction(MusicActionType.PLAY_PLAYLIST, playlist=name)

        artist_song = re.search(
            rf"(?P<artist>.+?)(?:의|가\s*부른)\s+(?P<title>.+?)\s*{_PLAY_COMMAND}$",
            text,
        )
        if artist_song:
            artist = artist_song.group("artist").strip()
            title = artist_song.group("title").strip()
            return MusicAction(
                MusicActionType.PLAY_SONG,
                title=title,
                artist=artist,
                source_query=f"{artist} {title}",
                artist_explicit=True,
            )

        named_song = re.search(
            rf"(?P<title>.+?)(?:이라는|이란)\s*노래\s*{_PLAY_COMMAND}$",
            text,
        )
        if named_song:
            title = named_song.group("title").strip()
            return MusicAction(
                MusicActionType.PLAY_SONG, title=title, source_query=title,
            )

        arbitrary_artist = re.search(
            rf"(?P<artist>.+?)\s+(?:노래|곡|음악)(?:\s+아무거나(?:\s*한\s*곡)?)?\s*"
            rf"{_PLAY_COMMAND}$",
            text,
        )
        if arbitrary_artist:
            return MusicAction(
                MusicActionType.PLAY_ARTIST,
                artist=arbitrary_artist.group("artist").strip(),
            )

        song = re.search(
            rf"(?:(?P<artist>\S+)\s+)?(?P<title>.+?)\s*{_PLAY_COMMAND}$", text,
        )
        if not song:
            return None
        title = song.group("title").strip()
        artist = (song.group("artist") or "").strip()
        if title.endswith("노래"):
            return MusicAction(
                MusicActionType.PLAY_ARTIST, artist=artist or title[:-2].strip(),
            )
        return MusicAction(
            MusicActionType.PLAY_SONG,
            title=title,
            artist=artist,
            source_query=" ".join(filter(None, (artist, title))),
        )


def _complete_linked_clause(clause: str) -> str:
    completions = {
        "틀": "틀어줘",
        "재생하": "재생해줘",
        "일시정지하": "일시정지해줘",
        "멈추": "멈춰",
        "넘기": "넘겨줘",
    }
    for stem, command in completions.items():
        if clause.endswith(stem):
            return clause[:-len(stem)] + command
    return clause
