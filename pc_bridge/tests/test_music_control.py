from __future__ import annotations

from dataclasses import replace
import json
from websocket import WebSocketTimeoutException

import pytest

from app.music_control import (
    AppleMusicPwaController,
    MusicControlError,
    MusicAction,
    MusicActionType,
    MusicItem,
    PersonalMusicItem,
    PersonalMusicSnapshot,
    PlaylistItem,
    PlaylistSnapshot,
    RuleBasedMusicActionParser,
    CdpAppleMusicBackend,
)
from app.tools import MusicControlTool
from app.music_control.controller import CatalogQueryVariant, SongCandidateJudgment


class FakeAppleMusicBackend:
    def __init__(self) -> None:
        self.playlists = PlaylistSnapshot((
            PlaylistItem("p.study", "공부할 때"),
            PlaylistItem("pl.aimyon", "아이묭 대표곡"),
        ))
        self.catalog_playlists = (PlaylistItem("pl.aimyon", "아이묭 대표곡"),)
        self.songs = (
            MusicItem("song.marigold", "Marigold", "aimyon", "Marigold - Single"),
        )
        self.queue = self.songs + (
            MusicItem("song.next", "Kimi Wa Rock Wo Kikanai", "aimyon"),
        )
        self.current = self.songs[0]
        self.position = 0
        self.list_calls = 0
        self.personal = PersonalMusicSnapshot(())
        self.personal_calls = 0

    def selector_health(self):
        return {"authorized": True, "player": True}

    def search_songs(self, query):
        return self.songs

    def search_artists(self, query):
        return (MusicItem("artist.aimyon", "", "aimyon"),)

    def search_playlists(self, query):
        return self.catalog_playlists

    def list_playlists(self):
        self.list_calls += 1
        return self.playlists

    def personal_songs(self):
        self.personal_calls += 1
        return self.personal

    def play_song(self, item_id):
        self.current = next(item for item in self.songs if item.item_id == item_id)
        return self.current

    def play_artist(self, item_id):
        return self.current

    def load_playlist(self, playlist_id):
        return self.queue

    def playlist_tracks(self, playlist_id):
        return self.queue

    def play_queue_item(self, index):
        self.position = index
        self.current = self.queue[index]
        return self.current

    def play(self):
        return self.current

    def pause(self):
        return self.current

    def next(self):
        self.position += 1
        self.current = self.queue[self.position]
        return self.current

    def previous(self):
        self.position -= 1
        self.current = self.queue[self.position]
        return self.current

    def now_playing(self):
        return self.current


class FakeCandidateSemantics:
    def __init__(
        self, *, song_index=None, playlist_index=None, rewrites=(),
        title_equivalent=True, artist_equivalent=True,
    ):
        self.song_index = song_index
        self.playlist_index = playlist_index
        self.rewrites = rewrites
        self.title_equivalent = title_equivalent
        self.artist_equivalent = artist_equivalent
        self.song_calls = []
        self.playlist_calls = []
        self.rewrite_calls = []

    def judge_song_candidate(self, action, candidates):
        self.song_calls.append((action, candidates))
        if self.song_index is None:
            return SongCandidateJudgment("none", rejection_reason="no_match")
        return SongCandidateJudgment(
            "match", self.song_index, self.title_equivalent, self.artist_equivalent,
            rejection_reason=(
                "title_not_equivalent" if not self.title_equivalent
                else "artist_not_equivalent" if not self.artist_equivalent else ""
            ),
        )

    def judge_playlist_candidate(self, surface, candidates):
        self.playlist_calls.append((surface, candidates))
        return self.playlist_index

    def rewrite_track_queries(self, artist_surface, title_surface):
        self.rewrite_calls.append((artist_surface, title_surface))
        return self.rewrites

class FakeWebSocket:
    def __init__(self, response=None, error=None) -> None:
        self.response = response
        self.error = error
        self.sent = []
        self.closed = False

    def send(self, message) -> None:
        self.sent.append(json.loads(message))

    def recv(self):
        if self.error:
            raise self.error
        return json.dumps(self.response)

    def close(self) -> None:
        self.closed = True


@pytest.mark.parametrize(("text", "expected"), [
    ("아이묭 마리골드 틀어줘", MusicAction(
        MusicActionType.PLAY_SONG, title="마리골드", artist="아이묭",
    )),
    ("아이묭 노래 틀어줘", MusicAction(MusicActionType.PLAY_ARTIST, artist="아이묭")),
    ("내 플레이리스트 공부할 때 틀어줘", MusicAction(
        MusicActionType.PLAY_PLAYLIST, playlist="공부할 때",
    )),
    ("아이묭 대표곡에서 마리골드 틀어줘", MusicAction(
        MusicActionType.PLAY_PLAYLIST_TRACK,
        playlist="아이묭 대표곡", title="마리골드",
    )),
    ("내 플레이리스트 뭐 있어?", MusicAction(MusicActionType.LIST_PLAYLISTS)),
    ("일시정지해줘", MusicAction(MusicActionType.PAUSE)),
    ("다음 곡 틀어줘", MusicAction(MusicActionType.NEXT)),
    ("이전 곡", MusicAction(MusicActionType.PREVIOUS)),
    ("지금 무슨 곡이야?", MusicAction(MusicActionType.GET_NOW_PLAYING)),
    ("지금 무슨 노래야?", MusicAction(MusicActionType.GET_NOW_PLAYING)),
    ("일시정지 좀 해줘", MusicAction(MusicActionType.PAUSE)),
    ("다시 재생해줘", MusicAction(MusicActionType.PLAY)),
    ("안녕 크리스 백넘버의 수평선 재생해줄래?", MusicAction(
        MusicActionType.PLAY_SONG, title="수평선", artist="백넘버",
    )),
    ("백넘버 수평선 틀어줘", MusicAction(
        MusicActionType.PLAY_SONG, title="수평선", artist="백넘버",
    )),
    ("백넘버가 부른 수평선 틀어줘", MusicAction(
        MusicActionType.PLAY_SONG, title="수평선", artist="백넘버",
    )),
    ("수평선이라는 노래 틀어줘", MusicAction(
        MusicActionType.PLAY_SONG, title="수평선",
    )),
    ("야 백넘버 노래 아무거나 틀어줘", MusicAction(
        MusicActionType.PLAY_ARTIST, artist="백넘버",
    )),
    ("저기 The Beatles 곡 아무거나 재생해줘", MusicAction(
        MusicActionType.PLAY_ARTIST, artist="The Beatles",
    )),
])
def test_parser_extracts_representative_music_actions(text, expected) -> None:
    assert RuleBasedMusicActionParser().parse(text).action == expected


def test_parser_ignores_minimal_conversational_song_wrappers() -> None:
    assert RuleBasedMusicActionParser().parse(
        "안녕 크리스 마리골드 좀 틀어줘",
    ).action == MusicAction(MusicActionType.PLAY_SONG, title="마리골드")


def test_parser_preserves_full_song_query_without_changing_structured_hints() -> None:
    action = RuleBasedMusicActionParser().parse("Color Your Night 틀어줘").action

    assert action == MusicAction(
        MusicActionType.PLAY_SONG, title="Your Night", artist="Color",
    )
    assert action.source_query == "Color Your Night"
    assert not action.artist_explicit


def test_parser_treats_any_playlist_track_as_deterministic_playlist_start() -> None:
    parser = RuleBasedMusicActionParser()
    expected = MusicAction(MusicActionType.PLAY_PLAYLIST, playlist="굿나잇")

    assert parser.parse("굿나잇 플레이리스트 아무거나 재생해줘").action == expected
    assert parser.parse("굿나잇 플레이리스트에서 아무거나 한 곡 재생해줘").action == expected


def test_controller_uses_verified_ids_and_preserves_playlist_queue() -> None:
    backend = FakeAppleMusicBackend()
    controller = AppleMusicPwaController(backend)

    song = controller.execute(MusicAction(
        MusicActionType.PLAY_SONG, title="Marigold", artist="aimyon",
    ))
    playlist = controller.execute(MusicAction(
        MusicActionType.PLAY_PLAYLIST, playlist="공부할 때",
    ))
    track = controller.execute(MusicAction(
        MusicActionType.PLAY_PLAYLIST_TRACK,
        playlist="아이묭 대표곡", title="Marigold", artist="aimyon",
    ))
    following = controller.execute(MusicAction(MusicActionType.NEXT))

    assert song.ok and song.data["now_playing"]["id"] == "song.marigold"
    assert playlist.ok and playlist.data["queue_length"] == 2
    assert track.ok and track.data["queue_position"] == 0
    assert following.data["now_playing"]["id"] == "song.next"


def test_controller_refuses_ambiguous_and_metadata_mismatch() -> None:
    backend = FakeAppleMusicBackend()
    backend.songs = backend.songs + (replace(backend.songs[0], item_id="duplicate"),)
    controller = AppleMusicPwaController(backend)

    ambiguous = controller.execute(MusicAction(
        MusicActionType.PLAY_SONG, title="Marigold", artist="aimyon",
    ))
    assert not ambiguous.ok
    assert ambiguous.data["reason"] == "ambiguous"

    backend.songs = (MusicItem("wanted", "Marigold", "aimyon"),)
    backend.current = MusicItem("wrong", "Different Song", "aimyon")
    backend.play_song = lambda item_id: backend.current
    mismatch = controller.execute(MusicAction(
        MusicActionType.PLAY_SONG, title="Marigold", artist="aimyon",
    ))
    assert not mismatch.ok
    assert mismatch.data["reason"] == "metadata_mismatch"


def test_controller_deduplicates_catalog_entries_only_for_the_same_isrc() -> None:
    backend = FakeAppleMusicBackend()
    recording = MusicItem(
        "single", "Marigold", "aimyon", "Marigold - Single", "JPWP01870687",
    )
    backend.songs = (
        recording,
        replace(recording, item_id="album", album="Momentary Sixth Sense"),
    )
    backend.current = recording

    result = AppleMusicPwaController(backend).execute(MusicAction(
        MusicActionType.PLAY_SONG, title="Marigold", artist="aimyon",
    ))

    assert result.ok
    assert result.data["now_playing"]["id"] == "single"


def test_title_only_namesakes_are_not_resolved_by_rank() -> None:
    backend = FakeAppleMusicBackend()
    original = MusicItem(
        "single", "Marigold", "aimyon", "Marigold - Single",
        "JPWP01870687", 0,
    )
    backend.songs = (
        original,
        replace(original, item_id="album", search_rank=3),
        MusicItem("cover", "Marigold", "Other Artist", "Cover", "OTHER", 4),
    )
    backend.current = original

    result = AppleMusicPwaController(backend).execute(MusicAction(
        MusicActionType.PLAY_SONG, title="Marigold",
    ))

    assert not result.ok
    assert result.data["reason"] == "ambiguous"


def test_title_only_remains_ambiguous_when_artists_compete_at_top_rank() -> None:
    backend = FakeAppleMusicBackend()
    backend.songs = (
        MusicItem("first", "Same Title", "First Artist", "One", "FIRST", 0),
        MusicItem("second", "Same Title", "Second Artist", "Two", "SECOND", 1),
    )

    result = AppleMusicPwaController(backend).execute(MusicAction(
        MusicActionType.PLAY_SONG, title="Same Title",
    ))

    assert not result.ok
    assert result.data["reason"] == "ambiguous"


def test_self_titled_top_result_does_not_hide_close_namesake() -> None:
    backend = FakeAppleMusicBackend()
    backend.songs = (
        MusicItem(
            "first", "Same Title", "First Artist", "Same Title - Single",
            "FIRST", 0,
        ),
        MusicItem("second", "Same Title", "Second Artist", "Album", "SECOND", 1),
    )

    result = AppleMusicPwaController(backend).execute(MusicAction(
        MusicActionType.PLAY_SONG, title="Same Title",
    ))

    assert not result.ok
    assert result.data["reason"] == "ambiguous"


def test_personal_membership_does_not_override_title_only_ambiguity() -> None:
    backend = FakeAppleMusicBackend()
    catalog_leader = MusicItem(
        "first", "Same Title", "First Artist", "Same Title - Single", "FIRST", 0,
    )
    personal_choice = MusicItem(
        "second", "Same Title", "Second Artist", "Album", "SECOND", 1,
    )
    backend.songs = (catalog_leader, personal_choice)
    backend.personal = PersonalMusicSnapshot((
        PersonalMusicItem(personal_choice, playlist_count=2),
    ))
    backend.current = personal_choice

    result = AppleMusicPwaController(backend).execute(MusicAction(
        MusicActionType.PLAY_SONG, title="Same Title",
    ))

    assert not result.ok
    assert result.data["reason"] == "ambiguous"


def test_personal_index_is_account_wide_cached_and_not_required_for_resolution() -> None:
    backend = FakeAppleMusicBackend()
    controller = AppleMusicPwaController(backend, personal_music_cache_seconds=60)

    assert controller.execute(MusicAction(
        MusicActionType.PLAY_SONG, title="Marigold",
    )).ok
    assert controller.execute(MusicAction(
        MusicActionType.PLAY_SONG, title="Marigold",
    )).ok
    assert backend.personal_calls == 1

    backend.personal_songs = lambda: (_ for _ in ()).throw(TimeoutError("slow"))
    fresh = AppleMusicPwaController(backend)
    assert fresh.execute(MusicAction(
        MusicActionType.PLAY_SONG, title="Marigold",
    )).ok


def test_artist_search_resolves_cross_script_artist_name() -> None:
    backend = FakeAppleMusicBackend()
    semantics = FakeCandidateSemantics(song_index=0)

    result = AppleMusicPwaController(
        backend, candidate_semantics=semantics,
    ).execute(MusicAction(
        MusicActionType.PLAY_SONG, title="Marigold", artist="아이묭",
    ))

    assert result.ok
    assert result.data["now_playing"]["artist"] == "aimyon"
    assert len(semantics.song_calls) == 1

    artist_result = AppleMusicPwaController(backend).execute(MusicAction(
        MusicActionType.PLAY_ARTIST, artist="아이묭",
    ))
    assert artist_result.ok
    assert artist_result.data["artist"] == "aimyon"


def test_title_only_exact_catalog_surface_does_not_jump_to_cross_script_leader() -> None:
    backend = FakeAppleMusicBackend()
    original = MusicItem(
        "original", "Marigold", "aimyon", "Marigold - Single", "ORIGINAL", 0,
    )
    korean_namesake = MusicItem(
        "namesake", "마리골드", "Other Artist", "마리골드 - Single", "OTHER", 2,
    )
    backend.songs = (original, korean_namesake)
    backend.current = korean_namesake

    result = AppleMusicPwaController(backend).execute(MusicAction(
        MusicActionType.PLAY_SONG, title="마리골드",
    ))

    assert result.ok
    assert result.data["now_playing"]["id"] == "namesake"


def test_cli_artist_title_requires_matching_leaders_from_two_queries() -> None:
    backend = FakeAppleMusicBackend()
    original = MusicItem(
        "original", "Marigold", "aimyon", "Marigold - Single", "ORIGINAL", 0,
    )
    supporting = tuple(
        MusicItem(f"aimyon-{rank}", f"Song {rank}", "aimyon", search_rank=rank)
        for rank in range(1, 5)
    )
    title_only = (original, MusicItem(
        "namesake", "마리골드", "Other Artist", search_rank=2,
    ))
    backend.search_songs = lambda query: (
        (original,) + supporting if "아이묭" in query else title_only
    )
    backend.songs = (original,) + supporting + title_only[1:]
    backend.current = original
    controller = AppleMusicPwaController(
        backend, candidate_semantics=FakeCandidateSemantics(song_index=0),
    )

    result = controller.execute(MusicAction(
        MusicActionType.PLAY_SONG, title="마리골드", artist="아이묭",
    ))
    assert result.ok
    assert result.data["now_playing"]["id"] == "original"

    conflicting = replace(original, item_id="different")
    backend.search_songs = lambda query: (
        (original,) + supporting if "아이묭" in query else (conflicting,)
    )
    refused = AppleMusicPwaController(backend).execute(MusicAction(
        MusicActionType.PLAY_SONG, title="마리골드", artist="아이묭",
    ))
    assert not refused.ok
    assert refused.data["reason"] == "no_match"


def test_artist_qualified_cross_script_title_uses_dominant_exact_artist() -> None:
    backend = FakeAppleMusicBackend()
    original = MusicItem(
        "original", "Suiheisen", "백 넘버", "Suiheisen - Single", "RECORDING", 0,
    )
    supporting = tuple(
        MusicItem(f"back-number-{rank}", f"Song {rank}", "백 넘버", search_rank=rank)
        for rank in range(1, 5)
    )
    title_only = (
        MusicItem("namesake", "수평선", "Other Artist", search_rank=0),
    )
    backend.search_songs = lambda query: (
        (original,) + supporting if "백넘버" in query else title_only
    )
    backend.songs = (original,) + supporting
    backend.current = original

    result = AppleMusicPwaController(
        backend, candidate_semantics=FakeCandidateSemantics(song_index=0),
    ).execute(MusicAction(
        MusicActionType.PLAY_SONG, title="수평선", artist="백넘버",
    ))

    assert result.ok
    assert result.data["now_playing"]["id"] == "original"


def test_corroborated_native_candidate_is_stable_without_semantic_judge() -> None:
    backend = FakeAppleMusicBackend()
    target = MusicItem(
        "target", "Canonical Title", "Canonical Artist",
        "Canonical Title - Single", "TARGET", 0,
    )
    supporting = tuple(
        MusicItem(
            f"support-{rank}", f"Other Song {rank}", "Canonical Artist",
            search_rank=rank,
        )
        for rank in range(1, 5)
    )
    backend.songs = (target, *supporting)
    backend.current = target
    backend.search_songs = lambda query: (
        (target, *supporting) if query == "spoken artist spoken title"
        else (target,) if query == "spoken title" else ()
    )
    semantics = FakeCandidateSemantics(song_index=None)
    controller = AppleMusicPwaController(
        backend, candidate_semantics=semantics, health_cache_seconds=60,
    )
    action = MusicAction(
        MusicActionType.PLAY_SONG, title="Canonical Title", artist="Canonical Artist",
        source_query="spoken artist spoken title", artist_explicit=True,
    )

    results = [controller.execute(action) for _ in range(3)]

    assert all(result.ok for result in results)
    assert [result.data["now_playing"]["id"] for result in results] == [
        "target", "target", "target",
    ]
    assert semantics.song_calls == []
    assert semantics.rewrite_calls == []


def test_semantic_fallback_handles_ambiguous_rule_parser_slots() -> None:
    backend = FakeAppleMusicBackend()
    original = MusicItem(
        "original", "Color Your Night", "Soundtrack Artist",
        "Original Soundtrack", "ORIGINAL", 0,
    )
    cover = MusicItem(
        "cover", "Color Your Night", "Cover Artist", "Cover", "COVER", 4,
    )
    backend.songs = (original, cover)
    backend.current = original
    action = RuleBasedMusicActionParser().parse("Color Your Night 틀어줘").action

    semantics = FakeCandidateSemantics(song_index=0)
    result = AppleMusicPwaController(
        backend, candidate_semantics=semantics,
    ).execute(action)

    assert result.ok
    assert result.data["now_playing"]["id"] == "original"
    assert len(semantics.song_calls) == 1


def test_artist_hint_rejects_title_similar_cover_from_different_artist() -> None:
    backend = FakeAppleMusicBackend()
    backend.songs = (
        MusicItem(
            "cover", "사랑을 전하고 싶다든가 [피아노]", "Cover Artist",
            search_rank=0,
        ),
    )
    action = RuleBasedMusicActionParser().parse(
        "아이묭 사랑을 전하고 싶다든가 틀어줘",
    ).action

    result = AppleMusicPwaController(backend).execute(action)

    assert not result.ok
    assert result.data["reason"] == "no_match"


def test_exact_unicode_artist_and_title_resolve_and_dedupe_recording() -> None:
    backend = FakeAppleMusicBackend()
    single = MusicItem(
        "single", "アイドル", "YOASOBI", "アイドル - Single", "SAME", 0,
    )
    backend.songs = (single, replace(single, item_id="album", search_rank=1))
    backend.current = single
    action = RuleBasedMusicActionParser().parse("YOASOBI アイドル 틀어줘").action

    result = AppleMusicPwaController(backend).execute(action)

    assert result.ok
    assert result.data["now_playing"]["id"] == "single"


def test_lower_ranked_exact_match_beats_rank_zero_wrong_song() -> None:
    backend = FakeAppleMusicBackend()
    wrong = MusicItem("wrong", "Different Song", "Artist", search_rank=0)
    expected = MusicItem("expected", "Target Song", "Artist", search_rank=6)
    backend.songs = (wrong, expected)
    backend.current = expected

    result = AppleMusicPwaController(backend).execute(MusicAction(
        MusicActionType.PLAY_SONG, title="Target Song", artist="Artist",
        artist_explicit=True,
    ))

    assert result.ok
    assert result.data["now_playing"]["id"] == "expected"


def test_retrieval_diagnostic_preserves_backend_candidate_count(
    monkeypatch, capsys,
) -> None:
    backend = FakeAppleMusicBackend()
    expected = MusicItem("expected", "Target Song", "Artist")
    backend.songs = (expected,)
    backend.current = expected
    monkeypatch.setenv("AMADEUS_MUSIC_DIAGNOSTICS", "1")

    result = AppleMusicPwaController(backend).execute(MusicAction(
        MusicActionType.PLAY_SONG, title="Target Song", artist="Artist",
        source_query="Artist Target Song", artist_explicit=True,
    ))

    assert result.ok
    diagnostic = capsys.readouterr().err
    assert 'parsed:{"action":"play_song","artist":"Artist","title":"Target Song"' in diagnostic
    assert 'candidate_pool:{"native_catalog_count":1,"rewritten_catalog_count":0' \
        in diagnostic
    assert '"personal_index_count":0,"merged_candidate_count":1' in diagnostic


def test_normalized_title_and_artist_match_without_semantics() -> None:
    backend = FakeAppleMusicBackend()
    expected = MusicItem("expected", "Target: Song!", "The Artist")
    backend.songs = (expected,)
    backend.current = expected
    semantics = FakeCandidateSemantics(song_index=None)

    result = AppleMusicPwaController(
        backend, candidate_semantics=semantics,
    ).execute(MusicAction(
        MusicActionType.PLAY_SONG, title="target song", artist="the-artist",
        artist_explicit=True,
    ))

    assert result.ok
    assert semantics.song_calls == []


@pytest.mark.parametrize("judgment", [
    SongCandidateJudgment("match", 99, True, True),
    SongCandidateJudgment("error", error_category="provider_error"),
])
def test_invalid_or_failed_semantic_match_never_executes(judgment) -> None:
    backend = FakeAppleMusicBackend()
    backend.songs = (MusicItem("actual", "Canonical", "Artist"),)
    backend.play_song = lambda item_id: pytest.fail(f"unexpected playback: {item_id}")

    class FailedSemantics(FakeCandidateSemantics):
        def judge_song_candidate(self, action, candidates):
            self.song_calls.append((action, candidates))
            return judgment

    result = AppleMusicPwaController(
        backend, candidate_semantics=FailedSemantics(),
    ).execute(MusicAction(
        MusicActionType.PLAY_SONG, title="surface", artist="spoken artist",
        artist_explicit=True,
    ))

    assert not result.ok
    assert result.data["reason"] == "no_match"


def test_structured_rewrite_queries_return_to_native_catalog_search() -> None:
    backend = FakeAppleMusicBackend()
    expected = MusicItem("canonical", "Canonical Title", "Canonical Artist")
    queries = []

    def search(query):
        queries.append(query)
        return (expected,) if query == "Canonical Title" else ()

    backend.search_songs = search
    backend.songs = (expected,)
    backend.current = expected
    semantics = FakeCandidateSemantics(
        song_index=0, rewrites=(CatalogQueryVariant(title="Canonical Title"),),
    )
    result = AppleMusicPwaController(
        backend, candidate_semantics=semantics,
    ).execute(MusicAction(
        MusicActionType.PLAY_SONG,
        title="surface title",
        source_query="surface query",
        alternate_queries=("Canonical Title",),
    ))

    assert result.ok
    assert queries[:2] == ["surface query", "Canonical Title"]
    assert len(semantics.song_calls) == 1


def test_rewritten_searches_are_bounded_and_deduplicated_by_catalog_id() -> None:
    backend = FakeAppleMusicBackend()
    target = MusicItem("catalog-id", "Canonical", "Canonical Artist")
    queries = []

    def search(query):
        queries.append(query)
        return () if len(queries) == 1 else (target,)

    backend.search_songs = search
    backend.songs = (target,)
    backend.current = target
    rewrites = tuple(
        CatalogQueryVariant(f"Artist {index}", f"Title {index}")
        for index in range(4)
    )
    semantics = FakeCandidateSemantics(song_index=0, rewrites=rewrites)

    result = AppleMusicPwaController(
        backend, candidate_semantics=semantics,
    ).execute(MusicAction(
        MusicActionType.PLAY_SONG, title="surface title", artist="surface artist",
        source_query="surface artist surface title", artist_explicit=True,
    ))

    assert result.ok
    assert len(queries) == 5
    assert semantics.rewrite_calls == [("surface artist", "surface title")]
    assert len(semantics.song_calls) == 1
    assert [item.item_id for item in semantics.song_calls[0][1]] == ["catalog-id"]


def test_clean_structured_fields_are_the_only_rewriter_input() -> None:
    backend = FakeAppleMusicBackend()
    backend.songs = ()
    semantics = FakeCandidateSemantics()

    result = AppleMusicPwaController(
        backend, candidate_semantics=semantics,
    ).execute(MusicAction(
        MusicActionType.PLAY_SONG, artist="clean artist", title="clean title",
        source_query="clean artist clean title", artist_explicit=True,
    ))

    assert not result.ok
    assert semantics.rewrite_calls == [("clean artist", "clean title")]


def test_playlist_resolver_uses_alternate_names_but_keeps_library_authoritative() -> None:
    backend = FakeAppleMusicBackend()
    controller = AppleMusicPwaController(backend)

    result = controller.execute(MusicAction(
        MusicActionType.PLAY_PLAYLIST,
        playlist="공부용",
        alternate_queries=("공부할 때",),
    ))

    assert result.ok
    assert result.data["playlist"] == "공부할 때"


def test_playlist_resolver_uses_unique_artist_membership_for_cross_script_name() -> None:
    backend = FakeAppleMusicBackend()
    expected = PlaylistItem("personal.artist", "BackNumber")
    other = PlaylistItem("personal.other", "Focus")
    backend.playlists = PlaylistSnapshot((expected, other))
    artist_tracks = tuple(
        MusicItem(f"artist-{rank}", f"Song {rank}", "back number", search_rank=rank)
        for rank in range(5)
    )
    other_tracks = tuple(
        MusicItem(f"other-{rank}", f"Other {rank}", "other artist", search_rank=rank)
        for rank in range(5)
    )
    backend.search_playlists = lambda query: ()
    backend.search_songs = lambda query: artist_tracks
    backend.playlist_tracks = lambda playlist_id: (
        artist_tracks if playlist_id == expected.playlist_id else other_tracks
    )
    backend.queue = artist_tracks
    backend.current = artist_tracks[0]

    result = AppleMusicPwaController(backend).execute(MusicAction(
        MusicActionType.PLAY_PLAYLIST, playlist="백넘버",
    ))

    assert result.ok
    assert result.data["playlist"] == "BackNumber"


def test_playlist_membership_fallback_refuses_competing_personal_playlists() -> None:
    backend = FakeAppleMusicBackend()
    backend.playlists = PlaylistSnapshot((
        PlaylistItem("personal.one", "BackNumber"),
        PlaylistItem("personal.two", "Favorites"),
    ))
    artist_tracks = tuple(
        MusicItem(f"artist-{rank}", f"Song {rank}", "back number", search_rank=rank)
        for rank in range(5)
    )
    backend.search_playlists = lambda query: ()
    backend.search_songs = lambda query: artist_tracks
    backend.playlist_tracks = lambda playlist_id: artist_tracks

    result = AppleMusicPwaController(backend).execute(MusicAction(
        MusicActionType.PLAY_PLAYLIST, playlist="백넘버",
    ))

    assert not result.ok
    assert result.data["reason"] == "ambiguous"
    assert result.data["candidate_options"] == ["BackNumber", "Favorites"]


def test_playlist_semantics_cannot_override_real_personal_candidate_ambiguity() -> None:
    backend = FakeAppleMusicBackend()
    backend.playlists = PlaylistSnapshot((
        PlaylistItem("personal.one", "Canonical Artist"),
        PlaylistItem("personal.two", "Canonical Artist Live Set"),
    ))
    artist_tracks = tuple(
        MusicItem(f"artist-{rank}", f"Song {rank}", "canonical artist", search_rank=rank)
        for rank in range(5)
    )
    backend.search_playlists = lambda query: ()
    backend.search_songs = lambda query: artist_tracks
    backend.playlist_tracks = lambda playlist_id: artist_tracks
    backend.queue = artist_tracks
    backend.current = artist_tracks[0]
    semantics = FakeCandidateSemantics(playlist_index=1)

    result = AppleMusicPwaController(
        backend, candidate_semantics=semantics,
    ).execute(MusicAction(
        MusicActionType.PLAY_PLAYLIST, playlist="translated live playlist",
    ))

    assert not result.ok
    assert result.data["reason"] == "ambiguous"
    assert result.data["candidate_options"] == [
        "Canonical Artist", "Canonical Artist Live Set",
    ]
    assert semantics.playlist_calls == []


def test_transport_bypasses_candidate_discovery() -> None:
    backend = FakeAppleMusicBackend()
    semantics = FakeCandidateSemantics(
        song_index=0, rewrites=(CatalogQueryVariant(title="unused"),),
    )

    result = AppleMusicPwaController(
        backend, candidate_semantics=semantics,
    ).execute(MusicAction(MusicActionType.PAUSE))

    assert result.ok
    assert semantics.song_calls == []
    assert semantics.playlist_calls == []
    assert semantics.rewrite_calls == []


def test_semantic_alternate_query_can_discover_cross_script_title() -> None:
    backend = FakeAppleMusicBackend()
    expected = MusicItem(
        "expected", "Suiheisen", "back number", "Suiheisen - Single", "RECORDING", 0,
    )
    wrong = MusicItem("wrong", "수평선", "other artist", "Other", "OTHER", 0)
    queries = []

    def search(query):
        queries.append(query)
        if query == "back number Suiheisen":
            return (expected,)
        if query == "수평선":
            return (wrong,)
        return ()

    backend.search_songs = search
    backend.songs = (expected, wrong)
    backend.current = expected
    semantics = FakeCandidateSemantics(
        song_index=0,
        rewrites=(CatalogQueryVariant("back number", "Suiheisen"),),
    )
    result = AppleMusicPwaController(
        backend, candidate_semantics=semantics,
    ).execute(MusicAction(
        MusicActionType.PLAY_SONG, title="수평선", artist="백넘버",
        source_query="백넘버 수평선",
        alternate_queries=("back number Suiheisen",), artist_explicit=True,
    ))

    assert result.ok
    assert result.data["now_playing"]["id"] == "expected"
    assert "back number Suiheisen" in queries


def test_retrieval_first_judges_only_actual_native_song_candidates() -> None:
    backend = FakeAppleMusicBackend()
    target = MusicItem(
        "target", "Canonical Cross Script Title", "Canonical Artist",
        "Catalog Album", "TARGET", 3,
    )
    broad = tuple(
        MusicItem(f"other-{rank}", f"Other {rank}", "Canonical Artist", search_rank=rank)
        for rank in range(3)
    ) + (target,)
    backend.songs = broad
    backend.current = target
    backend.search_songs = lambda query: broad
    semantics = FakeCandidateSemantics(song_index=3)

    result = AppleMusicPwaController(
        backend, candidate_semantics=semantics,
    ).execute(MusicAction(
        MusicActionType.PLAY_SONG, title="한국어 곡명", artist="한국어 가수",
        source_query="한국어 가수 한국어 곡명",
    ))

    assert result.ok
    assert result.data["now_playing"]["id"] == "target"
    assert [item.item_id for item in semantics.song_calls[0][1]] == [
        "other-0", "other-1", "other-2", "target",
    ]
    assert semantics.rewrite_calls == []


def test_uncertain_candidate_judgment_does_not_execute() -> None:
    backend = FakeAppleMusicBackend()
    broad = (MusicItem("wrong", "Wrong Song", "Canonical Artist"),)
    backend.search_songs = lambda query: broad
    backend.play_song = lambda item_id: pytest.fail(f"unexpected playback: {item_id}")
    semantics = FakeCandidateSemantics(song_index=None)

    result = AppleMusicPwaController(
        backend, candidate_semantics=semantics,
    ).execute(MusicAction(
        MusicActionType.PLAY_SONG, title="한국어 곡명", artist="한국어 가수",
    ))

    assert not result.ok
    assert result.data["reason"] == "no_match"
    assert len(semantics.song_calls) == 1


def test_candidate_judgment_with_artist_evidence_only_cannot_execute_explicit_title() -> None:
    backend = FakeAppleMusicBackend()
    broad = (MusicItem("artist-top", "Unrelated Top Song", "Canonical Artist"),)
    backend.search_songs = lambda query: broad
    backend.play_song = lambda item_id: pytest.fail(f"unexpected playback: {item_id}")
    semantics = FakeCandidateSemantics(
        song_index=0, title_equivalent=False, artist_equivalent=True,
    )

    result = AppleMusicPwaController(
        backend, candidate_semantics=semantics,
    ).execute(MusicAction(
        MusicActionType.PLAY_SONG, title="spoken title", artist="spoken artist",
        artist_explicit=True,
    ))

    assert not result.ok
    assert result.data["reason"] == "no_match"


def test_search_recovery_runs_only_when_initial_retrieval_is_empty() -> None:
    backend = FakeAppleMusicBackend()
    target = MusicItem(
        "target", "Canonical Title", "Canonical Artist",
        "Canonical Title - Single", "TARGET", 0,
    )
    queries = []

    def search(query):
        queries.append(query)
        return (target,) if query == "recovery hint" else ()

    backend.search_songs = search
    backend.songs = (target,)
    backend.current = target
    semantics = FakeCandidateSemantics(
        song_index=0, rewrites=(CatalogQueryVariant(title="recovery hint"),),
    )

    result = AppleMusicPwaController(
        backend, candidate_semantics=semantics,
    ).execute(MusicAction(
        MusicActionType.PLAY_SONG, title="surface title", artist="surface artist",
        source_query="surface artist surface title",
    ))

    assert result.ok
    assert queries == ["surface artist surface title", "recovery hint"]
    assert semantics.rewrite_calls == [("surface artist", "surface title")]


def test_search_recovery_hint_is_not_execution_evidence_without_title_equivalence() -> None:
    backend = FakeAppleMusicBackend()
    recovered = MusicItem(
        "wrong", "Generated Hint", "Canonical Artist", "Generated Hint - Single",
    )
    backend.search_songs = lambda query: (recovered,) if query == "generated hint" else ()
    backend.play_song = lambda item_id: pytest.fail(f"unexpected playback: {item_id}")
    semantics = FakeCandidateSemantics(
        song_index=0, rewrites=(CatalogQueryVariant(title="generated hint"),),
        title_equivalent=False, artist_equivalent=True,
    )

    result = AppleMusicPwaController(
        backend, candidate_semantics=semantics,
    ).execute(MusicAction(
        MusicActionType.PLAY_SONG, title="missing title", artist="spoken artist",
        artist_explicit=True,
    ))

    assert not result.ok
    assert result.data["reason"] == "no_match"


def test_native_exact_match_skips_candidate_semantics_and_recovery() -> None:
    backend = FakeAppleMusicBackend()
    semantics = FakeCandidateSemantics(
        song_index=0, rewrites=(CatalogQueryVariant(title="unused"),),
    )

    result = AppleMusicPwaController(
        backend, candidate_semantics=semantics,
    ).execute(MusicAction(MusicActionType.PLAY_SONG, title="Marigold"))

    assert result.ok
    assert semantics.song_calls == []
    assert semantics.rewrite_calls == []


def test_artist_resolver_uses_alternate_queries_and_verifies_now_playing() -> None:
    backend = FakeAppleMusicBackend()
    controller = AppleMusicPwaController(backend)

    result = controller.execute(MusicAction(
        MusicActionType.PLAY_ARTIST,
        artist="아이묭",
        alternate_queries=("aimyon",),
    ))

    assert result.ok
    assert result.data["artist"] == "aimyon"


def test_playlist_track_resolver_uses_alternate_combined_query() -> None:
    backend = FakeAppleMusicBackend()
    controller = AppleMusicPwaController(backend)

    result = controller.execute(MusicAction(
        MusicActionType.PLAY_PLAYLIST_TRACK,
        playlist="아이묭 대표곡",
        title="마리골드",
        artist="아이묭",
        alternate_queries=("aimyon Marigold",),
    ))

    assert result.ok
    assert result.data["now_playing"]["id"] == "song.marigold"


def test_playlist_listing_is_cached_and_preserves_partial_status() -> None:
    backend = FakeAppleMusicBackend()
    backend.playlists = PlaylistSnapshot(
        backend.playlists.items, partial=True, warning="sidebar_snapshot",
    )
    controller = AppleMusicPwaController(backend, playlist_cache_seconds=60)
    action = MusicAction(MusicActionType.LIST_PLAYLISTS)

    first = controller.execute(action)
    second = controller.execute(action)

    assert first.ok and first.data["partial"] is True
    assert first.data["warning"] == "sidebar_snapshot"
    assert second.ok
    assert backend.list_calls == 1


def test_playlist_listing_uses_stale_cache_after_refresh_timeout() -> None:
    backend = FakeAppleMusicBackend()
    now = [0.0]
    controller = AppleMusicPwaController(
        backend, playlist_cache_seconds=1, clock=lambda: now[0],
    )
    assert controller.execute(MusicAction(MusicActionType.LIST_PLAYLISTS)).ok
    now[0] = 2.0
    backend.list_playlists = lambda: (_ for _ in ()).throw(TimeoutError("slow"))

    stale = controller.execute(MusicAction(MusicActionType.LIST_PLAYLISTS))

    assert stale.ok
    assert stale.data["partial"] is True
    assert stale.data["warning"] == "playlist_refresh_failed"


def test_backend_unavailable_is_a_safe_failure() -> None:
    backend = FakeAppleMusicBackend()
    backend.play = lambda: (_ for _ in ()).throw(TimeoutError("CDP timeout"))
    result = AppleMusicPwaController(backend).execute(MusicAction(MusicActionType.PLAY))

    assert not result.ok
    assert result.data["reason"] == "backend_unavailable"


def test_controller_rejects_unhealthy_or_signed_out_pwa() -> None:
    backend = FakeAppleMusicBackend()
    backend.selector_health = lambda: {"authorized": False, "player": True}

    result = AppleMusicPwaController(backend).execute(
        MusicAction(MusicActionType.PLAY),
    )

    assert not result.ok
    assert result.data["reason"] == "pwa_unavailable"


def test_cdp_backend_launches_pwa_when_debug_browser_has_no_music_target(
    monkeypatch, tmp_path,
) -> None:
    backend = CdpAppleMusicBackend(
        chrome_path=tmp_path / "chrome.exe", timeout_seconds=0.05,
    )
    target = {
        "url": "https://music.apple.com/kr/home",
        "title": "Apple Music - 웹 플레이어",
        "webSocketDebuggerUrl": "ws://127.0.0.1/devtools/page/1",
    }
    reads = iter(([{"url": "chrome://settings"}], [target]))
    launched = []
    monkeypatch.setattr(backend, "_read_targets", lambda: next(reads))
    monkeypatch.setattr(backend, "_launch", lambda: launched.append(True))

    assert backend._target() == target
    assert launched == [True]


def test_cdp_backend_waits_for_stable_route_without_relaunching(monkeypatch) -> None:
    backend = CdpAppleMusicBackend(timeout_seconds=0.1)
    loading = {
        "url": "https://music.apple.com/kr/", "title": "",
        "webSocketDebuggerUrl": "ws://127.0.0.1/devtools/page/1",
    }
    ready = {
        "url": "https://music.apple.com/kr/home", "title": "Apple Music",
        "webSocketDebuggerUrl": "ws://127.0.0.1/devtools/page/1",
    }
    reads = iter(([loading], [ready]))
    launched = []
    monkeypatch.setattr(backend, "_read_targets", lambda: next(reads))
    monkeypatch.setattr(backend, "_launch", lambda: launched.append(True))

    assert backend._target() == ready
    assert launched == []


def test_cdp_launcher_limits_debugging_and_websocket_origin_to_localhost(
    monkeypatch, tmp_path,
) -> None:
    chrome = tmp_path / "chrome.exe"
    chrome.touch()
    profile = tmp_path / "profile"
    backend = CdpAppleMusicBackend(
        port=9333, chrome_path=chrome, profile_dir=profile,
    )
    calls = []
    monkeypatch.setattr(
        "app.music_control.cdp.subprocess.Popen",
        lambda args, **kwargs: calls.append((args, kwargs)),
    )

    backend._launch()

    args, kwargs = calls[0]
    assert "--remote-debugging-address=127.0.0.1" in args
    assert "--remote-debugging-port=9333" in args
    assert "--remote-allow-origins=http://127.0.0.1:9333" in args
    assert "--remote-allow-origins=*" not in args
    assert f"--user-data-dir={profile}" in args
    assert kwargs == {"shell": False, "close_fds": True}


def test_cdp_connection_timeout_is_reported_as_safe_backend_failure(monkeypatch) -> None:
    backend = CdpAppleMusicBackend()
    monkeypatch.setattr(backend, "_target", lambda: {
        "webSocketDebuggerUrl": "ws://127.0.0.1/devtools/page/1",
    })
    monkeypatch.setattr(
        "app.music_control.cdp.create_connection",
        lambda *args, **kwargs: (_ for _ in ()).throw(WebSocketTimeoutException()),
    )

    with pytest.raises(TimeoutError, match=(
        "operation=health_check, stage=connect, timeout=websocket_connect, "
        "recovery=target_rediscovery_attempted"
    )):
        backend.selector_health()


def test_cdp_connect_failure_rediscovers_target_once_before_send(monkeypatch) -> None:
    backend = CdpAppleMusicBackend()
    targets = iter((
        {"webSocketDebuggerUrl": "ws://127.0.0.1/devtools/page/stale"},
        {"webSocketDebuggerUrl": "ws://127.0.0.1/devtools/page/fresh"},
    ))
    socket = FakeWebSocket({
        "id": 1,
        "result": {"result": {"value": {
            "authorized": True, "navigation": True, "player": True,
        }}},
    })
    connections = []
    monkeypatch.setattr(backend, "_target", lambda: next(targets))

    def connect(url, **kwargs):
        connections.append(url)
        if url.endswith("/stale"):
            raise WebSocketTimeoutException()
        return socket

    monkeypatch.setattr("app.music_control.cdp.create_connection", connect)

    assert backend.selector_health()["authorized"] is True
    assert connections == [
        "ws://127.0.0.1/devtools/page/stale",
        "ws://127.0.0.1/devtools/page/fresh",
    ]
    assert len(socket.sent) == 1


def test_cdp_runtime_timeout_is_classified_and_never_retried(monkeypatch) -> None:
    backend = CdpAppleMusicBackend()
    socket = FakeWebSocket(error=WebSocketTimeoutException())
    target_calls = []
    connections = []
    monkeypatch.setattr(backend, "_target", lambda: (
        target_calls.append(True)
        or {"webSocketDebuggerUrl": "ws://127.0.0.1/devtools/page/1"}
    ))
    monkeypatch.setattr(
        "app.music_control.cdp.create_connection",
        lambda *args, **kwargs: connections.append(True) or socket,
    )

    with pytest.raises(TimeoutError, match=(
        "operation=catalog_song_search, stage=runtime_evaluation, "
        "timeout=websocket_receive, "
        "recovery=not_attempted"
    )):
        backend.search_songs("query")

    assert len(target_calls) == len(connections) == 1
    assert len(socket.sent) == 1


def test_cdp_websocket_closes_after_success_and_each_call_is_fresh(monkeypatch) -> None:
    backend = CdpAppleMusicBackend()
    monkeypatch.setattr(backend, "_target", lambda: {
        "webSocketDebuggerUrl": "ws://127.0.0.1/devtools/page/1",
    })
    sockets = [
        FakeWebSocket({
            "id": 1,
            "result": {"result": {"value": {
                "authorized": True, "navigation": True, "player": True,
            }}},
        }),
        FakeWebSocket({
            "id": 1,
            "result": {"result": {"value": {
                "authorized": True, "navigation": True, "player": True,
            }}},
        }),
    ]
    pending = iter(sockets)
    monkeypatch.setattr(
        "app.music_control.cdp.create_connection",
        lambda *args, **kwargs: next(pending),
    )

    assert backend.selector_health()["authorized"] is True
    assert backend.selector_health()["player"] is True
    assert all(socket.closed for socket in sockets)
    assert all(socket.sent[0]["method"] == "Runtime.evaluate" for socket in sockets)
    assert all(socket.sent[0]["params"]["userGesture"] is True for socket in sockets)
    assert all(
        "typeof MusicKit.getInstance" in socket.sent[0]["params"]["expression"]
        and "typeof mk.api?.music" in socket.sent[0]["params"]["expression"]
        for socket in sockets
    )


def test_cdp_websocket_closes_when_receive_times_out(monkeypatch) -> None:
    backend = CdpAppleMusicBackend()
    socket = FakeWebSocket(error=WebSocketTimeoutException())
    monkeypatch.setattr(backend, "_target", lambda: {
        "webSocketDebuggerUrl": "ws://127.0.0.1/devtools/page/1",
    })
    monkeypatch.setattr(
        "app.music_control.cdp.create_connection", lambda *args, **kwargs: socket,
    )

    with pytest.raises(TimeoutError, match="CDP timed out"):
        backend.selector_health()

    assert socket.closed


def test_cdp_websocket_closes_when_command_returns_error(monkeypatch) -> None:
    backend = CdpAppleMusicBackend()
    socket = FakeWebSocket({
        "id": 1,
        "result": {"result": {"subtype": "error", "description": "bad command"}},
    })
    monkeypatch.setattr(backend, "_target", lambda: {
        "webSocketDebuggerUrl": "ws://127.0.0.1/devtools/page/1",
    })
    monkeypatch.setattr(
        "app.music_control.cdp.create_connection", lambda *args, **kwargs: socket,
    )

    with pytest.raises(Exception, match="bad command"):
        backend.selector_health()

    assert socket.closed


def test_cdp_maps_readiness_failure_and_closes_connection(monkeypatch) -> None:
    backend = CdpAppleMusicBackend()
    socket = FakeWebSocket({
        "id": 1,
        "result": {
            "result": {"type": "object", "subtype": "error"},
            "exceptionDetails": {
                "exception": {
                    "description": "Error: [AMADEUS:player_not_ready] not ready",
                },
            },
        },
    })
    monkeypatch.setattr(backend, "_target", lambda: {
        "webSocketDebuggerUrl": "ws://127.0.0.1/devtools/page/1",
    })
    monkeypatch.setattr(
        "app.music_control.cdp.create_connection", lambda *args, **kwargs: socket,
    )

    with pytest.raises(MusicControlError) as error:
        backend.selector_health()

    assert error.value.code == "player_not_ready"
    assert socket.closed


def test_cdp_catalog_search_uses_music_api_and_preserves_isrc(monkeypatch) -> None:
    backend = CdpAppleMusicBackend()
    socket = FakeWebSocket({
        "id": 1,
        "result": {"result": {"value": [{
            "id": "1402042897", "title": "Marigold", "artist": "aimyon",
            "album": "Marigold - Single", "recordingId": "JPWP01870687",
            "searchRank": 0,
        }]}},
    })
    monkeypatch.setattr(backend, "_target", lambda: {
        "webSocketDebuggerUrl": "ws://127.0.0.1/devtools/page/1",
    })
    monkeypatch.setattr(
        "app.music_control.cdp.create_connection", lambda *args, **kwargs: socket,
    )

    songs = backend.search_songs("aimyon Marigold")

    assert songs[0].recording_id == "JPWP01870687"
    expression = socket.sent[0]["params"]["expression"]
    assert "mk.api.music" in expression
    assert "/v1/catalog/${mk.storefrontId}/search" in expression
    assert "__amadeusCatalogDiagnostics" in expression
    assert "playParamsId" in expression
    assert "playParamsKind" in expression
    assert "storefrontId" in expression
    assert socket.closed


def test_cdp_catalog_search_reports_raw_and_extracted_counts(monkeypatch, capsys) -> None:
    backend = CdpAppleMusicBackend()
    monkeypatch.setenv("AMADEUS_MUSIC_DIAGNOSTICS", "1")
    monkeypatch.setattr(backend, "_evaluate", lambda operation, expression, args=None: [
        {"id": "one", "title": "One", "artist": "Artist", "searchRank": 0},
        {"id": "two", "title": "Two", "artist": "Artist", "searchRank": 1},
    ])

    items = backend.search_songs("Artist Song")

    assert len(items) == 2
    diagnostic = capsys.readouterr().err
    assert '"backend_status":"ok"' in diagnostic
    assert '"raw_song_count":2' in diagnostic
    assert '"extracted_count":2' in diagnostic


def test_cdp_library_playlists_use_authorized_music_api_not_current_dom(monkeypatch) -> None:
    backend = CdpAppleMusicBackend()
    socket = FakeWebSocket({
        "id": 1,
        "result": {"result": {"value": {
            "items": [{"id": "p.goodnight", "name": "굿나잇"}],
            "partial": False, "warning": "",
        }}},
    })
    monkeypatch.setattr(backend, "_target", lambda: {
        "webSocketDebuggerUrl": "ws://127.0.0.1/devtools/page/1",
    })
    monkeypatch.setattr(
        "app.music_control.cdp.create_connection", lambda *args, **kwargs: socket,
    )

    snapshot = backend.list_playlists()

    assert snapshot.items == (PlaylistItem("p.goodnight", "굿나잇"),)
    expression = socket.sent[0]["params"]["expression"]
    assert "/v1/me/library/playlists" in expression


def test_cdp_personal_index_reads_library_and_all_playlists_without_known_ids(
    monkeypatch,
) -> None:
    backend = CdpAppleMusicBackend()
    expressions = []

    def fake_evaluate(operation, expression, args=None):
        assert operation == "personal_music_index"
        expressions.append(expression)
        return {"items": [{
            "id": "catalog.song", "title": "Song", "artist": "Artist",
            "album": "Album", "searchRank": 100, "inLibrary": True,
            "playlistCount": 3,
        }], "partial": False, "warning": ""}

    monkeypatch.setattr(backend, "_evaluate", fake_evaluate)

    snapshot = backend.personal_songs()

    assert snapshot.items[0].item.item_id == "catalog.song"
    assert snapshot.items[0].in_library
    assert snapshot.items[0].playlist_count == 3
    expression = expressions[0]
    assert "/v1/me/library/songs" in expression
    assert "/v1/me/library/playlists" in expression
    assert "playlist.id" in expression
    assert "굿나잇" not in expression
    assert "querySelectorAll" not in expression


def test_cdp_playlist_queue_and_metadata_use_bounded_readiness_polling(monkeypatch) -> None:
    backend = CdpAppleMusicBackend()
    responses = iter((
        {"id": 1, "result": {"result": {"value": [{
            "id": "song.first", "title": "First", "artist": "Artist",
        }]}}},
        {"id": 1, "result": {"result": {"value": {"dispatched": True}}}},
        {"id": 1, "result": {"result": {"value": {
            "token": "123", "phase": "play", "failed": False,
            "isPlaying": True, "expectedNowPlayingId": "song.first", "current": {
                "id": "song.first", "title": "First", "artist": "Artist",
            },
        }}}},
    ))
    sockets = []

    def connect(*args, **kwargs):
        socket = FakeWebSocket(next(responses))
        sockets.append(socket)
        return socket

    monkeypatch.setattr(backend, "_target", lambda: {
        "webSocketDebuggerUrl": "ws://127.0.0.1/devtools/page/1",
    })
    monkeypatch.setattr("app.music_control.cdp.create_connection", connect)
    monkeypatch.setattr("app.music_control.cdp.time.monotonic_ns", lambda: 123)

    queue = backend.load_playlist("p.goodnight")
    played = backend.play_queue_item(0)

    assert queue[0].item_id == played.item_id == "song.first"
    queue_expression = sockets[0].sent[0]["params"]["expression"]
    dispatch_expression = sockets[1].sent[0]["params"]["expression"]
    verify_expression = sockets[2].sent[0]["params"]["expression"]
    assert "queueDeadline" in queue_expression
    assert "await wait(100)" in queue_expression
    assert "void (async()=>" in dispatch_expression
    assert "state.phase='change_media_item'" in dispatch_expression
    assert "state.phase='play'" in dispatch_expression
    assert "__amadeusPlaybackCommand" in verify_expression


@pytest.mark.parametrize(("method", "expected_fragment"), [
    ("play", '"expectedPlaying": true'),
    ("pause", '"expectedPlaying": false'),
    ("next", "previousNowPlayingId"),
    ("previous", "previousNowPlayingId"),
])
def test_cdp_transport_waits_for_verified_state_change(
    monkeypatch, method, expected_fragment,
) -> None:
    backend = CdpAppleMusicBackend()
    calls = []

    def evaluate(operation, expression, args=None):
        calls.append((operation, expression, args))
        if operation == "playback_command_dispatch":
            return {"dispatched": True}
        return {
            "token": args["commandToken"], "phase": "command_complete",
            "failed": False, "isPlaying": method != "pause",
            "previousNowPlayingId": (
                "song.previous" if method in {"next", "previous"} else ""
            ),
            "current": {"id": "song.changed", "title": "Changed", "artist": "Artist"},
        }

    monkeypatch.setattr(backend, "_evaluate", evaluate)

    getattr(backend, method)()

    assert expected_fragment in calls[0][1] or expected_fragment in json.dumps(calls[0][2])
    assert [call[0] for call in calls] == [
        "playback_command_dispatch", "playback_verification",
    ]


def test_cdp_song_play_is_idempotent_for_current_track(monkeypatch) -> None:
    backend = CdpAppleMusicBackend()
    calls = []

    def evaluate(operation, expression, args=None):
        calls.append((operation, expression, args))
        if operation == "playback_command_dispatch":
            return {"dispatched": True}
        return {
            "token": args["commandToken"], "phase": "play", "failed": False,
            "isPlaying": True, "current": {
                "id": "1402042897", "title": "Marigold", "artist": "aimyon",
                "album": "Marigold - Single",
            },
        }

    monkeypatch.setattr(backend, "_evaluate", evaluate)

    assert backend.play_song("1402042897").item_id == "1402042897"
    dispatch = calls[0][1]
    assert "mk.nowPlayingItem?.id" in dispatch
    assert "mk.setQueue({songs:[args.id]})" in dispatch
    assert "mk.setQueue({song:args.id})" not in dispatch
    assert "state.phase='set_queue_started'" in dispatch
    assert "state.phase='set_queue_resolved'" in dispatch
    assert "queueContainsExpected" in dispatch
    assert "state.phase='play_started'" in dispatch
    assert "state.phase='play_resolved'" in dispatch
    assert "state.queueOptions={songs:[String(args.id)]}" in dispatch
    assert "state.currentAfterSetQueue" in dispatch
    assert "state.currentAfterPlay" in dispatch
    assert calls[1][0] == "playback_verification"


def test_cdp_success_diagnostic_is_explicitly_opt_in(monkeypatch, capsys) -> None:
    backend = CdpAppleMusicBackend()
    monkeypatch.setenv("AMADEUS_MUSIC_DIAGNOSTICS", "1")
    monkeypatch.setattr(backend, "_evaluate", lambda operation, expression, args=None: (
        {"dispatched": True} if operation == "playback_command_dispatch" else {
            "token": args["commandToken"], "phase": "command_complete",
            "failed": False, "isPlaying": True,
            "current": {"id": "song.expected", "title": "Expected", "artist": "Artist"},
            "diagnostics": {"queueOptions": {"songs": ["song.expected"]}},
        }
    ))

    backend.play_song("song.expected")

    diagnostic = capsys.readouterr().err
    assert diagnostic.startswith("[music_playback_diagnostic] ")
    assert '"queueOptions":{"songs":["song.expected"]}' in diagnostic


def test_cdp_unresolved_command_promise_times_out_in_verification_without_retry(
    monkeypatch,
) -> None:
    backend = CdpAppleMusicBackend(timeout_seconds=8)
    calls = []
    clock = iter((0.0, 0.0, 9.0))
    monkeypatch.setattr("app.music_control.cdp.time.monotonic", lambda: next(clock))
    monkeypatch.setattr("app.music_control.cdp.time.sleep", lambda seconds: None)

    def evaluate(operation, expression, args=None):
        calls.append((operation, expression, args))
        if operation == "playback_command_dispatch":
            return {"dispatched": True}
        return {
            "token": args["commandToken"], "phase": "set_queue", "failed": False,
            "isPlaying": False, "current": None,
        }

    monkeypatch.setattr(backend, "_evaluate", evaluate)

    with pytest.raises(TimeoutError, match=(
        "operation=playback_verification, stage=state_poll, "
        "timeout=deadline_exceeded, recovery=not_attempted, command_phase=set_queue"
    )):
        backend.play_song("song.expected")

    assert [call[0] for call in calls] == [
        "playback_command_dispatch", "playback_verification",
    ]
    assert sum(call[0] == "playback_command_dispatch" for call in calls) == 1


def test_cdp_command_completion_without_verified_playback_is_not_success(
    monkeypatch,
) -> None:
    backend = CdpAppleMusicBackend(timeout_seconds=8)
    clock = iter((0.0, 0.0, 9.0))
    monkeypatch.setattr("app.music_control.cdp.time.monotonic", lambda: next(clock))
    monkeypatch.setattr("app.music_control.cdp.time.sleep", lambda seconds: None)

    def evaluate(operation, expression, args=None):
        if operation == "playback_command_dispatch":
            return {"dispatched": True}
        return {
            "token": args["commandToken"], "phase": "command_complete",
            "failed": False, "isPlaying": False,
            "current": {"id": "song.previous", "title": "Previous", "artist": "Artist"},
        }

    monkeypatch.setattr(backend, "_evaluate", evaluate)

    with pytest.raises(TimeoutError, match="command_phase=command_complete") as error:
        backend.play_song("song.expected")

    message = str(error.value)
    assert '"firstObserved"' in message
    assert '"lastObserved"' in message
    assert '"current":{"id":"song.previous"' in message


def test_cdp_set_queue_rejection_is_distinct_from_pending(monkeypatch) -> None:
    backend = CdpAppleMusicBackend()

    def evaluate(operation, expression, args=None):
        if operation == "playback_command_dispatch":
            return {"dispatched": True}
        return {
            "token": args["commandToken"], "phase": "command_failed",
            "failurePhase": "set_queue", "failed": True,
            "isPlaying": False, "current": None,
        }

    monkeypatch.setattr(backend, "_evaluate", evaluate)

    with pytest.raises(MusicControlError, match="failed at set_queue") as error:
        backend.play_song("song.expected")

    assert error.value.code == "playback_command_failed"


def test_tool_failure_context_never_claims_playback_success() -> None:
    backend = FakeAppleMusicBackend()
    backend.songs = ()
    tool = MusicControlTool(
        RuleBasedMusicActionParser(), AppleMusicPwaController(backend),
    )

    result = tool.run("아이묭 마리골드 틀어줘")

    assert not result.ok
    context = tool.build_llm_context(result)
    assert "아무것도 재생하지 않았다" in context or "찾지 못했다" in context
    assert "성공했다고 말하지" in context


def test_ambiguous_title_only_context_requests_artist_clarification() -> None:
    backend = FakeAppleMusicBackend()
    backend.songs = (
        MusicItem("song.one", "수평선", "Artist One", search_rank=0),
        MusicItem("song.two", "수평선", "Artist Two", search_rank=1),
    )
    tool = MusicControlTool(
        RuleBasedMusicActionParser(), AppleMusicPwaController(backend),
    )

    result = tool.run("수평선 틀어줘")
    context = tool.build_llm_context(result)

    assert not result.ok
    assert result.data["reason"] == "ambiguous"
    assert result.data["requested_title"] == "수평선"
    assert "어느 가수 노래인지" in context


def test_success_context_separates_requested_display_from_canonical_metadata() -> None:
    backend = FakeAppleMusicBackend()
    tool = MusicControlTool(
        RuleBasedMusicActionParser(), AppleMusicPwaController(
            backend, candidate_semantics=FakeCandidateSemantics(
                song_index=0, rewrites=(CatalogQueryVariant(title="Marigold"),),
            ),
        ),
    )

    result = tool.run("마리골드 틀어줘")
    context = tool.build_llm_context(result)

    assert result.ok
    assert result.data["requested_title"] == "마리골드"
    assert result.data["canonical_title"] == "Marigold"
    assert result.data["canonical_artist"] == "aimyon"
    assert result.data["response_display_title"] == "마리골드"
    assert result.data["response_display_artist"] == "aimyon"
    assert "response_display_title" in context
    assert "canonical metadata는 검증 전용" in context


def test_success_display_uses_verified_artist_not_raw_request_surface() -> None:
    backend = FakeAppleMusicBackend()
    tool = MusicControlTool(
        RuleBasedMusicActionParser(), AppleMusicPwaController(
            backend, candidate_semantics=FakeCandidateSemantics(song_index=0),
        ),
    )

    result = tool.run("아이묭의 마리골드 틀어줘")

    assert result.ok
    assert result.data["requested_artist"] == "아이묭"
    assert result.data["response_display_artist"] == "aimyon"


def test_now_playing_without_requested_title_uses_canonical_display() -> None:
    backend = FakeAppleMusicBackend()
    tool = MusicControlTool(
        RuleBasedMusicActionParser(), AppleMusicPwaController(backend),
    )

    result = tool.run("지금 무슨 노래야?")

    assert result.ok
    assert "requested_title" not in result.data
    assert result.data["response_display_title"] == "Marigold"
    assert result.data["response_display_artist"] == "aimyon"
