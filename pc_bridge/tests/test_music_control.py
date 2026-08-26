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


class FakeAppleMusicBackend:
    def __init__(self) -> None:
        self.playlists = PlaylistSnapshot((PlaylistItem("p.study", "공부할 때"),))
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


def test_title_only_uses_unique_top_ranked_recording_group() -> None:
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

    assert result.ok
    assert result.data["now_playing"]["id"] == "single"


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


def test_personal_playlist_membership_resolves_title_only_namesake() -> None:
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

    assert result.ok
    assert result.data["now_playing"]["id"] == "second"


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

    result = AppleMusicPwaController(backend).execute(MusicAction(
        MusicActionType.PLAY_SONG, title="Marigold", artist="아이묭",
    ))

    assert result.ok
    assert result.data["now_playing"]["artist"] == "aimyon"

    artist_result = AppleMusicPwaController(backend).execute(MusicAction(
        MusicActionType.PLAY_ARTIST, artist="아이묭",
    ))
    assert artist_result.ok
    assert artist_result.data["artist"] == "aimyon"


def test_cli_title_uses_corroborated_cross_script_catalog_leader() -> None:
    backend = FakeAppleMusicBackend()
    original = MusicItem(
        "original", "Marigold", "aimyon", "Marigold - Single", "ORIGINAL", 0,
    )
    korean_namesake = MusicItem(
        "namesake", "마리골드", "Other Artist", "마리골드 - Single", "OTHER", 2,
    )
    backend.songs = (original, korean_namesake)
    backend.current = original

    result = AppleMusicPwaController(backend).execute(MusicAction(
        MusicActionType.PLAY_SONG, title="마리골드",
    ))

    assert result.ok
    assert result.data["now_playing"]["id"] == "original"


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
    controller = AppleMusicPwaController(backend)

    result = controller.execute(MusicAction(
        MusicActionType.PLAY_SONG, title="마리골드", artist="아이묭",
    ))
    assert result.ok
    assert result.data["now_playing"]["id"] == "original"

    conflicting = replace(original, item_id="different")
    backend.search_songs = lambda query: (
        (original,) + supporting if "아이묭" in query else (conflicting,)
    )
    refused = controller.execute(MusicAction(
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

    result = AppleMusicPwaController(backend).execute(MusicAction(
        MusicActionType.PLAY_SONG, title="수평선", artist="백넘버",
    ))

    assert result.ok
    assert result.data["now_playing"]["id"] == "original"


def test_full_query_title_exact_overrides_ambiguous_first_token_hint() -> None:
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

    result = AppleMusicPwaController(backend).execute(action)

    assert result.ok
    assert result.data["now_playing"]["id"] == "original"


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


def test_resolver_accepts_future_alternate_queries_without_changing_backend_contract() -> None:
    backend = FakeAppleMusicBackend()
    expected = MusicItem("canonical", "Canonical Title", "Canonical Artist")
    queries = []

    def search(query):
        queries.append(query)
        return (expected,) if query == "Canonical Title" else ()

    backend.search_songs = search
    backend.songs = (expected,)
    backend.current = expected
    result = AppleMusicPwaController(backend).execute(MusicAction(
        MusicActionType.PLAY_SONG,
        title="surface title",
        source_query="surface query",
        alternate_queries=("Canonical Title",),
    ))

    assert result.ok
    assert queries[:2] == ["surface query", "Canonical Title"]


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

    with pytest.raises(TimeoutError, match="CDP timed out"):
        backend.selector_health()


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
    assert socket.closed


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

    def fake_evaluate(expression, args=None):
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
        {"id": 1, "result": {"result": {"value": {
            "id": "song.first", "title": "First", "artist": "Artist",
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

    queue = backend.load_playlist("p.goodnight")
    played = backend.play_queue_item(0)

    assert queue[0].item_id == played.item_id == "song.first"
    queue_expression = sockets[0].sent[0]["params"]["expression"]
    play_expression = sockets[1].sent[0]["params"]["expression"]
    assert "queueDeadline" in queue_expression
    assert "await wait(100)" in queue_expression
    assert "metadataDeadline" in play_expression
    assert "expectedNowPlayingId" in play_expression


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
    socket = FakeWebSocket({
        "id": 1,
        "result": {"result": {"value": {
            "id": "song.changed", "title": "Changed", "artist": "Artist",
        }}},
    })
    monkeypatch.setattr(backend, "_target", lambda: {
        "webSocketDebuggerUrl": "ws://127.0.0.1/devtools/page/1",
    })
    monkeypatch.setattr(
        "app.music_control.cdp.create_connection", lambda *args, **kwargs: socket,
    )

    getattr(backend, method)()

    expression = socket.sent[0]["params"]["expression"]
    assert expected_fragment in expression
    assert "playbackReady" in expression
    assert "changedItemReady" in expression


def test_cdp_song_play_is_idempotent_for_current_track(monkeypatch) -> None:
    backend = CdpAppleMusicBackend()
    socket = FakeWebSocket({
        "id": 1,
        "result": {"result": {"value": {
            "id": "1402042897", "title": "Marigold", "artist": "aimyon",
            "album": "Marigold - Single",
        }}},
    })
    monkeypatch.setattr(backend, "_target", lambda: {
        "webSocketDebuggerUrl": "ws://127.0.0.1/devtools/page/1",
    })
    monkeypatch.setattr(
        "app.music_control.cdp.create_connection", lambda *args, **kwargs: socket,
    )

    assert backend.play_song("1402042897").item_id == "1402042897"
    expression = socket.sent[0]["params"]["expression"]
    assert "mk.nowPlayingItem?.id" in expression
    assert "mk.setQueue({song:args.id})" in expression
    assert "expectedNowPlayingId" in expression
    assert "metadataDeadline" in expression
    assert "await wait(100)" in expression
    assert socket.closed


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
        RuleBasedMusicActionParser(), AppleMusicPwaController(backend),
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
