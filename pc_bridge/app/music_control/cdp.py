from __future__ import annotations

import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from urllib.error import URLError
from urllib.parse import urlparse
from urllib.request import urlopen

from websocket import WebSocketException, WebSocketTimeoutException, create_connection

from .controller import (
    MusicControlError, MusicItem, PersonalMusicItem, PersonalMusicSnapshot,
    PlaylistItem, PlaylistSnapshot,
)


_APPLE_URL = "https://music.apple.com/kr/"


def _search_diagnostic(
    query: str, backend_status: str, raw_song_count: int,
    extracted_count: int, error_category: str,
) -> None:
    if os.environ.get("AMADEUS_MUSIC_DIAGNOSTICS") == "1":
        print(
            "[music_retrieval_diagnostic] search_result:" + json.dumps({
                "query": query, "backend_status": backend_status,
                "raw_song_count": raw_song_count,
                "extracted_count": extracted_count,
                "error_category": error_category,
            }, ensure_ascii=False, separators=(",", ":")),
            file=sys.stderr,
        )


class CdpAppleMusicBackend:
    """Control only the isolated Apple Music PWA through a localhost CDP endpoint."""

    def __init__(
        self,
        *,
        port: int = 9223,
        timeout_seconds: float = 8.0,
        chrome_path: Path = Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
        profile_dir: Path | None = None,
    ) -> None:
        self._port = port
        self._timeout = timeout_seconds
        self._chrome_path = chrome_path
        local = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
        self._profile_dir = profile_dir or local / "Amadeus" / "AppleMusicChrome"

    def selector_health(self) -> dict[str, bool]:
        return self._evaluate("health_check", """
          return {authorized: !!mk.isAuthorized,
            navigation: !!document.querySelector('a[href*="/library/"]'),
            player: !!document.querySelector('[data-testid="player-bar"]')};
        """)

    def search_songs(self, query: str) -> tuple[MusicItem, ...]:
        try:
            data = self._evaluate("catalog_song_search", """
              const result = await mk.api.music(`/v1/catalog/${mk.storefrontId}/search`,
                {term:args.query, types:'songs', limit:25});
              const rows = result?.data?.results?.songs?.data || [];
              const diagnostics=globalThis.__amadeusCatalogDiagnostics ||= new Map();
              rows.forEach(song=>diagnostics.set(String(song.id),{
                catalogId:String(song.id||''),type:String(song.type||''),
                playParamsId:String(song.attributes?.playParams?.id||''),
                playParamsKind:String(song.attributes?.playParams?.kind||''),
                storefrontId:String(mk.storefrontId||''),
                title:String(song.attributes?.name||''),
                artist:String(song.attributes?.artistName||'')}));
              return rows.map((song, rank) => ({id:song.id, title:song.attributes?.name || '',
                artist:song.attributes?.artistName || '', album:song.attributes?.albumName || '',
                recordingId:song.attributes?.isrc || '', searchRank:rank}));
            """, {"query": query})
        except (MusicControlError, OSError, TimeoutError) as exc:
            _search_diagnostic(query, "error", 0, 0, type(exc).__name__)
            raise
        items = _music_items(data)
        _search_diagnostic(query, "ok", len(data or ()), len(items), "")
        return items

    def search_artists(self, query: str) -> tuple[MusicItem, ...]:
        data = self._evaluate("catalog_artist_search", """
          const result = await mk.api.music(`/v1/catalog/${mk.storefrontId}/search`,
            {term:args.query, types:'artists', limit:15});
          const rows = result?.data?.results?.artists?.data || [];
          return rows.map((item, rank) => ({id:item.id, title:'',
            artist:item.attributes?.name || '', searchRank:rank}));
        """, {"query": query})
        return _music_items(data)

    def search_playlists(self, query: str) -> tuple[PlaylistItem, ...]:
        data = self._evaluate("catalog_playlist_search", """
          const result = await mk.api.music(`/v1/catalog/${mk.storefrontId}/search`,
            {term:args.query, types:'playlists', limit:25});
          const rows = result?.data?.results?.playlists?.data || [];
          return rows.map(item => ({id:item.id, name:item.attributes?.name || ''}));
        """, {"query": query})
        return tuple(
            PlaylistItem(str(item.get("id", "")), str(item.get("name", "")))
            for item in (data or ()) if item.get("id") and item.get("name")
        )

    def list_playlists(self) -> PlaylistSnapshot:
        data = self._evaluate("library_playlist_list", """
          const response=await mk.api.music('/v1/me/library/playlists',{limit:100});
          const rows=response?.data?.data || [];
          return {items:rows.map(item => ({id:item.id,
            name:item.attributes?.name || ''})), partial:!!response?.data?.next,
            warning:response?.data?.next ? 'playlist_page_limit' : ''};
        """)
        items = tuple(PlaylistItem(str(item["id"]), str(item["name"])) for item in data["items"])
        if not items:
            raise MusicControlError("not_found", "no library playlists were visible")
        return PlaylistSnapshot(items, bool(data.get("partial")), str(data.get("warning", "")))

    def personal_songs(self) -> PersonalMusicSnapshot:
        data = self._evaluate("personal_music_index", """
          const MAX_PLAYLISTS=500, MAX_TRACKS=5000;
          const readPages=async(path, params, limit)=>{
            const rows=[]; let next=path; let first=true; let partial=false;
            while(next && rows.length<limit){
              const response=await mk.api.music(next, first ? params : undefined);
              const page=response?.data || {};
              rows.push(...(page.data || [])); next=page.next || ''; first=false;
            }
            if(next) partial=true;
            return {rows:rows.slice(0,limit), partial};
          };
          const songs=new Map();
          const add=(item, source)=>{
            const attrs=item.attributes || {};
            const catalogId=String(attrs.playParams?.catalogId || item.id || '');
            if(!catalogId) return;
            const current=songs.get(catalogId) || {id:catalogId,
              title:attrs.name || '', artist:attrs.artistName || '',
              album:attrs.albumName || '', recordingId:attrs.isrc || '',
              searchRank:100, inLibrary:false, playlistCount:0};
            if(source==='library') current.inLibrary=true;
            if(source==='playlist') current.playlistCount+=1;
            songs.set(catalogId,current);
          };
          let partial=false;
          const safeRead=async(path,params,limit)=>{
            try { return await readPages(path,params,limit); }
            catch(error) { return {rows:[],partial:true}; }
          };
          const [library,playlists]=await Promise.all([
            safeRead('/v1/me/library/songs',{limit:100},MAX_TRACKS),
            safeRead('/v1/me/library/playlists',{limit:100},MAX_PLAYLISTS),
          ]);
          library.rows.forEach(item=>add(item,'library'));
          partial ||= library.partial || playlists.partial;
          const trackPages=await Promise.all(playlists.rows.map(playlist=>
            safeRead(`/v1/me/library/playlists/${playlist.id}/tracks`,
              {limit:100},MAX_TRACKS)));
          trackPages.forEach(tracks=>{
            tracks.rows.forEach(item=>add(item,'playlist'));
            partial ||= tracks.partial;
          });
          if(songs.size>MAX_TRACKS) partial=true;
          return {items:Array.from(songs.values()), partial,
            warning:partial ? 'personal_index_partial' : ''};
        """)
        items = tuple(
            PersonalMusicItem(
                _music_item(item), bool(item.get("inLibrary")),
                int(item.get("playlistCount", 0)),
            )
            for item in data.get("items", ())
        )
        return PersonalMusicSnapshot(
            items, bool(data.get("partial")), str(data.get("warning", "")),
        )

    def play_song(self, item_id: str) -> MusicItem:
        return self._command(
            "const itemSnapshot=item=>item ? {id:String(item.id||''),"
            "type:String(item.type||item.kind||''),"
            "playParamsId:String(item.playParams?.id||''),"
            "playParamsKind:String(item.playParams?.kind||''),"
            "title:String(item.title||''),artist:String(item.artistName||'')} : null; "
            "state.resolvedMetadata=globalThis.__amadeusCatalogDiagnostics?.get(String(args.id))"
            "||{catalogId:String(args.id),storefrontId:String(mk.storefrontId||'')}; "
            "state.queueOptions={songs:[String(args.id)]}; "
            "state.before={current:itemSnapshot(mk.nowPlayingItem),isPlaying:!!mk.isPlaying}; "
            "if(String(mk.nowPlayingItem?.id||'')!==String(args.id)){ "
            "state.phase='set_queue_started'; await mk.setQueue({songs:[args.id]}); "
            "state.setQueueResolved=true; state.phase='set_queue_resolved'; "
            "state.queueContainsExpected=Array.from(mk.queue.items||[]).some(item=>"
            "String(item.id)===String(args.id)); "
            "state.queueAfterSetQueue=Array.from(mk.queue.items||[]).slice(0,5).map(itemSnapshot); "
            "state.currentAfterSetQueue=itemSnapshot(mk.nowPlayingItem); "
            "if(!state.queueContainsExpected) throw new Error('queue item unavailable'); } "
            "state.phase='play_started'; await mk.play(); state.playResolved=true; "
            "state.phase='play_resolved'; state.currentAfterPlay=itemSnapshot(mk.nowPlayingItem); "
            "state.isPlayingAfterPlay=!!mk.isPlaying;",
            {"id": item_id, "expectedNowPlayingId": item_id, "expectedPlaying": True},
        )

    def play_artist(self, item_id: str) -> MusicItem:
        return self._command(
            "state.phase='set_queue'; await mk.setQueue({artist:args.id}); "
            "state.phase='play'; await mk.play();",
            {"id": item_id, "expectedPlaying": True},
        )

    def load_playlist(self, playlist_id: str) -> tuple[MusicItem, ...]:
        data = self._evaluate("playlist_queue_load", """
          await mk.setQueue({playlist:args.id});
          const queueDeadline=Date.now()+READY_TIMEOUT_MS;
          while(Date.now()<queueDeadline){
            const items=Array.from(mk.queue.items || []);
            if(items.length) return items.map(item => ({id:item.id,
              title:item.title || '', artist:item.artistName || '', album:item.albumName || ''}));
            await wait(100);
          }
          throw new Error('[AMADEUS:queue_unavailable] playlist queue unavailable');
        """, {"id": playlist_id})
        return _music_items(data)

    def playlist_tracks(self, playlist_id: str) -> tuple[MusicItem, ...]:
        data = self._evaluate("library_playlist_tracks", """
          const response=await mk.api.music(
            `/v1/me/library/playlists/${args.id}/tracks`, {limit:100});
          const rows=response?.data?.data || [];
          return rows.map((song, rank) => {
            const attrs=song.attributes || {};
            return {id:String(attrs.playParams?.catalogId || song.id || ''),
              title:attrs.name || '', artist:attrs.artistName || '',
              album:attrs.albumName || '', recordingId:attrs.isrc || '',
              searchRank:rank};
          });
        """, {"id": playlist_id})
        return _music_items(data)

    def play_queue_item(self, index: int) -> MusicItem:
        return self._command(
            "const item=mk.queue.items[args.index]; if(!item) throw new Error('queue item missing'); "
            "state.expectedNowPlayingId=String(item.id); "
            "state.phase='change_media_item'; await mk.changeToMediaItem(item); "
            "state.phase='play'; await mk.play();",
            {"index": index, "expectedPlaying": True},
        )

    def play(self) -> MusicItem:
        return self._command(
            "state.phase='play'; await mk.play();", {"expectedPlaying": True},
        )

    def pause(self) -> MusicItem:
        return self._command("state.phase='pause'; mk.pause();", {"expectedPlaying": False})

    def next(self) -> MusicItem:
        return self._command(
            "state.previousNowPlayingId=String(mk.nowPlayingItem?.id||''); "
            "state.phase='skip_next'; await mk.skipToNextItem();",
        )

    def previous(self) -> MusicItem:
        return self._command(
            "state.previousNowPlayingId=String(mk.nowPlayingItem?.id||''); "
            "state.phase='skip_previous'; await mk.skipToPreviousItem();",
        )

    def now_playing(self) -> MusicItem:
        return _music_item(self._evaluate("now_playing", "return snapshot();"))

    def _command(self, statement: str, args: dict[str, object] | None = None) -> MusicItem:
        command_args = dict(args or {})
        command_args["commandToken"] = str(time.monotonic_ns())
        self._evaluate("playback_command_dispatch", """
          const state={token:args.commandToken,phase:'dispatched',failed:false};
          globalThis.__amadeusPlaybackCommand=state;
          void (async()=>{try{
            BODY
            state.phase='command_complete';
          }catch(error){state.failed=true;state.failurePhase=state.phase;
            state.phase='command_failed';}})();
          return {dispatched:true};
        """.replace("BODY", statement), command_args)

        deadline = time.monotonic() + self._timeout
        phase = "dispatched"
        first_observed = None
        last_observed = None
        while time.monotonic() < deadline:
            data = self._evaluate("playback_verification", """
              const state=globalThis.__amadeusPlaybackCommand;
              const current=snapshot();
              return {token:state?.token||'',phase:state?.phase||'state_missing',
                failed:!!state?.failed,isPlaying:!!mk.isPlaying,current,
                failurePhase:state?.failurePhase||'',
                queueContainsExpected:state?.queueContainsExpected===true,
                expectedNowPlayingId:state?.expectedNowPlayingId||'',
                previousNowPlayingId:state?.previousNowPlayingId||'',
                diagnostics:{resolvedMetadata:state?.resolvedMetadata||null,
                  queueOptions:state?.queueOptions||null,before:state?.before||null,
                  setQueueResolved:state?.setQueueResolved===true,
                  queueAfterSetQueue:state?.queueAfterSetQueue||[],
                  currentAfterSetQueue:state?.currentAfterSetQueue||null,
                  playResolved:state?.playResolved===true,
                  currentAfterPlay:state?.currentAfterPlay||null,
                  isPlayingAfterPlay:state?.isPlayingAfterPlay===true}};
            """, command_args)
            observed = {
                "phase": data.get("phase"), "isPlaying": data.get("isPlaying"),
                "current": data.get("current"), "diagnostics": data.get("diagnostics"),
            }
            if first_observed is None:
                first_observed = observed
            last_observed = observed
            if data.get("token") != command_args["commandToken"]:
                raise MusicControlError(
                    "playback_state_lost", "playback command state was replaced",
                )
            phase = str(data.get("phase", "unknown"))
            if data.get("failed"):
                failure_phase = str(data.get("failurePhase", "unknown"))
                raise MusicControlError(
                    "playback_command_failed",
                    f"playback command failed at {failure_phase}",
                )
            current = data.get("current")
            expected_id = data.get("expectedNowPlayingId") or command_args.get(
                "expectedNowPlayingId",
            )
            previous_id = data.get("previousNowPlayingId") or command_args.get(
                "previousNowPlayingId",
            )
            expected_item_ready = not expected_id or (
                current and str(current.get("id", ""))
                == str(expected_id)
            )
            changed_item_ready = not previous_id or (
                current and str(current.get("id", ""))
                != str(previous_id)
            )
            playback_ready = "expectedPlaying" not in command_args or (
                bool(data.get("isPlaying")) is bool(command_args["expectedPlaying"])
            )
            if current and expected_item_ready and changed_item_ready and playback_ready:
                if os.environ.get("AMADEUS_MUSIC_DIAGNOSTICS") == "1":
                    print(
                        "[music_playback_diagnostic] " + json.dumps(
                            observed, ensure_ascii=False, separators=(",", ":"),
                        ),
                        file=sys.stderr,
                    )
                return _music_item(current)
            time.sleep(0.1)
        raise TimeoutError(
            "Apple Music CDP timed out "
            "(operation=playback_verification, stage=state_poll, "
            f"timeout=deadline_exceeded, recovery=not_attempted, command_phase={phase}, "
            f"diagnostic={json.dumps({'firstObserved': first_observed, 'lastObserved': last_observed}, ensure_ascii=False, separators=(',', ':'))})"
        )

    def _evaluate(
        self, operation: str, body: str, args: dict[str, object] | None = None,
    ):
        try:
            target = self._target()
        except TimeoutError as exc:
            raise TimeoutError(
                self._timeout_message(operation, "target_discovery", "not_attempted"),
            ) from exc
        expression = """
        (async()=>{const args=ARGS;
          const wait=ms=>new Promise(resolve=>setTimeout(resolve,ms));
          const readyDeadline=Date.now()+READY_TIMEOUT_MS;
          let mk=null; let readinessFailure='dom_not_ready';
          while(Date.now()<readyDeadline){
            if(document.readyState!=='interactive' && document.readyState!=='complete'){
              readinessFailure='dom_not_ready';
            } else if(!globalThis.MusicKit || typeof MusicKit.getInstance!=='function'){
              readinessFailure='musickit_unavailable';
            } else {
              mk=MusicKit.getInstance();
              if(!mk){ readinessFailure='player_not_ready'; }
              else if(typeof mk.isAuthorized!=='boolean'){
                mk=null; readinessFailure='authorization_unavailable';
              } else if(!mk.isAuthorized){
                throw new Error('[AMADEUS:authorization_required] Apple Music login required');
              } else if(typeof mk.api?.music!=='function' ||
                typeof mk.setQueue!=='function' || typeof mk.play!=='function' ||
                !document.querySelector('[data-testid="player-bar"]')){
                mk=null; readinessFailure='player_not_ready';
              } else { break; }
            }
            await wait(100);
          }
          if(!mk) throw new Error(`[AMADEUS:${readinessFailure}] Apple Music not ready`);
          const snapshot=()=>{const item=mk.nowPlayingItem; if(!item) return null;
            return {id:item.id,title:item.title||'',artist:item.artistName||'',album:item.albumName||''};};
          BODY
        })()
        """.replace("ARGS", json.dumps(args or {}, ensure_ascii=False)).replace(
            "BODY", body,
        ).replace("READY_TIMEOUT_MS", str(int(self._timeout * 1000)))
        ws = None
        recovery = "not_attempted"
        try:
            try:
                ws = create_connection(
                    target["webSocketDebuggerUrl"], timeout=self._timeout,
                )
            except (OSError, WebSocketException):
                recovery = "target_rediscovery_attempted"
                try:
                    target = self._target()
                    ws = create_connection(
                        target["webSocketDebuggerUrl"], timeout=self._timeout,
                    )
                except TimeoutError as exc:
                    raise TimeoutError(self._timeout_message(
                        operation, "target_rediscovery", recovery,
                        timeout_class="target_wait",
                    )) from exc
            ws.send(json.dumps({
                "id": 1, "method": "Runtime.evaluate",
                "params": {
                    "expression": expression, "awaitPromise": True,
                    "returnByValue": True, "userGesture": True,
                },
            }))
            deadline = time.monotonic() + self._timeout
            while time.monotonic() < deadline:
                message = json.loads(ws.recv())
                if message.get("id") != 1:
                    continue
                result = message.get("result", {}).get("result", {})
                if result.get("subtype") == "error" or "exceptionDetails" in message.get("result", {}):
                    details = message.get("result", {}).get("exceptionDetails", {})
                    exception = details.get("exception", {})
                    description = exception.get("description") or details.get("text")
                    message_text = description or result.get("description", "CDP error")
                    match = re.search(r"\[AMADEUS:([a-z_]+)\]", message_text)
                    raise MusicControlError(
                        match.group(1) if match else "backend_error", message_text,
                    )
                if "value" not in result:
                    raise MusicControlError("backend_error", "CDP returned no value")
                return result.get("value")
        except WebSocketTimeoutException as exc:
            stage = "connect" if ws is None else "runtime_evaluation"
            raise TimeoutError(self._timeout_message(
                operation, stage, recovery,
                timeout_class=(
                    "websocket_connect" if ws is None else "websocket_receive"
                ),
            )) from exc
        finally:
            if ws is not None:
                try:
                    ws.close()
                except (OSError, WebSocketException):
                    pass
        raise TimeoutError(
            self._timeout_message(operation, "runtime_evaluation", recovery),
        )

    @staticmethod
    def _timeout_message(
        operation: str, stage: str, recovery: str,
        *, timeout_class: str = "deadline_exceeded",
    ) -> str:
        return (
            "Apple Music CDP timed out "
            f"(operation={operation}, stage={stage}, timeout={timeout_class}, "
            f"recovery={recovery})"
        )

    def _target(self) -> dict[str, object]:
        try:
            targets = self._read_targets()
        except (OSError, URLError):
            self._launch()
        else:
            page = _ready_apple_music_page(targets)
            if page:
                return page
            if not _apple_music_pages(targets):
                self._launch()
        deadline = time.monotonic() + self._timeout
        while time.monotonic() < deadline:
            try:
                page = _ready_apple_music_page(self._read_targets())
                if page:
                    return page
            except (OSError, URLError):
                pass
            time.sleep(0.2)
        raise TimeoutError("Apple Music PWA did not start")

    def _read_targets(self) -> list[dict[str, object]]:
        with urlopen(f"http://127.0.0.1:{self._port}/json/list", timeout=self._timeout) as response:
            return json.load(response)

    def _launch(self) -> None:
        if not self._chrome_path.is_file():
            raise MusicControlError("chrome_unavailable", "Chrome executable not found")
        self._profile_dir.mkdir(parents=True, exist_ok=True)
        subprocess.Popen(self._launch_args(), shell=False, close_fds=True)

    def _launch_args(self) -> list[str]:
        cdp_origin = f"http://127.0.0.1:{self._port}"
        return [
            str(self._chrome_path),
            "--remote-debugging-address=127.0.0.1",
            f"--remote-debugging-port={self._port}",
            f"--remote-allow-origins={cdp_origin}",
            f"--user-data-dir={self._profile_dir}",
            "--no-first-run",
            "--no-default-browser-check",
            f"--app={_APPLE_URL}",
        ]


def _music_items(data) -> tuple[MusicItem, ...]:
    return tuple(_music_item(item) for item in (data or ()))


def _apple_music_pages(targets) -> list[dict[str, object]]:
    return [
        item for item in targets
        if str(item.get("url", "")).startswith("https://music.apple.com/")
        and item.get("webSocketDebuggerUrl")
    ]


def _ready_apple_music_page(targets) -> dict[str, object] | None:
    for item in _apple_music_pages(targets):
        url = urlparse(str(item.get("url", "")))
        route_parts = [part for part in url.path.split("/") if part]
        title = str(item.get("title", "")).strip()
        if len(route_parts) >= 2 and title:
            return item
    return None


def _music_item(data) -> MusicItem:
    if not data:
        raise MusicControlError("not_found", "now-playing metadata is unavailable")
    return MusicItem(
        str(data.get("id", "")), str(data.get("title", "")),
        str(data.get("artist", "")), str(data.get("album", "")),
        str(data.get("recordingId", "")),
        int(data.get("searchRank", 0)),
    )
