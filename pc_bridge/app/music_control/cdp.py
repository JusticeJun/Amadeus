from __future__ import annotations

import json
import os
from pathlib import Path
import re
import subprocess
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
        return self._evaluate("""
          return {authorized: !!mk.isAuthorized,
            navigation: !!document.querySelector('a[href*="/library/"]'),
            player: !!document.querySelector('[data-testid="player-bar"]')};
        """)

    def search_songs(self, query: str) -> tuple[MusicItem, ...]:
        data = self._evaluate("""
          const result = await mk.api.music(`/v1/catalog/${mk.storefrontId}/search`,
            {term:args.query, types:'songs', limit:25});
          const rows = result?.data?.results?.songs?.data || [];
          return rows.map((song, rank) => ({id:song.id, title:song.attributes?.name || '',
            artist:song.attributes?.artistName || '', album:song.attributes?.albumName || '',
            recordingId:song.attributes?.isrc || '', searchRank:rank}));
        """, {"query": query})
        return _music_items(data)

    def search_artists(self, query: str) -> tuple[MusicItem, ...]:
        data = self._evaluate("""
          const result = await mk.api.music(`/v1/catalog/${mk.storefrontId}/search`,
            {term:args.query, types:'artists', limit:15});
          const rows = result?.data?.results?.artists?.data || [];
          return rows.map((item, rank) => ({id:item.id, title:'',
            artist:item.attributes?.name || '', searchRank:rank}));
        """, {"query": query})
        return _music_items(data)

    def search_playlists(self, query: str) -> tuple[PlaylistItem, ...]:
        data = self._evaluate("""
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
        data = self._evaluate("""
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
        data = self._evaluate("""
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
            "if(String(mk.nowPlayingItem?.id||'')!==String(args.id)) "
            "await mk.setQueue({song:args.id}); await mk.play();",
            {"id": item_id, "expectedNowPlayingId": item_id, "expectedPlaying": True},
        )

    def play_artist(self, item_id: str) -> MusicItem:
        return self._command(
            "await mk.setQueue({artist:args.id}); await mk.play();",
            {"id": item_id, "expectedPlaying": True},
        )

    def load_playlist(self, playlist_id: str) -> tuple[MusicItem, ...]:
        data = self._evaluate("""
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

    def play_queue_item(self, index: int) -> MusicItem:
        return self._command(
            "const item=mk.queue.items[args.index]; if(!item) throw new Error('queue item missing'); "
            "args.expectedNowPlayingId=String(item.id); "
            "await mk.changeToMediaItem(item); await mk.play();",
            {"index": index, "expectedPlaying": True},
        )

    def play(self) -> MusicItem:
        return self._command("await mk.play();", {"expectedPlaying": True})

    def pause(self) -> MusicItem:
        return self._command("mk.pause();", {"expectedPlaying": False})

    def next(self) -> MusicItem:
        return self._command(
            "args.previousNowPlayingId=String(mk.nowPlayingItem?.id||''); "
            "await mk.skipToNextItem();",
        )

    def previous(self) -> MusicItem:
        return self._command(
            "args.previousNowPlayingId=String(mk.nowPlayingItem?.id||''); "
            "await mk.skipToPreviousItem();",
        )

    def now_playing(self) -> MusicItem:
        return _music_item(self._evaluate("return snapshot();"))

    def _command(self, statement: str, args: dict[str, object] | None = None) -> MusicItem:
        data = self._evaluate(statement + """
          const metadataDeadline=Date.now()+READY_TIMEOUT_MS;
          while(Date.now()<metadataDeadline){
            const current=snapshot();
            const expectedItemReady=!args.expectedNowPlayingId || (current &&
              String(current.id)===String(args.expectedNowPlayingId));
            const changedItemReady=!args.previousNowPlayingId || (current &&
              String(current.id)!==String(args.previousNowPlayingId));
            const playbackReady=typeof args.expectedPlaying!=='boolean' ||
              mk.isPlaying===args.expectedPlaying;
            if(current && expectedItemReady && changedItemReady && playbackReady) return current;
            await wait(100);
          }
          throw new Error('[AMADEUS:metadata_unavailable] now-playing metadata unavailable');
        """, args)
        return _music_item(data)

    def _evaluate(self, body: str, args: dict[str, object] | None = None):
        target = self._target()
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
        try:
            ws = create_connection(
                target["webSocketDebuggerUrl"], timeout=self._timeout,
            )
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
            raise TimeoutError("Apple Music CDP timed out") from exc
        finally:
            if ws is not None:
                try:
                    ws.close()
                except (OSError, WebSocketException):
                    pass
        raise TimeoutError("Apple Music CDP did not return a result")

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
