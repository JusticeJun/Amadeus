from __future__ import annotations

import json

from ..music_control import (
    MusicAction, MusicActionParser, MusicController, MusicSequenceExecutor,
)
from .base import ToolResult


class MusicControlTool:
    name = "music_control"
    side_effecting = True

    def __init__(self, parser: MusicActionParser, controller: MusicController) -> None:
        self._parser = parser
        self._executor = MusicSequenceExecutor(controller)

    def run(self, user_text: str) -> ToolResult:
        parsed = self._parser.parse(user_text)
        if not parsed.ok or parsed.sequence is None:
            return ToolResult(
                self.name, False, {"reason": parsed.error_code},
                f"music action parsing failed: {parsed.error_code}",
            )
        sequence_result = self._executor.execute(parsed.sequence)
        action_data = [
            self._action_data(result.action, result.ok, result.data)
            for result in sequence_result.results
        ]
        data: dict[str, object] = {
            "status": sequence_result.status,
            "actions": action_data,
        }
        if len(action_data) == 1:
            data.update(action_data[0])
        failed = [result for result in sequence_result.results if not result.ok]
        if failed:
            data["reason"] = str(failed[0].data.get("reason") or "execution_failed")
        errors = "; ".join(result.error for result in failed if result.error)
        return ToolResult(self.name, sequence_result.ok, data, errors)

    @staticmethod
    def _action_data(
        action: MusicAction,
        ok: bool,
        result_data: dict[str, object],
    ) -> dict[str, object]:
        status = "success" if ok else str(result_data.get("status") or "failed")
        data = {
            "type": action.action_type.value,
            "requested_action": action.action_type.value,
            "status": status,
            **({"requested_title": action.title} if action.title else {}),
            **({"requested_artist": action.artist} if action.artist else {}),
            **({"requested_playlist": action.playlist} if action.playlist else {}),
            **result_data,
        }
        now_playing = data.get("now_playing")
        if ok and isinstance(now_playing, dict):
            canonical_title = str(now_playing.get("title") or "")
            canonical_artist = str(now_playing.get("artist") or "")
            data.update({
                "canonical_title": canonical_title,
                "canonical_artist": canonical_artist,
                "response_display_title": action.title or canonical_title,
                "response_display_artist": action.artist or canonical_artist,
            })
        return data

    def build_llm_context(self, result: ToolResult) -> str:
        if not result.ok:
            reason = str(result.data.get("reason") or "execution_failed")
            actions = result.data.get("actions")
            if isinstance(actions, list) and len(actions) > 1:
                compact = json.dumps(result.data, ensure_ascii=False, separators=(",", ":"))
                return (
                    f"검증된 Apple Music 순차 실행 결과: {compact}\n"
                    "status가 success인 동작만 실제로 완료됐다. failed는 실패했고 skipped는 "
                    "앞선 실패 때문에 실행하지 않았다. 각 결과를 빠짐없이 구분해 자연스럽게 "
                    "말하고, 전체가 성공했다고 하거나 실행되지 않은 동작을 곧 실행할 것처럼 "
                    "말하지 않는다. 내부 상태명이나 구현 용어는 사용자에게 말하지 않는다."
                )
            title_only_ambiguity = (
                reason == "ambiguous"
                and result.data.get("type") == "play_song"
                and result.data.get("requested_title")
                and not result.data.get("requested_artist")
            )
            safe = {
                "ambiguous": "요청과 일치하는 음악이나 플레이리스트가 여러 개라 임의로 재생하지 않았다.",
                "not_found": "요청과 정확히 일치하는 음악이나 플레이리스트를 찾지 못했다.",
                "no_match": "요청과 합리적으로 일치하는 곡을 찾지 못해 아무것도 재생하지 않았다.",
                "metadata_mismatch": "재생 결과가 요청과 달라 성공으로 처리하지 않았다.",
                "pwa_unavailable": "Apple Music을 지금 제어할 수 없어 아무것도 재생하지 않았다.",
                "dom_not_ready": "Apple Music 화면이 아직 준비되지 않아 아무것도 재생하지 않았다.",
                "musickit_unavailable": "Apple Music 재생 기능이 준비되지 않아 아무것도 재생하지 않았다.",
                "player_not_ready": "Apple Music 플레이어가 아직 준비되지 않아 아무것도 재생하지 않았다.",
                "authorization_unavailable": "Apple Music 로그인 상태를 확인할 수 없어 아무것도 재생하지 않았다.",
                "authorization_required": "Apple Music 로그인이 필요해 아무것도 재생하지 않았다.",
                "backend_unavailable": "Apple Music 연결이 준비되지 않아 아무것도 재생하지 않았다.",
            }.get(reason, "요청한 음악 동작은 실행되지 않았다.")
            if title_only_ambiguity:
                safe = (
                    f"'{result.data['requested_title']}'라는 제목과 일치하는 곡이 여러 개라 "
                    "아무 곡도 재생하지 않았다. 어느 가수 노래인지 짧게 물어본다."
                )
            return (
                f"검증된 Apple Music 실행 결과: {safe} "
                "성공했다고 말하지 말고 내부 오류나 구현 용어 없이 짧고 자연스럽게 답한다."
            )
        compact = json.dumps(result.data, ensure_ascii=False, separators=(",", ":"))
        return (
            f"검증된 Apple Music 실행 결과: {compact}\n"
            "이 결과에 있는 동작과 곡만 성공했다고 말하고 다른 실행을 지어내지 않는다. "
            "사용자에게 곡명이나 아티스트를 말할 때는 response_display_title과 "
            "response_display_artist를 우선 사용한다. canonical metadata는 검증 전용이며, "
            "response_display 값과 다르면 canonical 표기를 사용자 응답에서 읽지 않는다."
        )
