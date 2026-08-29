from __future__ import annotations

import json
import re
from ..music_control import (
    MusicAction, MusicActionParser, MusicController, MusicSemanticInterpreter,
    MusicActionSequence, MusicActionType, MusicSequenceExecutor,
)
from ..models import ChatMessage
from .base import ToolResult


class MusicControlTool:
    name = "music_control"
    side_effecting = True

    def __init__(
        self,
        parser: MusicActionParser | MusicSemanticInterpreter,
        controller: MusicController,
    ) -> None:
        self._parser = parser
        self._executor = MusicSequenceExecutor(controller)

    def run(self, user_text: str) -> ToolResult:
        return self.run_with_context(user_text, ())

    def run_with_context(
        self, user_text: str, history: tuple[ChatMessage, ...],
    ) -> ToolResult:
        if _is_track_replay_reference(user_text):
            return ToolResult(
                self.name, False, {"reason": "ambiguous"},
                "track reference has no capability-owned referent",
            )
        interpret = getattr(self._parser, "interpret", None)
        parsed = interpret(user_text, history) if interpret else self._parser.parse(user_text)
        if not parsed.ok or parsed.sequence is None:
            detail = getattr(getattr(self._parser, "metrics", None), "last_error", "")
            return ToolResult(
                self.name, False, {"reason": parsed.error_code},
                f"music action parsing failed: {parsed.error_code}"
                + (f" ({detail})" if detail else ""),
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
        if (
            len(action_data) == 1
            and data.get("reason") == "ambiguous"
            and action_data[0].get("type") == "play_song"
            and action_data[0].get("requested_title")
            and not action_data[0].get("requested_artist")
        ):
            data["clarification"] = {
                "kind": "missing_entity", "field": "artist",
                "action": {"type": "play_song", "title": action_data[0]["requested_title"]},
            }
        _attach_track_correction(data, action_data)
        options = data.get("candidate_options")
        if (
            len(action_data) == 1
            and data.get("reason") == "ambiguous"
            and action_data[0].get("type") == "play_playlist"
            and action_data[0].get("requested_playlist")
            and isinstance(options, list)
            and 1 < len(options) <= 5
            and all(isinstance(option, str) and option.strip() for option in options)
        ):
            data["clarification"] = {
                "kind": "candidate_selection",
                "action": {"type": "play_playlist"},
                "candidate_options": options,
            }
        errors = "; ".join(result.error for result in failed if result.error)
        return ToolResult(self.name, sequence_result.ok, data, errors)

    def continue_clarification(
        self, pending: dict[str, object], user_text: str,
        history: tuple[ChatMessage, ...] = (),
    ) -> ToolResult | None:
        action_data = pending.get("action")
        if pending.get("kind") == "track_correction":
            if not isinstance(action_data, dict):
                return None
            if _is_track_replay_reference(user_text):
                artist = action_data.get("artist")
                title = action_data.get("title")
                if not isinstance(artist, str) or not artist.strip() \
                        or not isinstance(title, str) or not title.strip():
                    return ToolResult(self.name, False, {
                        "reason": "ambiguous", "clarification": pending,
                    }, "track reference remains ambiguous")
                return self._execute_continuation(MusicActionSequence((MusicAction(
                    action_type=MusicActionType.PLAY_SONG,
                    title=title.strip(), artist=artist.strip(),
                    source_query=f"{artist.strip()} {title.strip()}", artist_explicit=True,
                ),)))
            if not _is_bounded_entity_answer(user_text):
                return None
            semantic_continuation = getattr(
                self._parser, "continue_track_correction", None,
            )
            if semantic_continuation is not None:
                parsed = semantic_continuation(pending, user_text, history)
                if not parsed.ok or parsed.sequence is None:
                    return ToolResult(self.name, False, {
                        "reason": "ambiguous", "clarification": pending,
                    }, "track correction remains ambiguous")
                return self._execute_continuation(parsed.sequence)
            corrected = _plain_track_correction(user_text)
            if corrected is None:
                return None
            artist, title = corrected
            return self._execute_continuation(MusicActionSequence((MusicAction(
                action_type=MusicActionType.PLAY_SONG, title=title, artist=artist,
                source_query=f"{artist} {title}", artist_explicit=True,
            ),)))
        if (
            pending.get("kind") == "candidate_selection"
            and isinstance(action_data, dict)
            and action_data.get("type") == "play_playlist"
        ):
            options = pending.get("candidate_options")
            selected = _selected_candidate(options, user_text)
            if selected is None and isinstance(options, list):
                semantic_selection = getattr(
                    self._parser, "select_playlist_candidate", None,
                )
                if semantic_selection is not None:
                    judgment = semantic_selection(user_text, tuple(options))
                    if (
                        getattr(judgment, "approved", False)
                        and isinstance(getattr(judgment, "index", None), int)
                        and 0 <= judgment.index < len(options)
                    ):
                        selected = options[judgment.index]
                    else:
                        return ToolResult(self.name, False, {
                            "reason": "ambiguous",
                            "clarification": pending,
                            "candidate_options": options,
                        }, "playlist selection remains ambiguous")
            if selected is None:
                return None
            sequence = MusicActionSequence((MusicAction(
                action_type=MusicActionType.PLAY_PLAYLIST, playlist=selected,
                source_query=selected,
            ),))
            return self._execute_continuation(sequence)
        if (
            pending.get("kind") != "missing_entity" or pending.get("field") != "artist"
            or not isinstance(action_data, dict) or action_data.get("type") != "play_song"
            or not isinstance(action_data.get("title"), str)
            or not _is_bounded_entity_answer(user_text)
        ):
            return None
        semantic_continuation = getattr(self._parser, "continue_clarification", None)
        if semantic_continuation is not None:
            parsed = semantic_continuation(pending, user_text, history)
            if not parsed.ok or parsed.sequence is None:
                return None
            sequence = parsed.sequence
        else:
            artist = _plain_entity_answer(user_text)
            if not artist:
                return None
            sequence = MusicActionSequence((MusicAction(
                action_type=MusicActionType.PLAY_SONG, title=action_data["title"],
                artist=artist, source_query=f'{artist} {action_data["title"]}',
                artist_explicit=True,
            ),))
        return self._execute_continuation(sequence)

    def _execute_continuation(self, sequence: MusicActionSequence) -> ToolResult:
        sequence_result = self._executor.execute(sequence)
        result = sequence_result.results[0]
        action_result = self._action_data(result.action, result.ok, result.data)
        data = {"status": sequence_result.status, "actions": [action_result], **action_result}
        if not result.ok:
            data["reason"] = str(result.data.get("reason") or "execution_failed")
        _attach_track_correction(data, [action_result])
        return ToolResult(self.name, result.ok, data, result.error)

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
                "response_display_artist": canonical_artist or action.artist,
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
            options = result.data.get("candidate_options")
            if reason == "ambiguous" and isinstance(options, list) and options:
                names = ", ".join(str(item) for item in options[:5])
                safe = (
                    f"실제 personal playlist 후보 {names}가 서로 경쟁해 "
                    "아무 것도 재생하지 않았다. 후보 이름을 그대로 사용해 "
                    "사용자에게 어느 playlist인지 짧게 물어본다."
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


def _is_bounded_entity_answer(text: str) -> bool:
    value = text.strip().rstrip(".!?~")
    if not value or len(value) > 80 or len(value.split()) > 5:
        return False
    compact = "".join(value.casefold().split())
    if compact in {"안녕", "몰라", "모르겠어", "취소", "그만"}:
        return False
    blocked = ("틀어", "재생", "멈춰", "날씨", "온도", "알려", "해줘", "할래", "해줄래")
    return not any(term in compact for term in blocked)


def _plain_entity_answer(text: str) -> str:
    """Conservative no-LLM fallback for a bare entity, not conversational prose."""
    value = text.strip().rstrip(".!?~")
    if not _is_bounded_entity_answer(value) or len(value.split()) != 1:
        return ""
    return value[:-1] if value.endswith("야") and len(value) > 1 else value


def _plain_track_correction(text: str) -> tuple[str, str] | None:
    value = text.strip().rstrip(".!?~")
    match = re.fullmatch(r"(?P<artist>.+?)의\s*(?P<title>.+)", value)
    if not match:
        return None
    artist, title = match.group("artist").strip(), match.group("title").strip()
    return (artist, title) if artist and title else None


def _is_track_replay_reference(text: str) -> bool:
    compact = "".join(text.strip().rstrip(".!?~").casefold().split())
    compact = re.sub(r"^(?:아니|그러니까|내말은)", "", compact)
    return bool(re.fullmatch(
        r"(?:그곡|그노래)(?:을|를)?(?:틀어줘|재생해줘|틀어달라고|재생해달라고)",
        compact,
    ))


def _attach_track_correction(
    data: dict[str, object], action_data: list[dict[str, object]],
) -> None:
    if (
        len(action_data) != 1
        or data.get("reason") != "no_match"
        or action_data[0].get("type") != "play_song"
        or not action_data[0].get("requested_title")
    ):
        return
    data["clarification"] = {
        "kind": "track_correction",
        "action": {
            "type": "play_song",
            "title": action_data[0]["requested_title"],
            "artist": action_data[0].get("requested_artist") or "",
        },
    }


def _selected_candidate(options: object, text: str) -> str | None:
    if (
        not isinstance(options, list) or not 1 < len(options) <= 5
        or not all(isinstance(option, str) and option.strip() for option in options)
    ):
        return None
    compact = "".join(text.strip().rstrip(".!?~").casefold().split())
    exact = [option for option in options if "".join(option.casefold().split()) == compact]
    if len(exact) == 1:
        return exact[0]
    ordinals = {
        "첫번째": 0, "첫째": 0, "1번": 0,
        "두번째": 1, "둘째": 1, "2번": 1,
        "세번째": 2, "셋째": 2, "3번": 2,
        "네번째": 3, "넷째": 3, "4번": 3,
        "다섯번째": 4, "다섯째": 4, "5번": 4,
    }
    index = ordinals.get(compact)
    return options[index] if index is not None and index < len(options) else None
