from pathlib import Path

from app.conversation import ConversationManager
from app.llm import LlmClient
from app.models import LlmResult
from app.music_control import (
    MusicAction, MusicActionParseResult, MusicActionResult, MusicActionSequence,
    MusicActionType, MusicSemanticInterpreter, RuleBasedMusicActionParser,
)
from app.routing import RuleBasedSemanticRouter
from app.semantic_llm import SemanticLlmMetrics, SemanticLlmResponse
from app.tools import MusicControlTool, ToolExecutor


class SequenceInput:
    def __init__(self, values):
        self.values = iter(values)

    def read(self, on_activity=None):
        del on_activity
        return next(self.values, None)


class RepliesLlm(LlmClient):
    def __init__(self, replies):
        self.replies = iter(replies)

    def complete(self, user_text, history):
        del user_text, history
        return LlmResult(reply=next(self.replies))


class RepeatingLlm(LlmClient):
    def __init__(self, reply):
        self.reply = reply

    def complete(self, user_text, history):
        del user_text, history
        return LlmResult(reply=self.reply)


class CapturingTts:
    def __init__(self):
        self.results = []

    def synthesize(self, result):
        self.results.append(result)
        return Path("unused.wav")

    def play(self, path):
        del path


class CapturingSerial:
    def send_state(self, emotion):
        del emotion


class ClarifyingMusicController:
    def __init__(self):
        self.actions = []

    def execute(self, action):
        self.actions.append(action)
        if not action.artist:
            return MusicActionResult(
                action, False, {"reason": "ambiguous"}, "multiple songs matched",
            )
        return MusicActionResult(action, True, {
            "now_playing": {"title": action.title, "artist": action.artist},
        })


class PlaylistClarifyingController:
    def __init__(self):
        self.actions = []
        self.options = ("Focus Mix", "Live Set")

    def execute(self, action):
        self.actions.append(action)
        if action.action_type is not MusicActionType.PLAY_PLAYLIST:
            raise AssertionError("unexpected action")
        if action.playlist not in self.options:
            return MusicActionResult(action, False, {
                "reason": "ambiguous", "candidate_options": list(self.options),
            }, "multiple personal playlists matched")
        return MusicActionResult(action, True, {
            "playlist": action.playlist,
            "now_playing": {"title": "Track", "artist": "Performer"},
        })


class CorrectableTrackController:
    def __init__(self, failures=1):
        self.actions = []
        self.failures = failures

    def execute(self, action):
        self.actions.append(action)
        if len(self.actions) <= self.failures:
            return MusicActionResult(
                action, False, {"reason": "no_match"}, "no reasonable candidate",
            )
        return MusicActionResult(action, True, {
            "now_playing": {"title": action.title, "artist": action.artist},
        })


def manager_for(inputs, replies, controller):
    tts = CapturingTts()
    manager = ConversationManager(
        SequenceInput([*inputs, "/quit"]), replies, tts, CapturingSerial(),
        neutral_hold_seconds=0,
        semantic_router=RuleBasedSemanticRouter({
            "music_control": lambda request: "틀어줘" in request.text,
        }),
        tool_executor=ToolExecutor([
            MusicControlTool(RuleBasedMusicActionParser(), controller),
        ]),
    )
    return manager, tts


def test_clarification_completes_pending_request_and_executes():
    controller = ClarifyingMusicController()
    manager, tts = manager_for(
        ["수평선 틀어줘", "백넘버야"],
        RepliesLlm(["어느 가수야?", "백넘버의 수평선을 재생했어."]), controller,
    )
    manager.run()
    assert len(controller.actions) == 2
    assert controller.actions[-1].title == "수평선"
    assert controller.actions[-1].artist == "백넘버"
    assert tts.results[-1].reply == "백넘버의 수평선을 재생했어."


def test_unrelated_turn_consumes_pending_without_later_execution():
    controller = ClarifyingMusicController()
    manager, _ = manager_for(
        ["수평선 틀어줘", "오늘 날씨 알려줘", "백넘버야"],
        RepliesLlm(["어느 가수야?", "확인하지 못했어.", "응, 백넘버구나."]), controller,
    )
    manager.run()
    assert len(controller.actions) == 1


def test_invalid_clarification_state_does_not_execute():
    controller = ClarifyingMusicController()
    tool = MusicControlTool(RuleBasedMusicActionParser(), controller)
    result = tool.continue_clarification(
        {"kind": "missing_entity", "field": "artist", "action": {"type": "pause"}},
        "백넘버야",
    )
    assert result is None
    assert controller.actions == []


def test_playlist_options_create_bounded_clarification_and_exact_continuation():
    controller = PlaylistClarifyingController()
    tool = MusicControlTool(RuleBasedMusicActionParser(), controller)

    ambiguous = tool.run("추천 플레이리스트 틀어줘")
    pending = ambiguous.data["clarification"]
    continued = tool.continue_clarification(pending, "Focus Mix")

    assert not ambiguous.ok
    assert pending == {
        "kind": "candidate_selection",
        "action": {"type": "play_playlist"},
        "candidate_options": ["Focus Mix", "Live Set"],
    }
    assert "Focus Mix" in tool.build_llm_context(ambiguous)
    assert "Live Set" in tool.build_llm_context(ambiguous)
    assert continued is not None and continued.ok
    assert controller.actions[-1].playlist == "Focus Mix"


def test_playlist_ordinal_selection_is_bounded_and_unrelated_reply_does_not_execute():
    controller = PlaylistClarifyingController()
    tool = MusicControlTool(RuleBasedMusicActionParser(), controller)
    pending = tool.run("추천 플레이리스트 틀어줘").data["clarification"]

    assert tool.continue_clarification(pending, "오늘 날씨는 어때?") is None
    assert len(controller.actions) == 1
    continued = tool.continue_clarification(pending, "두 번째")
    assert continued is not None and continued.ok
    assert controller.actions[-1].playlist == "Live Set"


def test_playlist_natural_followup_uses_only_bounded_semantic_candidates():
    class SemanticClient:
        def __init__(self):
            self.requests = []

        def complete(self, request):
            self.requests.append(request)
            return SemanticLlmResponse(
                {"status": "match", "candidate_index": 1}, SemanticLlmMetrics(),
            )

    client = SemanticClient()
    controller = PlaylistClarifyingController()
    parser = MusicSemanticInterpreter(RuleBasedMusicActionParser(), client)
    tool = MusicControlTool(parser, controller)
    pending = {
        "kind": "candidate_selection",
        "action": {"type": "play_playlist"},
        "candidate_options": ["Focus Mix", "Live Set"],
    }

    continued = tool.continue_clarification(pending, "라이브 셋으로 해줘")

    assert continued is not None and continued.ok
    assert controller.actions[-1].playlist == "Live Set"
    assert client.requests[0].input["candidates"] == [
        {"index": 0, "name": "Focus Mix"},
        {"index": 1, "name": "Live Set"},
    ]


def test_playlist_semantic_ambiguity_reasks_without_side_effect():
    class AmbiguousParser(RuleBasedMusicActionParser):
        def select_playlist_candidate(self, surface, options):
            del surface, options
            return type("Judgment", (), {"approved": False, "index": -1})()

    controller = PlaylistClarifyingController()
    tool = MusicControlTool(AmbiguousParser(), controller)
    pending = tool.run("추천 플레이리스트 틀어줘").data["clarification"]

    continued = tool.continue_clarification(pending, "그 가수 걸로 해줘")

    assert continued is not None and not continued.ok
    assert continued.data["clarification"] == pending
    assert len(controller.actions) == 1


def test_invalid_playlist_selection_state_does_not_execute():
    controller = PlaylistClarifyingController()
    tool = MusicControlTool(RuleBasedMusicActionParser(), controller)

    result = tool.continue_clarification({
        "kind": "candidate_selection",
        "action": {"type": "play_playlist"},
        "candidate_options": ["invented"],
    }, "invented")

    assert result is None
    assert controller.actions == []


def test_playlist_generic_failure_is_replaced_with_candidate_question():
    controller = PlaylistClarifyingController()
    manager, tts = manager_for(
        ["추천 플레이리스트 틀어줘"], RepeatingLlm("아직 처리할 수 없어."), controller,
    )

    manager.run()

    reply = tts.results[0].reply
    assert reply != "아직 처리할 수 없어."
    assert all(option in reply for option in controller.options)
    assert "?" in reply


def test_unexecuted_future_music_promise_is_blocked():
    controller = ClarifyingMusicController()
    manager, tts = manager_for(
        ["백넘버야"], RepeatingLlm("지금 바로 재생할게."), controller,
    )
    manager.run()
    assert tts.results[0].reply != "지금 바로 재생할게."
    assert controller.actions == []


def test_failed_track_request_accepts_bounded_artist_title_correction_followup():
    controller = CorrectableTrackController()
    manager, tts = manager_for(
        ["surface artist의 surface title 틀어줘", "corrected artist의 corrected title"],
        RepliesLlm(["그 곡은 재생하지 못했어.", "corrected title이 재생돼!"]),
        controller,
    )

    manager.run()

    assert len(controller.actions) == 2
    assert controller.actions[-1].artist == "corrected artist"
    assert controller.actions[-1].title == "corrected title"
    assert tts.results[-1].reply == "corrected title이 재생돼!"


def test_ambiguous_track_correction_reasks_without_side_effect():
    class AmbiguousCorrectionParser(RuleBasedMusicActionParser):
        def continue_track_correction(self, pending, text, history):
            del pending, text, history
            return MusicActionParseResult(error_code="ambiguous")

    controller = CorrectableTrackController()
    tool = MusicControlTool(AmbiguousCorrectionParser(), controller)
    failed = tool.run("surface artist의 surface title 틀어줘")

    continued = tool.continue_clarification(
        failed.data["clarification"], "maybe artist의 maybe title",
    )

    assert continued is not None and not continued.ok
    assert continued.data["reason"] == "ambiguous"
    assert continued.data["clarification"] == failed.data["clarification"]
    assert len(controller.actions) == 1


def test_unexecuted_present_tense_music_success_claim_is_blocked():
    controller = CorrectableTrackController()
    manager, tts = manager_for(
        ["artist의 title"], RepeatingLlm("title이 재생돼!"), controller,
    )

    manager.run()

    assert tts.results[0].reply == "그 요청은 실행되지 않았어. 아직은 제대로 처리할 수 없어."
    assert controller.actions == []


def test_failed_music_tool_present_tense_success_claim_is_blocked():
    controller = CorrectableTrackController()
    manager, tts = manager_for(
        ["artist의 title 틀어줘"], RepeatingLlm("title이 재생돼!"), controller,
    )

    manager.run()

    assert tts.results[0].reply == "그 요청은 실행되지 않았어. 아직은 제대로 처리할 수 없어."
    assert len(controller.actions) == 1


def test_malformed_first_action_still_creates_unfixed_correction_state():
    class MalformedFirstParser:
        def parse(self, text):
            del text
            return MusicActionParseResult(MusicActionSequence((MusicAction(
                MusicActionType.PLAY_SONG, title="artist surface", source_query="artist surface",
            ),)))

    controller = CorrectableTrackController()
    tool = MusicControlTool(MalformedFirstParser(), controller)

    failed = tool.run("artist surface verb-like title 틀어줘")
    pending = failed.data["clarification"]
    corrected = tool.continue_clarification(pending, "correct artist의 correct title")

    assert pending["action"] == {
        "type": "play_song", "title": "artist surface", "artist": "",
    }
    assert corrected is not None and corrected.ok
    assert controller.actions[-1].artist == "correct artist"
    assert controller.actions[-1].title == "correct title"


def test_corrected_failed_track_can_be_replayed_by_bounded_anaphor():
    controller = CorrectableTrackController(failures=2)
    manager, _ = manager_for(
        [
            "first artist의 first title 틀어줘",
            "correct artist의 correct title",
            "아니 그 곡을 틀어달라고",
        ],
        RepeatingLlm("재생했어."),
        controller,
    )

    manager.run()

    assert len(controller.actions) == 3
    assert [(action.artist, action.title) for action in controller.actions[-2:]] == [
        ("correct artist", "correct title"),
        ("correct artist", "correct title"),
    ]


def test_anaphor_with_incomplete_pending_referent_is_ambiguous():
    controller = CorrectableTrackController()
    tool = MusicControlTool(RuleBasedMusicActionParser(), controller)
    pending = {
        "kind": "track_correction",
        "action": {"type": "play_song", "title": "uncertain", "artist": ""},
    }

    result = tool.continue_clarification(pending, "그 곡 틀어줘")

    assert result is not None and not result.ok
    assert result.data["reason"] == "ambiguous"
    assert controller.actions == []


def test_anaphoric_track_request_without_pending_context_never_executes():
    controller = CorrectableTrackController(failures=0)
    tool = MusicControlTool(RuleBasedMusicActionParser(), controller)

    result = tool.run("그 곡 틀어줘")

    assert not result.ok
    assert result.data["reason"] == "ambiguous"
    assert controller.actions == []
