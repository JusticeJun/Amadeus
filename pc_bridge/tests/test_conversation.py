from pathlib import Path

from app.conversation import ConversationManager
from app.llm import LlmClient
from app.models import LlmResult
from app.routing import RuleBasedSemanticRouter
from app.tts import TtsEngine
from app.tools import ToolExecutor
from app.tools.base import ToolResult


class SequenceInput:
    def __init__(self, values: list[str]) -> None:
        self._values = iter(values)

    def read(self, on_activity=None) -> str | None:
        del on_activity
        return next(self._values, None)


class ActivityInput(SequenceInput):
    def read(self, on_activity=None) -> str | None:
        value = next(self._values, None)
        if value not in {None, "/quit"} and on_activity is not None:
            on_activity(True)
            on_activity(False)
        return value


class EchoThenRewriteLlm(LlmClient):
    def __init__(self) -> None:
        self.calls = 0

    def complete(self, user_text, history):
        self.calls += 1
        if self.calls == 1:
            return LlmResult(reply=user_text)
        return LlmResult(reply="좋게 들리셨다니 다행이에요.", emotion="happy")


class CapturingLlm(LlmClient):
    def __init__(self) -> None:
        self.history = []

    def complete(self, user_text, history):
        self.history = history
        return LlmResult(reply="오늘은 비가 올 수 있겠네.")


class RepeatingLlm(LlmClient):
    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.histories = []

    def complete(self, user_text, history):
        self.histories.append(history)
        return LlmResult(reply=self.reply)


class RepliesLlm(LlmClient):
    def __init__(self, replies: list[str]) -> None:
        self.replies = iter(replies)
        self.histories = []

    def complete(self, user_text, history):
        self.histories.append(history)
        return LlmResult(reply=next(self.replies))


class StubWeatherTool:
    name = "weather"

    def run(self, user_text: str) -> ToolResult:
        return ToolResult("weather", True, {"location": "Busan", "temperature_c": 27})

    def build_llm_context(self, result: ToolResult) -> str:
        return f'검증된 외부 정보: {{"temperature_c":{result.data["temperature_c"]}}}'


class StubMediaTool:
    name = "media_control"

    def run(self, user_text: str) -> ToolResult:
        return ToolResult("media_control", True, {"action": "playing"})

    def build_llm_context(self, result: ToolResult) -> str:
        return '검증된 실행 결과: {"action":"playing"}'


class FailedSideEffectTool:
    name = "pc_control"
    side_effecting = True

    def run(self, user_text: str) -> ToolResult:
        return ToolResult("pc_control", False, {}, "unsupported_action")

    def build_llm_context(self, result: ToolResult) -> str:
        return "검증된 실행 결과: 요청한 조작은 실행되지 않았다."


class AmbiguousSideEffectTool(FailedSideEffectTool):
    name = "music_control"

    def run(self, user_text: str) -> ToolResult:
        return ToolResult("music_control", False, {"reason": "ambiguous"}, "ambiguous")

    def build_llm_context(self, result: ToolResult) -> str:
        return "같은 제목의 곡이 여러 개라 재생하지 않았다. 어느 가수인지 물어본다."


class PartialMusicTool(FailedSideEffectTool):
    name = "music_control"

    def run(self, user_text: str) -> ToolResult:
        return ToolResult("music_control", False, {
            "status": "partial_failure",
            "actions": [
                {"type": "pause", "status": "success"},
                {"type": "play_song", "status": "failed", "reason": "no_match"},
            ],
            "reason": "no_match",
        }, "no matching song")

    def build_llm_context(self, result: ToolResult) -> str:
        return (
            "검증된 순차 실행 결과: pause는 success, play_song은 failed다. "
            "성공과 실패를 구분하고 실행되지 않은 재생을 약속하지 않는다."
        )


class CapturingTts(TtsEngine):
    def __init__(self) -> None:
        self.results: list[LlmResult] = []

    def synthesize(self, result: LlmResult) -> Path | None:
        self.results.append(result)
        return None


class CapturingSerial:
    def __init__(self) -> None:
        self.states: list[str] = []

    def send_state(self, emotion: str) -> None:
        self.states.append(emotion)


def test_direct_echo_is_regenerated_before_tts() -> None:
    llm = EchoThenRewriteLlm()
    tts = CapturingTts()
    serial = CapturingSerial()
    manager = ConversationManager(
        SequenceInput(["지금은 아주 좋은데?", "/quit"]),
        llm,
        tts,
        serial,
        neutral_hold_seconds=0,
    )

    manager.run()

    assert llm.calls == 2
    assert tts.results[0].reply == "좋게 들리셨다니 다행이에요."
    assert "happy" in serial.states
    assert "listening" not in serial.states


def test_input_activity_controls_listening_state() -> None:
    serial = CapturingSerial()
    manager = ConversationManager(
        ActivityInput(["지금은 아주 좋은데?", "/quit"]),
        EchoThenRewriteLlm(),
        CapturingTts(),
        serial,
        neutral_hold_seconds=0,
    )

    manager.run()

    listening_index = serial.states.index("listening")
    assert serial.states[listening_index + 1] == "neutral"


def test_tool_facts_are_passed_to_llm_without_becoming_the_reply() -> None:
    llm = CapturingLlm()
    tts = CapturingTts()
    manager = ConversationManager(
        SequenceInput(["오늘 날씨 어때?", "/quit"]), llm, tts, CapturingSerial(),
        neutral_hold_seconds=0,
        semantic_router=RuleBasedSemanticRouter({"weather": lambda request: True}),
        tool_executor=ToolExecutor([StubWeatherTool()]),
    )

    manager.run()

    context = [item.content for item in llm.history if item.role == "system"]
    assert any('"temperature_c":27' in item for item in context)
    assert tts.results[0].reply == "오늘은 비가 올 수 있겠네."


def test_multiple_tool_contexts_are_passed_to_one_character_llm_call() -> None:
    llm = CapturingLlm()
    manager = ConversationManager(
        SequenceInput(["비 오는지 확인하고 음악 틀어줘.", "/quit"]),
        llm,
        CapturingTts(),
        CapturingSerial(),
        neutral_hold_seconds=0,
        semantic_router=RuleBasedSemanticRouter({
            "weather": lambda request: True,
            "media_control": lambda request: True,
        }),
        tool_executor=ToolExecutor([StubWeatherTool(), StubMediaTool()]),
    )

    manager.run()

    context = [item.content for item in llm.history if item.role == "system"]
    assert any('"temperature_c":27' in item for item in context)
    assert any('"action":"playing"' in item for item in context)


def test_planning_required_request_executes_no_tools_and_informs_character_llm() -> None:
    llm = CapturingLlm()
    weather = StubWeatherTool()
    media = StubMediaTool()
    router = RuleBasedSemanticRouter(
        {"weather": lambda request: True, "media_control": lambda request: True},
        planning_detectors=(lambda request, capabilities: "conditional_dependency",),
    )
    manager = ConversationManager(
        SequenceInput(["비 오면 음악 틀어줘.", "/quit"]),
        llm,
        CapturingTts(),
        CapturingSerial(),
        neutral_hold_seconds=0,
        semantic_router=router,
        tool_executor=ToolExecutor([weather, media]),
    )

    manager.run()

    assert not any(item.content.startswith("검증된 외부 정보") for item in llm.history)
    assert not any(item.content.startswith("검증된 실행 결과") for item in llm.history)
    assert any("어떤 기능도 실행하지 않았다" in item.content for item in llm.history)


def test_failed_side_effect_cannot_be_reported_as_success() -> None:
    llm = RepeatingLlm("볼륨을 30으로 줄였어!")
    tts = CapturingTts()
    manager = ConversationManager(
        SequenceInput(["볼륨 30으로 줄여봐", "/quit"]),
        llm,
        tts,
        CapturingSerial(),
        neutral_hold_seconds=0,
        semantic_router=RuleBasedSemanticRouter({"pc_control": lambda request: True}),
        tool_executor=ToolExecutor([FailedSideEffectTool()]),
    )

    manager.run()

    assert len(llm.histories) == 2
    assert any(
        "실제 조작은 실패했고 변경된 것은 없다" in item.content
        for item in llm.histories[-1]
    )
    assert tts.results[0].reply == "그 요청은 실행되지 않았어. 아직은 제대로 처리할 수 없어."


def test_unrouted_side_effect_cannot_be_reported_as_success() -> None:
    llm = RepeatingLlm("잠깐 정지했어.")
    tts = CapturingTts()
    manager = ConversationManager(
        SequenceInput(["일시정지 좀 해줘", "/quit"]),
        llm,
        tts,
        CapturingSerial(),
        neutral_hold_seconds=0,
    )

    manager.run()

    assert len(llm.histories) == 2
    assert any(
        "검증된 실제 조작 성공 결과가 없다" in item.content
        for item in llm.histories[-1]
    )
    assert tts.results[0].reply == "그 요청은 실행되지 않았어. 아직은 제대로 처리할 수 없어."


def test_ambiguous_side_effect_allows_truthful_clarification() -> None:
    llm = RepeatingLlm("수평선이라는 곡이 여러 개 있어. 어느 가수 노래를 말하는 거야?")
    tts = CapturingTts()
    manager = ConversationManager(
        SequenceInput(["수평선 틀어줘", "/quit"]),
        llm,
        tts,
        CapturingSerial(),
        neutral_hold_seconds=0,
        semantic_router=RuleBasedSemanticRouter({"music_control": lambda request: True}),
        tool_executor=ToolExecutor([AmbiguousSideEffectTool()]),
    )

    manager.run()

    assert len(llm.histories) == 1
    assert tts.results[0].reply == "수평선이라는 곡이 여러 개 있어. 어느 가수 노래를 말하는 거야?"


def test_planning_reply_cannot_claim_success_or_offer_unrelated_advice() -> None:
    llm = RepeatingLlm("볼륨을 줄일 수는 없지만, 시원한 음료 한잔 어때?")
    tts = CapturingTts()
    router = RuleBasedSemanticRouter(
        {"weather": lambda request: True, "pc_control": lambda request: True},
        planning_detectors=(lambda request, capabilities: "conditional_dependency",),
    )
    manager = ConversationManager(
        SequenceInput(["밖에 더우면 볼륨 내려줘", "/quit"]),
        llm,
        tts,
        CapturingSerial(),
        neutral_hold_seconds=0,
        semantic_router=router,
        tool_executor=ToolExecutor([StubWeatherTool(), FailedSideEffectTool()]),
    )

    manager.run()

    assert len(llm.histories) == 2
    assert tts.results[0].reply == "그렇게 조건을 걸어서 조작하는 건 아직 못 해."


def test_partial_music_result_reports_success_and_failure_without_future_claim() -> None:
    llm = RepliesLlm([
        "지금 노래는 정지했어. 이제 크리스마스송을 재생할게!",
        "지금 노래는 일시정지했는데, 크리스마스송은 재생하지 못했어.",
    ])
    tts = CapturingTts()
    manager = ConversationManager(
        SequenceInput([
            "지금 재생중인 노래 일시정지하고 백넘버의 크리스마스송 틀어줘",
            "/quit",
        ]),
        llm,
        tts,
        CapturingSerial(),
        neutral_hold_seconds=0,
        semantic_router=RuleBasedSemanticRouter({"music_control": lambda request: True}),
        tool_executor=ToolExecutor([PartialMusicTool()]),
    )

    manager.run()

    assert len(llm.histories) == 2
    assert tts.results[0].reply == (
        "지금 노래는 일시정지했는데, 크리스마스송은 재생하지 못했어."
    )


def test_current_weather_claim_without_verified_source_is_blocked() -> None:
    llm = RepeatingLlm("오늘 밖은 진짜 덥고 기온이 35도야.")
    tts = CapturingTts()
    manager = ConversationManager(
        SequenceInput(["오늘도 밖에 많이 덥나?", "/quit"]),
        llm,
        tts,
        CapturingSerial(),
        neutral_hold_seconds=0,
    )

    manager.run()

    assert len(llm.histories) == 2
    assert tts.results[0].reply == "지금 날씨는 실제로 확인하지 못했어."


def test_current_weather_claim_with_verified_weather_result_is_allowed() -> None:
    llm = RepeatingLlm("오늘 밖은 27도고 비는 안 와.")
    tts = CapturingTts()
    manager = ConversationManager(
        SequenceInput(["오늘 밖에 덥나?", "/quit"]),
        llm,
        tts,
        CapturingSerial(),
        neutral_hold_seconds=0,
        semantic_router=RuleBasedSemanticRouter({"weather": lambda request: True}),
        tool_executor=ToolExecutor([StubWeatherTool()]),
    )

    manager.run()

    assert len(llm.histories) == 1
    assert tts.results[0].reply == "오늘 밖은 27도고 비는 안 와."


def test_live_data_guard_does_not_block_ordinary_today_conversation() -> None:
    llm = RepeatingLlm("오늘은 코딩 공부 얘기나 더 해보자.")
    tts = CapturingTts()
    manager = ConversationManager(
        SequenceInput(["오늘 뭐 할까?", "/quit"]),
        llm,
        tts,
        CapturingSerial(),
        neutral_hold_seconds=0,
    )

    manager.run()

    assert len(llm.histories) == 1
    assert tts.results[0].reply == "오늘은 코딩 공부 얘기나 더 해보자."
