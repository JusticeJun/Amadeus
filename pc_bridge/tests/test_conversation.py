from pathlib import Path

from app.conversation import ConversationManager
from app.llm import LlmClient
from app.models import LlmResult
from app.tts import TtsEngine
from app.tools.base import ToolResult
from app.tools.router import ToolRouter


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


class StubWeatherTool:
    name = "weather"

    def matches(self, user_text: str) -> bool:
        return "날씨" in user_text

    def run(self, user_text: str) -> ToolResult:
        return ToolResult("weather", True, {"location": "Busan", "temperature_c": 27})


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
        neutral_hold_seconds=0, tool_router=ToolRouter([StubWeatherTool()]),
    )

    manager.run()

    context = [item.content for item in llm.history if item.role == "system"]
    assert any('"temperature_c":27' in item for item in context)
    assert tts.results[0].reply == "오늘은 비가 올 수 있겠네."
