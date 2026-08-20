from pathlib import Path

from app.conversation import ConversationManager
from app.llm import LlmClient
from app.models import LlmResult
from app.tts import TtsEngine


class SequenceInput:
    def __init__(self, values: list[str]) -> None:
        self._values = iter(values)

    def read(self) -> str | None:
        return next(self._values, None)


class EchoThenRewriteLlm(LlmClient):
    def __init__(self) -> None:
        self.calls = 0

    def complete(self, user_text, history):
        self.calls += 1
        if self.calls == 1:
            return LlmResult(reply=user_text)
        return LlmResult(reply="좋게 들리셨다니 다행이에요.", emotion="happy")


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
