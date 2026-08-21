from collections.abc import Callable
from typing import Protocol


InputActivityCallback = Callable[[bool], None]


class InputProvider(Protocol):
    def read(self, on_activity: InputActivityCallback | None = None) -> str | None: ...


class TextInputProvider:
    def read(self, on_activity: InputActivityCallback | None = None) -> str | None:
        # A blocking terminal prompt cannot detect when typing actually starts,
        # so it deliberately leaves the device in neutral instead of pretending
        # to be listening for the entire prompt duration.
        del on_activity
        try:
            value = input("나 > ").strip()
        except (EOFError, KeyboardInterrupt):
            return None
        return value or ""


class VoiceInputProvider:
    def read(self, on_activity: InputActivityCallback | None = None) -> str | None:
        # The future VAD recorder must call on_activity(True) only after speech
        # begins and on_activity(False) as soon as recording ends or times out.
        del on_activity
        raise NotImplementedError("INMP441 음성 입력은 마이크 도착 후 구현합니다.")

