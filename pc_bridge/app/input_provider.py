from typing import Protocol


class InputProvider(Protocol):
    def read(self) -> str | None: ...


class TextInputProvider:
    def read(self) -> str | None:
        try:
            value = input("나 > ").strip()
        except (EOFError, KeyboardInterrupt):
            return None
        return value or ""


class VoiceInputProvider:
    def read(self) -> str | None:
        raise NotImplementedError("INMP441 음성 입력은 마이크 도착 후 구현합니다.")

