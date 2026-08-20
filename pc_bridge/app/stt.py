from abc import ABC, abstractmethod
from pathlib import Path


class SpeechToTextEngine(ABC):
    @abstractmethod
    def transcribe(self, audio_path: Path) -> str: ...


class MockSttEngine(SpeechToTextEngine):
    def transcribe(self, audio_path: Path) -> str:
        return f"mock transcription for {audio_path.name}"


class GroqWhisperEngine(SpeechToTextEngine):
    def transcribe(self, audio_path: Path) -> str:
        del audio_path
        raise NotImplementedError("INMP441 도착 후 Groq Whisper STT를 활성화합니다.")

