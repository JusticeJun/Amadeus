import json
from pathlib import Path

from app.config import Settings
from app.models import LlmResult
from app.tts import GptSovitsEngine, create_tts_engine


class FakeResponse:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self) -> bytes:
        return b"RIFF" + (b"\x00" * 64)


def make_settings(tmp_path: Path) -> Settings:
    references = tmp_path / "references"
    references.mkdir()
    (references / "one.wav").write_bytes(b"reference-one")
    (references / "two.wav").write_bytes(b"reference-two")
    return Settings(
        mode="mock", groq_api_key="", groq_api_url="", groq_model="mock",
        groq_timeout_seconds=1, tts_engine="gpt_sovits", serial_enabled=False,
        serial_port="COM3", serial_baud=115200, idle_sleep_seconds=300,
        voice_reference_dir=references, generated_dir=tmp_path / "generated",
        cache_dir=tmp_path / "cache",
        gpt_sovits_api_url="http://127.0.0.1:9880/tts",
        gpt_sovits_prompt_language="ja", gpt_sovits_text_language="ko",
        gpt_sovits_timeout_seconds=120,
    )


def test_gpt_sovits_request_uses_cross_language_settings(tmp_path: Path, monkeypatch) -> None:
    settings = make_settings(tmp_path)
    captured = {}

    def fake_urlopen(request, timeout):
        captured["body"] = json.loads(request.data.decode("utf-8"))
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr("app.tts.urlopen", fake_urlopen)
    engine = create_tts_engine(settings)
    assert isinstance(engine, GptSovitsEngine)
    output = engine.synthesize(LlmResult(reply="안녕하세요."))
    assert output.read_bytes().startswith(b"RIFF")
    assert captured["body"]["text_lang"] == "ko"
    assert captured["body"]["prompt_lang"] == "ja"
    assert captured["body"]["prompt_text"] == ""
    assert len(captured["body"]["aux_ref_audio_paths"]) == 1
    assert captured["timeout"] == 120


def test_gpt_sovits_is_the_default_backend(monkeypatch) -> None:
    monkeypatch.delenv("TTS_ENGINE", raising=False)
    assert Settings.from_env().tts_engine == "gpt_sovits"
