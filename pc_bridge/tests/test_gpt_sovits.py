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
        groq_timeout_seconds=1, groq_temperature=0.45,
        groq_max_completion_tokens=256, groq_max_retries=2,
        groq_reasoning_effort="low",
        tts_engine="gpt_sovits", serial_enabled=False,
        serial_port="COM3", serial_baud=115200, idle_sleep_seconds=300,
        voice_reference_dir=references, generated_dir=tmp_path / "generated",
        cache_dir=tmp_path / "cache",
        gpt_sovits_api_url="http://127.0.0.1:9880/tts",
        gpt_sovits_prompt_language="ja", gpt_sovits_prompt_text="",
        gpt_sovits_text_language="ko",
        gpt_sovits_timeout_seconds=120,
        gpt_sovits_speed_factor=1.0,
        gpt_sovits_text_split_method="cut1", gpt_sovits_seed=42,
        gpt_sovits_primary_reference="", gpt_sovits_use_aux_references=True,
        gpt_sovits_aux_references="",
        gpt_sovits_top_k=5, gpt_sovits_top_p=0.85, gpt_sovits_temperature=0.7,
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
    assert captured["body"]["speed_factor"] == 1.0
    assert captured["body"]["text_split_method"] == "cut1"
    assert captured["body"]["seed"] == 42
    assert captured["body"]["top_k"] == 5
    assert captured["body"]["top_p"] == 0.85
    assert captured["body"]["temperature"] == 0.7
    assert len(captured["body"]["aux_ref_audio_paths"]) == 1
    assert captured["timeout"] == 120


def test_primary_reference_can_be_isolated(tmp_path: Path, monkeypatch) -> None:
    settings = make_settings(tmp_path)
    values = dict(settings.__dict__)
    values["gpt_sovits_primary_reference"] = "two.wav"
    values["gpt_sovits_use_aux_references"] = False
    captured = {}

    def fake_urlopen(request, timeout):
        captured["body"] = json.loads(request.data.decode("utf-8"))
        return FakeResponse()

    monkeypatch.setattr("app.tts.urlopen", fake_urlopen)
    engine = create_tts_engine(Settings(**values))
    engine.synthesize(LlmResult(reply="안녕하세요."))
    assert captured["body"]["ref_audio_path"].endswith("two.wav")
    assert captured["body"]["aux_ref_audio_paths"] == []


def test_typographic_english_punctuation_is_normalized_for_tts(tmp_path: Path, monkeypatch) -> None:
    settings = make_settings(tmp_path)
    captured = {}

    def fake_urlopen(request, timeout):
        captured["body"] = json.loads(request.data.decode("utf-8"))
        return FakeResponse()

    monkeypatch.setattr("app.tts.urlopen", fake_urlopen)
    create_tts_engine(settings).synthesize(
        LlmResult(reply="Python is a high‑level language, and it’s readable…")
    )
    assert captured["body"]["text"] == "Python is a high-level language, and it's readable..."


def test_common_llm_unicode_symbols_are_normalized_for_tts(tmp_path: Path, monkeypatch) -> None:
    settings = make_settings(tmp_path)
    captured = {}

    def fake_urlopen(request, timeout):
        captured["body"] = json.loads(request.data.decode("utf-8"))
        return FakeResponse()

    monkeypatch.setattr("app.tts.urlopen", fake_urlopen)
    create_tts_engine(settings).synthesize(
        LlmResult(reply="ＡＰＩ\u200b • A→B, 3×4=12, x≤10, 거의≈같아 😊 日本語")
    )
    assert captured["body"]["text"] == "API - A -> B, 3 x 4=12, x <= 10, 거의 ~ 같아 日本語"


def test_explicit_auxiliary_reference_is_selected(tmp_path: Path, monkeypatch) -> None:
    settings = make_settings(tmp_path)
    values = dict(settings.__dict__)
    values["gpt_sovits_primary_reference"] = "two.wav"
    values["gpt_sovits_aux_references"] = "one.wav"
    captured = {}

    def fake_urlopen(request, timeout):
        captured["body"] = json.loads(request.data.decode("utf-8"))
        return FakeResponse()

    monkeypatch.setattr("app.tts.urlopen", fake_urlopen)
    create_tts_engine(Settings(**values)).synthesize(LlmResult(reply="테스트입니다."))
    assert captured["body"]["ref_audio_path"].endswith("two.wav")
    assert len(captured["body"]["aux_ref_audio_paths"]) == 1
    assert captured["body"]["aux_ref_audio_paths"][0].endswith("one.wav")


def test_gpt_sovits_is_the_default_backend(monkeypatch) -> None:
    monkeypatch.delenv("TTS_ENGINE", raising=False)
    assert Settings.from_env().tts_engine == "gpt_sovits"
