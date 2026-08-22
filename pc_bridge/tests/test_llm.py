from email.message import Message
from io import BytesIO
import json
from pathlib import Path
import urllib.error

import pytest

from app.config import Settings
from app.llm import GroqLlmClient, LlmError
from app.models import ChatMessage


class FakeResponse:
    def __init__(self, payload: dict) -> None:
        self._body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.headers = {"x-request-id": "request-123"}

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self) -> bytes:
        return self._body


def make_settings(tmp_path: Path, retries: int = 2) -> Settings:
    return Settings(
        mode="groq", groq_api_key="secret-test-key",
        groq_api_url="https://api.groq.test/chat/completions",
        groq_model="openai/gpt-oss-20b", groq_timeout_seconds=3,
        groq_temperature=0.45, groq_max_completion_tokens=256,
        groq_max_retries=retries, groq_reasoning_effort="low",
        tts_engine="silent", serial_enabled=False,
        serial_port="COM3", serial_baud=115200, idle_sleep_seconds=300,
        voice_reference_dir=tmp_path, generated_dir=tmp_path,
        cache_dir=tmp_path, gpt_sovits_api_url="", gpt_sovits_prompt_language="ja",
        gpt_sovits_prompt_text="",
        gpt_sovits_text_language="ko", gpt_sovits_timeout_seconds=1,
        gpt_sovits_speed_factor=1.0,
        gpt_sovits_text_split_method="cut1", gpt_sovits_seed=42,
        gpt_sovits_primary_reference="", gpt_sovits_use_aux_references=True,
        gpt_sovits_aux_references="",
        gpt_sovits_top_k=5, gpt_sovits_top_p=0.85, gpt_sovits_temperature=0.7,
    )


def successful_payload() -> dict:
    return {
        "model": "openai/gpt-oss-20b",
        "choices": [{"message": {"content": json.dumps({
            "reply": "도와드릴게요.",
            "emotion": "happy",
            "speech_style": {"speed": 1.0, "pitch": 0, "energy": 1.0},
        }, ensure_ascii=False)}}],
        "usage": {"prompt_tokens": 42, "completion_tokens": 18},
    }


def test_request_uses_strict_schema_and_history(tmp_path: Path) -> None:
    captured = {}

    def opener(request, timeout):
        captured["body"] = json.loads(request.data.decode("utf-8"))
        captured["timeout"] = timeout
        captured["user_agent"] = request.get_header("User-agent")
        return FakeResponse(successful_payload())

    client = GroqLlmClient(make_settings(tmp_path), opener=opener)
    result = client.complete("안녕하세요", [ChatMessage("assistant", "어서 오세요.")])

    assert result.emotion == "happy"
    assert captured["body"]["model"] == "openai/gpt-oss-20b"
    assert captured["body"]["messages"][-1]["content"] == "안녕하세요"
    assert captured["user_agent"] == "Amadeus-PC-Bridge/0.1"
    assert captured["body"]["reasoning_effort"] == "low"
    assert captured["body"]["include_reasoning"] is False
    schema = captured["body"]["response_format"]["json_schema"]
    assert schema["strict"] is True
    emotion_values = schema["schema"]["properties"]["emotion"]["enum"]
    assert "sleep" not in emotion_values
    assert "thinking" not in emotion_values
    assert "answering" in emotion_values
    assert "wondering" in emotion_values
    assert captured["body"]["messages"][-2]["content"] == "어서 오세요."
    assert captured["timeout"] == 3
    assert client.last_metrics.prompt_tokens == 42
    assert client.last_metrics.request_id == "request-123"


def http_error(code: int, retry_after: str = "") -> urllib.error.HTTPError:
    headers = Message()
    if retry_after:
        headers["retry-after"] = retry_after
    return urllib.error.HTTPError(
        "https://api.groq.test", code, "error", headers, BytesIO(b'{"error":"test"}')
    )


def structured_output_error(generated: str) -> urllib.error.HTTPError:
    body = json.dumps({
        "error": {
            "message": "Failed to generate JSON.",
            "code": "json_validate_failed",
            "failed_generation": generated,
        }
    }, ensure_ascii=False).encode("utf-8")
    return urllib.error.HTTPError(
        "https://api.groq.test", 400, "error", Message(), BytesIO(body)
    )


def test_plain_text_from_schema_failure_is_recovered(tmp_path: Path) -> None:
    def opener(request, timeout):
        raise structured_output_error("말씀해 보세요. 듣고 있어요.")

    client = GroqLlmClient(make_settings(tmp_path), opener=opener)
    result = client.complete("말 좀 해 봐", [])

    assert result.reply == "말씀해 보세요. 듣고 있어요."
    assert result.emotion == "answering"
    assert client.last_metrics.recovered_structured_output is True


def test_tool_call_from_schema_failure_is_recovered(tmp_path: Path) -> None:
    failed = json.dumps({
        "name": "amadeus_reply",
        "arguments": {
            "reply": "그 이야기는 여기까지만 할게요. 다른 얘기해요.",
            "emotion": "angry",
            "speech_style": {"speed": 1, "pitch": 0, "energy": 1},
        },
    }, ensure_ascii=False)

    def opener(request, timeout):
        body = json.dumps({
            "error": {"code": "tool_use_failed", "failed_generation": failed}
        }, ensure_ascii=False).encode("utf-8")
        raise urllib.error.HTTPError(
            "https://api.groq.test", 400, "error", Message(), BytesIO(body)
        )

    client = GroqLlmClient(make_settings(tmp_path), opener=opener)
    result = client.complete("테스트", [])

    assert result.reply == "그 이야기는 여기까지만 할게요. 다른 얘기해요."
    assert result.emotion == "angry"
    assert client.last_metrics.recovered_structured_output is True


def test_transient_rate_limit_is_retried(tmp_path: Path) -> None:
    calls = 0
    sleeps = []

    def opener(request, timeout):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise http_error(429, "0.25")
        return FakeResponse(successful_payload())

    client = GroqLlmClient(make_settings(tmp_path), opener=opener, sleeper=sleeps.append)
    assert client.complete("테스트", []).reply == "도와드릴게요."
    assert calls == 2
    assert sleeps == [0.25]


def test_authentication_error_is_not_retried(tmp_path: Path) -> None:
    calls = 0

    def opener(request, timeout):
        nonlocal calls
        calls += 1
        raise http_error(401)

    client = GroqLlmClient(make_settings(tmp_path), opener=opener, sleeper=lambda _: None)
    with pytest.raises(LlmError, match="401"):
        client.complete("테스트", [])
    assert calls == 1


def test_output_parse_failure_is_retried(tmp_path: Path) -> None:
    calls = 0

    def opener(request, timeout):
        nonlocal calls
        calls += 1
        if calls == 1:
            body = json.dumps({
                "error": {
                    "code": "output_parse_failed",
                    "failed_generation": "Need to answer with the name.",
                }
            }).encode("utf-8")
            raise urllib.error.HTTPError(
                "https://api.groq.test", 400, "error", Message(), BytesIO(body)
            )
        return FakeResponse(successful_payload())

    client = GroqLlmClient(make_settings(tmp_path), opener=opener, sleeper=lambda _: None)
    assert client.complete("이름이 뭐야?", []).reply == "도와드릴게요."
    assert calls == 2
