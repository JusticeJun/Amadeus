from __future__ import annotations

from email.message import Message
from io import BytesIO
import json
from types import SimpleNamespace
import urllib.error

import pytest

from app.groq_semantic import GroqSemanticLlmClient
from app.semantic_llm import SemanticLlmError, SemanticLlmRequest


class FakeResponse:
    headers = {"x-request-id": "semantic-request"}

    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return json.dumps(self.payload).encode()


def settings(retries=0):
    return SimpleNamespace(
        groq_api_key="secret",
        groq_api_url="https://api.groq.test/chat/completions",
        groq_model="semantic-model",
        groq_max_completion_tokens=256,
        groq_reasoning_effort="low",
        music_semantic_model="semantic-model",
        music_semantic_reasoning_effort="medium",
        groq_max_retries=retries,
        groq_timeout_seconds=3,
    )


def request():
    return SemanticLlmRequest(
        task="music_interpretation",
        system_prompt="Return structured music actions.",
        input={"utterance": "수평선 틀어줘", "music_context": []},
        schema_name="music_action_sequence",
        schema={"type": "object"},
    )


def test_groq_semantic_adapter_sends_task_schema_and_reports_usage() -> None:
    captured = {}

    def opener(http_request, timeout):
        captured["body"] = json.loads(http_request.data)
        captured["timeout"] = timeout
        return FakeResponse({
            "model": "resolved-model",
            "choices": [{"message": {"content": '{"status":"ambiguous","actions":[]}'}}],
            "usage": {"prompt_tokens": 30, "completion_tokens": 8},
        })

    result = GroqSemanticLlmClient(settings(), opener=opener).complete(request())

    assert result.data["status"] == "ambiguous"
    assert captured["body"]["response_format"]["json_schema"]["strict"] is True
    assert captured["body"]["reasoning_effort"] == "medium"
    assert json.loads(captured["body"]["messages"][-1]["content"])["utterance"] == (
        "수평선 틀어줘"
    )
    assert captured["timeout"] == 3
    assert result.metrics.provider == "groq"
    assert result.metrics.model == "resolved-model"
    assert (result.metrics.input_tokens, result.metrics.output_tokens) == (30, 8)


def test_groq_semantic_adapter_falls_back_to_character_provider_model() -> None:
    configured = settings()
    configured.music_semantic_model = ""
    captured = {}

    def opener(http_request, timeout):
        captured["body"] = json.loads(http_request.data)
        return FakeResponse({
            "choices": [{"message": {"content": '{"status":"not_music","actions":[]}'}}],
        })

    GroqSemanticLlmClient(configured, opener=opener).complete(request())

    assert captured["body"]["model"] == configured.groq_model


def test_groq_semantic_adapter_classifies_rate_limit() -> None:
    def opener(http_request, timeout):
        raise urllib.error.HTTPError(
            http_request.full_url, 429, "limited", Message(), BytesIO(b"{}"),
        )

    with pytest.raises(SemanticLlmError) as error:
        GroqSemanticLlmClient(settings(), opener=opener).complete(request())

    assert error.value.code == "rate_limit"


def test_groq_semantic_adapter_reports_safe_http_error_details() -> None:
    headers = Message()
    headers["x-request-id"] = "request-400"

    def opener(http_request, timeout):
        raise urllib.error.HTTPError(
            http_request.full_url,
            400,
            "bad request",
            headers,
            BytesIO(json.dumps({"error": {
                "type": "invalid_request_error",
                "code": "json_schema_invalid",
                "message": "secret prompt and bearer credential",
            }}).encode()),
        )

    with pytest.raises(SemanticLlmError) as error:
        GroqSemanticLlmClient(settings(), opener=opener).complete(request())

    detail = str(error.value)
    assert error.value.code == "provider_error"
    assert "HTTP 400" in detail
    assert "request_id=request-400" in detail
    assert "type=invalid_request_error" in detail
    assert "code=json_schema_invalid" in detail
    assert "secret prompt" not in detail
    assert "credential" not in detail


def test_groq_semantic_adapter_retries_schema_generation_failure() -> None:
    calls = 0

    def opener(http_request, timeout):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise urllib.error.HTTPError(
                http_request.full_url,
                400,
                "bad request",
                Message(),
                BytesIO(json.dumps({"error": {
                    "type": "invalid_request_error",
                    "code": "json_validate_failed",
                }}).encode()),
            )
        return FakeResponse({
            "choices": [{"message": {
                "content": '{"status":"ambiguous","actions":[]}',
            }}],
        })

    delays = []
    result = GroqSemanticLlmClient(
        settings(retries=1), opener=opener, sleeper=delays.append,
    ).complete(request())

    assert result.data == {"status": "ambiguous", "actions": []}
    assert calls == 2
    assert delays == [0.5]


def test_groq_semantic_adapter_rejects_malformed_json() -> None:
    def opener(http_request, timeout):
        return FakeResponse({"choices": [{"message": {"content": "not json"}}]})

    with pytest.raises(SemanticLlmError) as error:
        GroqSemanticLlmClient(settings(), opener=opener).complete(request())

    assert error.value.code == "malformed_response"


def test_groq_semantic_adapter_requires_provider_credentials() -> None:
    configured = settings()
    configured.groq_api_key = ""

    with pytest.raises(SemanticLlmError) as error:
        GroqSemanticLlmClient(configured)

    assert error.value.code == "provider_unavailable"
