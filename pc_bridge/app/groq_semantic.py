from __future__ import annotations

import json
import socket
import time
import urllib.error
import urllib.request
from typing import Callable

from .config import Settings
from .semantic_llm import (
    SemanticLlmClient, SemanticLlmError, SemanticLlmMetrics, SemanticLlmRequest,
    SemanticLlmResponse,
)


_MIN_SEMANTIC_COMPLETION_TOKENS = 1024


class GroqSemanticLlmClient(SemanticLlmClient):
    """Provider adapter for strict structured semantic tasks."""

    def __init__(
        self,
        settings: Settings,
        *,
        opener: Callable[..., object] = urllib.request.urlopen,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        if not settings.groq_api_key:
            raise SemanticLlmError("provider_unavailable", "GROQ_API_KEY is not configured")
        self._settings = settings
        self._opener = opener
        self._sleep = sleeper

    def complete(self, request: SemanticLlmRequest) -> SemanticLlmResponse:
        model = self._settings.music_semantic_model or self._settings.groq_model
        body = json.dumps({
            "model": model,
            "messages": [
                {"role": "system", "content": request.system_prompt},
                {"role": "user", "content": json.dumps(request.input, ensure_ascii=False)},
            ],
            "temperature": 0,
            "max_completion_tokens": max(
                _MIN_SEMANTIC_COMPLETION_TOKENS,
                self._settings.groq_max_completion_tokens,
            ),
            "reasoning_effort": self._settings.music_semantic_reasoning_effort,
            "include_reasoning": False,
            "response_format": {"type": "json_schema", "json_schema": {
                "name": request.schema_name,
                "strict": True,
                "schema": request.schema,
            }},
        }, ensure_ascii=False).encode("utf-8")
        http_request = urllib.request.Request(
            self._settings.groq_api_url,
            data=body,
            headers={
                "Authorization": f"Bearer {self._settings.groq_api_key}",
                "Content-Type": "application/json",
                "User-Agent": "Amadeus-PC-Bridge/0.1",
            },
            method="POST",
        )
        started = time.perf_counter()
        for attempt in range(self._settings.groq_max_retries + 1):
            try:
                with self._opener(
                    http_request, timeout=self._settings.groq_timeout_seconds,
                ) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                    request_id = response.headers.get("x-request-id", "")
                data = json.loads(payload["choices"][0]["message"]["content"])
                if not isinstance(data, dict):
                    raise ValueError("structured response is not an object")
                usage = payload.get("usage") or {}
                return SemanticLlmResponse(data, SemanticLlmMetrics(
                    provider="groq",
                    model=str(payload.get("model") or model),
                    elapsed_seconds=time.perf_counter() - started,
                    input_tokens=int(usage.get("prompt_tokens") or 0),
                    output_tokens=int(usage.get("completion_tokens") or 0),
                    request_id=request_id,
                ))
            except urllib.error.HTTPError as exc:
                code = "rate_limit" if exc.code == 429 else (
                    "provider_unavailable" if exc.code >= 500 else "provider_error"
                )
                detail, provider_code = _safe_http_error_details(exc, request.schema)
                retryable = code in {"rate_limit", "provider_unavailable"} or (
                    provider_code == "json_validate_failed"
                )
                if attempt < self._settings.groq_max_retries and retryable:
                    self._sleep(min(0.5 * (2 ** attempt), 2.0))
                    continue
                raise SemanticLlmError(code, detail) from exc
            except (urllib.error.URLError, TimeoutError, socket.timeout) as exc:
                if attempt < self._settings.groq_max_retries:
                    self._sleep(min(0.5 * (2 ** attempt), 2.0))
                    continue
                code = "timeout" if isinstance(exc, (TimeoutError, socket.timeout)) else (
                    "provider_unavailable"
                )
                raise SemanticLlmError(code, "semantic provider unavailable") from exc
            except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise SemanticLlmError("malformed_response", "invalid structured response") from exc
        raise SemanticLlmError("provider_unavailable", "semantic request failed")


def _safe_http_error_details(
    exc: urllib.error.HTTPError,
    schema: dict[str, object],
) -> tuple[str, str]:
    details = [f"Groq HTTP {exc.code}"]
    request_id = _safe_identifier(exc.headers.get("x-request-id", ""))
    if request_id:
        details.append(f"request_id={request_id}")
    try:
        payload = json.loads(exc.read().decode("utf-8"))
        error = payload.get("error") if isinstance(payload, dict) else None
    except (AttributeError, UnicodeDecodeError, json.JSONDecodeError):
        error = None
    provider_code = ""
    if isinstance(error, dict):
        for field in ("type", "code"):
            value = _safe_identifier(error.get(field))
            if value:
                details.append(f"{field}={value}")
                if field == "code":
                    provider_code = value
        failed_generation = error.get("failed_generation")
        if failed_generation is None and isinstance(payload, dict):
            failed_generation = payload.get("failed_generation")
        mismatch = _safe_schema_mismatch(failed_generation, schema)
        if mismatch:
            details.append(f"schema_mismatch={mismatch}")
    return " ".join(details), provider_code


def _safe_identifier(value: object) -> str:
    text = str(value or "")
    return text if text and len(text) <= 100 and all(
        character.isalnum() or character in "._-" for character in text
    ) else ""


def _safe_schema_mismatch(generation: object, schema: dict[str, object]) -> str:
    try:
        value = json.loads(generation) if isinstance(generation, str) else generation
    except json.JSONDecodeError:
        return "$:invalid_json"
    return _first_schema_mismatch(value, schema, "$") if generation is not None else ""


def _first_schema_mismatch(value: object, schema: object, path: str) -> str:
    if not isinstance(schema, dict):
        return ""
    expected = schema.get("type")
    type_matches = {
        "object": isinstance(value, dict),
        "array": isinstance(value, list),
        "string": isinstance(value, str),
        "boolean": isinstance(value, bool),
        "integer": isinstance(value, int) and not isinstance(value, bool),
        "number": isinstance(value, (int, float)) and not isinstance(value, bool),
    }
    if expected in type_matches and not type_matches[expected]:
        return f"{path}:type"
    enum = schema.get("enum")
    if isinstance(enum, list) and value not in enum:
        return f"{path}:enum"
    if isinstance(value, dict) and expected == "object":
        properties = schema.get("properties")
        properties = properties if isinstance(properties, dict) else {}
        required = schema.get("required")
        if isinstance(required, list):
            for field in required:
                if isinstance(field, str) and field not in value:
                    return f"{path}.{field}:required"
        if schema.get("additionalProperties") is False:
            extra = next((field for field in value if field not in properties), "")
            if extra:
                return f"{path}.{extra}:additional_property"
        for field, child in value.items():
            if field in properties:
                mismatch = _first_schema_mismatch(child, properties[field], f"{path}.{field}")
                if mismatch:
                    return mismatch
    if isinstance(value, list) and expected == "array":
        for index, item in enumerate(value):
            mismatch = _first_schema_mismatch(item, schema.get("items"), f"{path}[{index}]")
            if mismatch:
                return mismatch
    return ""
