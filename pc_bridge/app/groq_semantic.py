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
            "max_completion_tokens": self._settings.groq_max_completion_tokens,
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
                if attempt < self._settings.groq_max_retries and code in {
                    "rate_limit", "provider_unavailable",
                }:
                    self._sleep(min(0.5 * (2 ** attempt), 2.0))
                    continue
                raise SemanticLlmError(code, f"Groq HTTP error {exc.code}") from exc
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
