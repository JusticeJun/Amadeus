from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import json
import socket
import time
import urllib.error
import urllib.request
from typing import Callable

from .config import Settings
from .models import ChatMessage, LlmResult
from .prompts import SYSTEM_PROMPT


class LlmError(RuntimeError):
    pass


@dataclass(frozen=True)
class LlmMetrics:
    model: str = ""
    elapsed_seconds: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    request_id: str = ""
    recovered_structured_output: bool = False


class LlmClient(ABC):
    last_metrics = LlmMetrics()

    @abstractmethod
    def complete(self, user_text: str, history: list[ChatMessage]) -> LlmResult: ...


class MockLlmClient(LlmClient):
    def complete(self, user_text: str, history: list[ChatMessage]) -> LlmResult:
        del history
        lowered = user_text.lower()
        if any(word in lowered for word in ("사랑", "좋아해")):
            emotion, reply = "shy", "갑자기 그렇게 말하면 좀 부끄럽잖아. 그래도 고마워."
        elif any(word in lowered for word in ("싫어", "바보", "미워")):
            emotion, reply = "angry", "그런 말은 기분 나쁜데. 장난이라도 적당히 해."
        elif any(word in lowered for word in ("슬퍼", "실망", "속상")):
            emotion, reply = "sad", "그랬구나. 그건 정말 속상했겠다."
        elif any(word in lowered for word in ("고마워", "예쁘", "잘했", "반가")):
            emotion, reply = "happy", "그렇게 말해 주니 기분 좋네. 고마워."
        else:
            emotion, reply = "answering", f"응, '{user_text}' 얘기를 조금 더 해 보자."
        return LlmResult.parse(json.dumps({
            "reply": reply,
            "emotion": emotion,
            "speech_style": {"speed": 1.0, "pitch": 0, "energy": 1.0},
        }, ensure_ascii=False))


_RESPONSE_SCHEMA = {
    "name": "amadeus_reply",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "reply": {"type": "string", "minLength": 1, "maxLength": 650},
            "emotion": {
                "type": "string",
                "enum": ["answering", "happy", "angry", "shy", "sad", "wondering"],
            },
            "speech_style": {
                "type": "object",
                "properties": {
                    "speed": {"type": "number", "minimum": 0.75, "maximum": 1.25},
                    "pitch": {"type": "number", "minimum": -6, "maximum": 6},
                    "energy": {"type": "number", "minimum": 0.7, "maximum": 1.3},
                },
                "required": ["speed", "pitch", "energy"],
                "additionalProperties": False,
            },
        },
        "required": ["reply", "emotion", "speech_style"],
        "additionalProperties": False,
    },
}


class GroqLlmClient(LlmClient):
    def __init__(
        self,
        settings: Settings,
        *,
        opener: Callable[..., object] = urllib.request.urlopen,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        if not settings.groq_api_key:
            raise LlmError("GROQ_API_KEY가 없습니다. .env를 설정하거나 mock 모드를 사용하세요.")
        self._settings = settings
        self._opener = opener
        self._sleep = sleeper
        self.last_metrics = LlmMetrics(model=settings.groq_model)

    def complete(self, user_text: str, history: list[ChatMessage]) -> LlmResult:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.extend({"role": item.role, "content": item.content} for item in history)
        messages.append({"role": "user", "content": user_text})
        body = json.dumps({
            "model": self._settings.groq_model,
            "messages": messages,
            "temperature": self._settings.groq_temperature,
            "max_completion_tokens": self._settings.groq_max_completion_tokens,
            "reasoning_effort": self._settings.groq_reasoning_effort,
            "include_reasoning": False,
            "response_format": {"type": "json_schema", "json_schema": _RESPONSE_SCHEMA},
        }, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
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
                with self._opener(request, timeout=self._settings.groq_timeout_seconds) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                    request_id = response.headers.get("x-request-id", "")
                content = payload["choices"][0]["message"]["content"]
                usage = payload.get("usage") or {}
                self.last_metrics = LlmMetrics(
                    model=str(payload.get("model") or self._settings.groq_model),
                    elapsed_seconds=time.perf_counter() - started,
                    prompt_tokens=int(usage.get("prompt_tokens") or 0),
                    completion_tokens=int(usage.get("completion_tokens") or 0),
                    request_id=request_id,
                )
                return LlmResult.parse(content)
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")[:4096]
                recovered = self._recover_failed_generation(exc.code, detail)
                if recovered is not None:
                    self.last_metrics = LlmMetrics(
                        model=self._settings.groq_model,
                        elapsed_seconds=time.perf_counter() - started,
                        recovered_structured_output=True,
                    )
                    return LlmResult.parse(recovered)
                if attempt < self._settings.groq_max_retries and self._error_code(detail) == "output_parse_failed":
                    self._sleep(0.1)
                    continue
                if attempt < self._settings.groq_max_retries and (
                    exc.code == 429 or exc.code == 498 or exc.code >= 500
                ):
                    self._sleep(self._retry_delay(exc, attempt))
                    continue
                raise LlmError(f"Groq HTTP 오류 {exc.code}: {detail[:500]}") from exc
            except (urllib.error.URLError, TimeoutError, socket.timeout) as exc:
                if attempt < self._settings.groq_max_retries:
                    self._sleep(min(0.5 * (2 ** attempt), 2.0))
                    continue
                raise LlmError(f"Groq 네트워크/시간초과 오류: {exc}") from exc
            except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise LlmError("Groq 응답 형식을 해석하지 못했습니다.") from exc
        raise LlmError("Groq 요청을 완료하지 못했습니다.")

    @staticmethod
    def _retry_delay(exc: urllib.error.HTTPError, attempt: int) -> float:
        try:
            retry_after = float(exc.headers.get("retry-after", ""))
            return min(max(retry_after, 0.0), 15.0)
        except (TypeError, ValueError):
            return min(0.5 * (2 ** attempt), 2.0)

    @staticmethod
    def _recover_failed_generation(status_code: int, detail: str) -> str | None:
        if status_code != 400:
            return None
        try:
            error = json.loads(detail).get("error") or {}
        except (AttributeError, json.JSONDecodeError):
            return None
        generated = error.get("failed_generation")
        if error.get("code") not in {"json_validate_failed", "tool_use_failed"} or not isinstance(generated, str):
            return None
        generated = generated.strip()
        if not generated:
            return None
        if error.get("code") == "tool_use_failed":
            try:
                tool_call = json.loads(generated)
                arguments = tool_call.get("arguments")
            except (AttributeError, json.JSONDecodeError):
                return None
            if isinstance(arguments, dict):
                return json.dumps(arguments, ensure_ascii=False)
            if isinstance(arguments, str):
                return arguments
            return None
        return generated

    @staticmethod
    def _error_code(detail: str) -> str:
        try:
            error = json.loads(detail).get("error") or {}
            return str(error.get("code") or "")
        except (AttributeError, json.JSONDecodeError):
            return ""
