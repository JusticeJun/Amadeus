from __future__ import annotations

from abc import ABC, abstractmethod
import json
import socket
import urllib.error
import urllib.request

from .config import Settings
from .models import ChatMessage, LlmResult
from .prompts import SYSTEM_PROMPT


class LlmError(RuntimeError):
    pass


class LlmClient(ABC):
    @abstractmethod
    def complete(self, user_text: str, history: list[ChatMessage]) -> LlmResult: ...


class MockLlmClient(LlmClient):
    def complete(self, user_text: str, history: list[ChatMessage]) -> LlmResult:
        del history
        lowered = user_text.lower()
        if any(word in lowered for word in ("고마워", "잘했", "예쁘", "귀엽")):
            emotion, reply = "shy", "그렇게 말씀하시면 조금 민망하네요. 그래도 고마워요."
        elif any(word in lowered for word in ("놀라", "깜짝")):
            emotion, reply = "surprised", "조금 놀랐어요. 갑자기 그러시면 곤란해요."
        elif any(word in lowered for word in ("싫어", "바보", "미워")):
            emotion, reply = "pout", "그런 말은 별로네요. 그래도 필요한 일은 도와드릴게요."
        elif any(word in lowered for word in ("안녕", "반가")):
            emotion, reply = "happy", "안녕하세요. 오늘도 천천히 시작해 봐요."
        else:
            emotion, reply = "neutral", f"네, '{user_text}'에 대해 조금 더 생각해 볼게요."
        return LlmResult.parse(json.dumps({
            "reply": reply,
            "emotion": emotion,
            "speech_style": {"speed": 1.0, "pitch": 0, "energy": 1.0},
        }, ensure_ascii=False))


class GroqLlmClient(LlmClient):
    def __init__(self, settings: Settings) -> None:
        if not settings.groq_api_key:
            raise LlmError("GROQ_API_KEY가 없습니다. .env를 설정하거나 mock 모드를 사용하세요.")
        self._settings = settings

    def complete(self, user_text: str, history: list[ChatMessage]) -> LlmResult:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.extend({"role": item.role, "content": item.content} for item in history)
        messages.append({"role": "user", "content": user_text})
        body = json.dumps({
            "model": self._settings.groq_model,
            "messages": messages,
            "temperature": 0.7,
            "response_format": {"type": "json_object"},
        }).encode("utf-8")
        request = urllib.request.Request(
            self._settings.groq_api_url,
            data=body,
            headers={
                "Authorization": f"Bearer {self._settings.groq_api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self._settings.groq_timeout_seconds) as response:
                payload = json.loads(response.read().decode("utf-8"))
            content = payload["choices"][0]["message"]["content"]
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            raise LlmError(f"Groq HTTP 오류 {exc.code}: {detail}") from exc
        except (urllib.error.URLError, TimeoutError, socket.timeout) as exc:
            raise LlmError(f"Groq 네트워크/시간초과 오류: {exc}") from exc
        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
            raise LlmError("Groq 응답 형식을 해석하지 못했습니다.") from exc
        return LlmResult.parse(content)

