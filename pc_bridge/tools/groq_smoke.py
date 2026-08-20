"""Run a small Korean response/emotion check without TTS or ESP32 hardware."""

from __future__ import annotations

import json
from pathlib import Path
import sys


BRIDGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BRIDGE_ROOT))

from app.config import Settings  # noqa: E402
from app.llm import GroqLlmClient, LlmError  # noqa: E402


CASES = (
    "안녕하세요. 오늘 기분은 어때요?",
    "오늘 해야 할 일이 너무 많아서 조금 지쳤어.",
    "크리스 목소리가 정말 예쁘네.",
    "갑자기 큰 소리가 나서 깜짝 놀랐어.",
    "오늘 저녁 메뉴를 차분하게 추천해 줘.",
)


def main() -> int:
    settings = Settings.from_env()
    try:
        client = GroqLlmClient(settings)
    except LlmError as exc:
        print(f"[config] {exc}")
        return 2

    for index, text in enumerate(CASES, 1):
        try:
            result = client.complete(text, [])
        except LlmError as exc:
            print(f"[{index}] ERROR: {exc}")
            return 1
        metrics = client.last_metrics
        print(json.dumps({
            "case": index,
            "input": text,
            "reply": result.reply,
            "emotion": result.emotion,
            "speech_style": result.speech_style.__dict__,
            "elapsed_seconds": round(metrics.elapsed_seconds, 3),
            "prompt_tokens": metrics.prompt_tokens,
            "completion_tokens": metrics.completion_tokens,
            "model": metrics.model,
        }, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
