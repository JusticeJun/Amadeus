"""Run a small Korean response/emotion check without TTS or ESP32 hardware."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


BRIDGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BRIDGE_ROOT))

from app.config import Settings  # noqa: E402
from app.llm import GroqLlmClient, LlmError  # noqa: E402
from app.models import ChatMessage, is_repetitive_reply  # noqa: E402


CASES = (
    "안녕하세요. 오늘 기분은 어때요?",
    "오늘 해야 할 일이 너무 많아서 조금 지쳤어.",
    "크리스 목소리가 정말 예쁘네.",
    "갑자기 큰 소리가 나서 깜짝 놀랐어.",
    "오늘 저녁 메뉴를 차분하게 추천해 줘.",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name-regression", action="store_true")
    args = parser.parse_args()
    settings = Settings.from_env()
    try:
        client = GroqLlmClient(settings)
    except LlmError as exc:
        print(f"[config] {exc}")
        return 2

    if args.name_regression:
        return run_name_regression(client)
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


def run_name_regression(client: GroqLlmClient) -> int:
    history: list[ChatMessage] = []
    inputs = (
        "이제부터 마키세 크리스가 너의 이름이야.",
        "너의 이름이 뭐라고?",
        "이름이 뭐라고 너?",
    )
    for index, text in enumerate(inputs, 1):
        try:
            result = client.complete(text, history)
        except LlmError as exc:
            print(json.dumps({"case": index, "error": str(exc)}, ensure_ascii=True))
            return 1
        repetitive = is_repetitive_reply(result.reply, text, history)
        print(json.dumps({
            "case": index,
            "input": text,
            "reply": result.reply,
            "emotion": result.emotion,
            "incorrectly_flagged_repetitive": repetitive,
            "elapsed_seconds": round(client.last_metrics.elapsed_seconds, 3),
        }, ensure_ascii=True))
        if repetitive:
            return 1
        history.extend([ChatMessage("user", text), ChatMessage("assistant", result.reply)])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
