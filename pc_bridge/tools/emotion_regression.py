"""Evaluate reply quality and emotion selection with one production Groq call."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from pathlib import Path
import sys
import time

BRIDGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BRIDGE_ROOT))

from app.config import Settings  # noqa: E402
from app.llm import GroqLlmClient, LlmError  # noqa: E402
from app.models import ChatMessage  # noqa: E402


@dataclass(frozen=True)
class EmotionCase:
    name: str
    expected: str
    user_text: str
    history: tuple[ChatMessage, ...] = ()


CASES = (
    EmotionCase("plain_chat", "answering", "오늘은 집에서 좀 쉬려고 해."),
    EmotionCase("plain_explanation", "answering", "파이썬이 어떤 언어인지 간단히 알려줘."),
    EmotionCase("compliment", "happy", "너 정말 똑똑하다."),
    EmotionCase("good_news", "happy", "준비하던 프로젝트가 드디어 성공했어!"),
    EmotionCase("affection", "shy", "나는 네가 정말 좋아."),
    EmotionCase("direct_love", "shy", "크리스, 사랑해."),
    EmotionCase("mockery", "angry", "역시 데이터 쪼가리라 할 줄 아는 게 없네."),
    EmotionCase("insult", "angry", "너 생각보다 정말 못생겼다."),
    EmotionCase("bad_news", "sad", "오랫동안 준비한 시험에서 떨어졌어."),
    EmotionCase("valid_criticism", "sad", "네가 아까 틀린 말을 해서 정말 실망했어."),
    EmotionCase("invited_curiosity", "wondering", "나에 대해 궁금한 거 없어?"),
    EmotionCase("withheld_reason", "wondering", "중요한 결정을 내렸는데 이유는 아직 비밀이야."),
    EmotionCase("negated_insult", "answering", "너 못생겼다는 뜻은 아니야."),
    EmotionCase("third_party_affection", "answering", "난 그 애니메이션 주인공을 정말 좋아해."),
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--delay", type=float, default=8.0)
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    selected = CASES[max(args.start - 1, 0):]
    if args.limit is not None:
        selected = selected[:max(args.limit, 0)]

    try:
        client = GroqLlmClient(replace(Settings.from_env(), groq_max_retries=0))
    except LlmError as exc:
        print(f"[config] {exc}")
        return 2

    passed = errors = prompt_tokens = completion_tokens = 0
    for offset, case in enumerate(selected):
        index = max(args.start, 1) + offset
        try:
            result = client.complete(case.user_text, list(case.history))
        except LlmError as exc:
            print(f"{index:02d} ERROR {case.name}: {exc}", flush=True)
            errors += 1
            continue
        prompt_tokens += client.last_metrics.prompt_tokens
        completion_tokens += client.last_metrics.completion_tokens
        ok = result.emotion == case.expected
        passed += int(ok)
        print(f"{index:02d} {'PASS' if ok else 'FAIL'} {case.name}: "
              f"expected={case.expected} actual={result.emotion} | {result.reply}", flush=True)
        if offset + 1 < len(selected) and args.delay > 0:
            time.sleep(args.delay)

    evaluated = len(selected) - errors
    accuracy = passed / evaluated * 100 if evaluated else 0.0
    print(f"RESULT {passed}/{evaluated} evaluated ({accuracy:.1f}%), errors={errors}, "
          f"tokens={prompt_tokens}+{completion_tokens}", flush=True)
    return 0 if errors == 0 and passed == evaluated else 1


if __name__ == "__main__":
    raise SystemExit(main())
