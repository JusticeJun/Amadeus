from __future__ import annotations

import time

from .input_provider import InputProvider
from .llm import LlmClient, LlmError
from .memory import ConversationMemory
from .models import ChatMessage, LlmResult, is_repetitive_reply
from .serial_bridge import SerialBridge
from .tts import TtsEngine, TtsError


class ConversationManager:
    def __init__(self, input_provider: InputProvider, llm: LlmClient,
                 tts: TtsEngine, serial_bridge: SerialBridge,
                 neutral_hold_seconds: float = 1.2) -> None:
        self._input = input_provider
        self._llm = llm
        self._tts = tts
        self._serial = serial_bridge
        self._memory = ConversationMemory()
        self._neutral_hold = neutral_hold_seconds

    def run(self) -> None:
        print("Amadeus PC Bridge - 종료하려면 /quit 또는 Ctrl+C")
        while True:
            self._serial.send_state("neutral")
            user_text = self._input.read(self._set_input_activity)
            if user_text is None or user_text.lower() in {"/quit", "/exit"}:
                break
            if not user_text:
                continue
            turn_started = time.perf_counter()
            history = self._memory.messages()
            try:
                result = self._llm.complete(user_text, history)
                result = self._replace_repetitive_reply(user_text, history, result)
            except LlmError as exc:
                print(f"[llm] {exc}")
                self._serial.send_state("neutral")
                continue
            print(f"크리스 [{result.emotion}] > {result.reply}")
            self._memory.add_turn(user_text, result.reply)
            llm_finished = time.perf_counter()
            metrics = self._llm.last_metrics
            if metrics.recovered_structured_output:
                print("[groq] JSON 형식이 깨진 응답을 안전하게 복구했습니다.")
            if metrics.prompt_tokens or metrics.completion_tokens:
                print(
                    f"[groq] {metrics.model} | 입력 {metrics.prompt_tokens} tokens | "
                    f"출력 {metrics.completion_tokens} tokens"
                )
            try:
                audio = self._tts.synthesize(result)
                tts_finished = time.perf_counter()
                self._serial.send_state(result.emotion)
                self._tts.play(audio)
                print(
                    f"[timing] LLM {llm_finished - turn_started:.2f}s | "
                    f"TTS {tts_finished - llm_finished:.2f}s | "
                    f"음성 시작 {tts_finished - turn_started:.2f}s"
                )
            except TtsError as exc:
                print(f"[tts] {exc}")
                self._serial.send_state(result.emotion)
            time.sleep(self._neutral_hold)
            self._serial.send_state("neutral")
        self._serial.send_state("sleep")

    def _set_input_activity(self, active: bool) -> None:
        self._serial.send_state("listening" if active else "neutral")

    def _replace_repetitive_reply(
        self,
        user_text: str,
        history: list[ChatMessage],
        result: LlmResult,
    ) -> LlmResult:
        if not is_repetitive_reply(result.reply, user_text, history):
            return result
        print("[llm] 사용자 문장 반향/최근 답변 반복을 감지해 다시 표현합니다.")
        retry_history = history + [ChatMessage(
            "system",
            "방금 만든 후보 답변이 사용자의 말을 그대로 따라 하거나 최근 답변과 겹쳤다. "
            "사용자의 최신 의도에는 답하되, 같은 뜻을 자연스럽게 다른 말로 표현하고 "
            "짧은 새로운 반응이나 정보를 하나 더해라. 후보 문장을 반복하지 마라: "
            + result.reply,
        )]
        try:
            replacement = self._llm.complete(user_text, retry_history)
        except LlmError as exc:
            print(f"[llm] 재표현 요청 실패, 원래의 유효한 답변을 사용합니다: {exc}")
            return result
        return replacement if not is_repetitive_reply(replacement.reply, user_text, history) else result
