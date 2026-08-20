from __future__ import annotations

import time

from .input_provider import InputProvider
from .llm import LlmClient, LlmError
from .memory import ConversationMemory
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
            self._serial.send_state("listening")
            user_text = self._input.read()
            if user_text is None or user_text.lower() in {"/quit", "/exit"}:
                break
            if not user_text:
                continue
            self._serial.send_state("thinking")
            try:
                result = self._llm.complete(user_text, self._memory.messages())
            except LlmError as exc:
                print(f"[llm] {exc}")
                self._serial.send_state("neutral")
                continue
            print(f"크리스 [{result.emotion}] > {result.reply}")
            self._memory.add_turn(user_text, result.reply)
            try:
                audio = self._tts.synthesize(result)
                self._serial.send_state(result.emotion)
                self._tts.play(audio)
            except TtsError as exc:
                print(f"[tts] {exc}")
                self._serial.send_state(result.emotion)
            time.sleep(self._neutral_hold)
            self._serial.send_state("neutral")
        self._serial.send_state("sleep")
