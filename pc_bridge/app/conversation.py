from __future__ import annotations

import time

from .input_provider import InputProvider
from .llm import LlmClient, LlmError
from .memory import ConversationMemory
from .models import ChatMessage, LlmResult, is_repetitive_reply
from .routing import RoutingRequest, RuleBasedSemanticRouter, SemanticRouter
from .serial_bridge import SerialBridge
from .tts import TtsEngine, TtsError
from .tools import ToolExecutor


_PLANNING_REQUIRED_CONTEXT = (
    "사용자의 조건부 요청은 이해했지만 현재는 안전하게 처리할 수 없어 조회와 조작을 포함한 "
    "어떤 기능도 실행하지 않았다. 실행했다고 절대 말하지 말고, 조건을 건 조작은 아직 수행할 수 "
    "없다는 사실만 크리스의 말투로 짧고 정확하게 답한다. 내부 구조를 언급하거나 요청과 무관한 "
    "조언과 대안을 덧붙이지 않는다."
)
_SIDE_EFFECT_FAILURE_CONTEXT = (
    "요청한 실제 조작은 실패했고 변경된 것은 없다. 성공하거나 완료했다고 절대 말하지 말고, "
    "이번 요청은 실행되지 않았다는 사실을 크리스의 말투로 짧고 정확하게 답한다. 내부 오류명이나 "
    "구현 용어는 말하지 않는다."
)
_NON_EXECUTION_FALLBACK = "그 요청은 실행되지 않았어. 아직은 제대로 처리할 수 없어."
_PLANNING_FALLBACK = "그렇게 조건을 걸어서 조작하는 건 아직 못 해."


class ConversationManager:
    def __init__(self, input_provider: InputProvider, llm: LlmClient,
                 tts: TtsEngine, serial_bridge: SerialBridge,
                 neutral_hold_seconds: float = 1.2,
                 semantic_router: SemanticRouter | None = None,
                 tool_executor: ToolExecutor | None = None) -> None:
        self._input = input_provider
        self._llm = llm
        self._tts = tts
        self._serial = serial_bridge
        self._memory = ConversationMemory()
        self._neutral_hold = neutral_hold_seconds
        self._router = semantic_router or RuleBasedSemanticRouter()
        self._executor = tool_executor or ToolExecutor()

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
            route = self._router.route(RoutingRequest(user_text, tuple(history)))
            if route.planning_required:
                print(f"[route] planning required: {route.planning_reason}")
                history = history + [ChatMessage("system", _PLANNING_REQUIRED_CONTEXT)]
            executions = self._executor.execute(route, user_text)
            failed_side_effect = False
            for execution in executions:
                tool_result = execution.result
                status = "ok" if tool_result.ok else f"failed: {tool_result.error}"
                print(f"[tool:{tool_result.name}] {status}")
                history = history + [ChatMessage("system", execution.llm_context)]
                failed_side_effect = failed_side_effect or (
                    execution.side_effecting and not tool_result.ok
                )
            if failed_side_effect:
                history = history + [ChatMessage("system", _SIDE_EFFECT_FAILURE_CONTEXT)]
            try:
                result = self._llm.complete(user_text, history)
                result = self._replace_repetitive_reply(user_text, history, result)
                result = self._enforce_execution_truth(
                    user_text,
                    history,
                    result,
                    planning_required=route.planning_required,
                    failed_side_effect=failed_side_effect,
                )
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

    def _enforce_execution_truth(
        self,
        user_text: str,
        history: list[ChatMessage],
        result: LlmResult,
        *,
        planning_required: bool,
        failed_side_effect: bool,
    ) -> LlmResult:
        if not (planning_required or failed_side_effect):
            return result
        if _is_truthful_non_execution_reply(result.reply, planning_required):
            return result
        print("[llm] 실행되지 않은 작업을 성공으로 표현하지 않도록 답변을 다시 생성합니다.")
        retry_context = (
            _PLANNING_REQUIRED_CONTEXT if planning_required else _SIDE_EFFECT_FAILURE_CONTEXT
        )
        retry_history = history + [ChatMessage(
            "system",
            retry_context + " 방금 답변은 이 제약을 충족하지 못했으므로 정확하게 다시 답한다.",
        )]
        try:
            replacement = self._llm.complete(user_text, retry_history)
        except LlmError:
            replacement = result
        if _is_truthful_non_execution_reply(replacement.reply, planning_required):
            return replacement
        fallback = _PLANNING_FALLBACK if planning_required else _NON_EXECUTION_FALLBACK
        return LlmResult(reply=fallback, emotion="answering")


def _is_truthful_non_execution_reply(reply: str, planning_required: bool) -> bool:
    compact = "".join(reply.lower().split())
    acknowledges_failure = any(phrase in compact for phrase in (
        "못해", "못했", "수없", "실행되지않", "실행하지않", "안됐", "실패",
        "처리되지않", "변경되지않",
    ))
    success_claim = any(phrase in compact for phrase in (
        "줄였", "올렸", "맞췄", "설정했", "켰어", "켰다", "열었", "실행했",
        "완료했", "바꿨", "음소거했", "재생했", "멈췄",
    ))
    unrelated_suggestion = planning_required and any(phrase in compact for phrase in (
        "어때", "대신", "추천", "마셔", "챙겨",
    ))
    return acknowledges_failure and not success_claim and not unrelated_suggestion
