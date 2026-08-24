from __future__ import annotations

import json

from ..pc_control import PcActionParser, PcController
from .base import ToolResult


class PcControlTool:
    name = "pc_control"
    side_effecting = True

    def __init__(self, parser: PcActionParser, controller: PcController) -> None:
        self._parser = parser
        self._controller = controller

    def run(self, user_text: str) -> ToolResult:
        parsed = self._parser.parse(user_text)
        if not parsed.ok:
            return ToolResult(
                self.name,
                False,
                {"reason": parsed.error_code},
                f"PC action parsing failed: {parsed.error_code}",
            )
        results = tuple(self._controller.execute(action) for action in parsed.actions)
        data = {
            "actions": [
                {
                    "type": result.action.action_type.value,
                    **({"target": result.action.target} if result.action.target else {}),
                    "ok": result.ok,
                    **result.data,
                }
                for result in results
            ]
        }
        ok = all(result.ok for result in results)
        errors = "; ".join(result.error for result in results if result.error)
        return ToolResult(self.name, ok, data, errors)

    def build_llm_context(self, result: ToolResult) -> str:
        if not result.ok:
            reason = str(result.data.get("reason") or "execution_failed")
            safe_reason = {
                "unsafe_input": "요청에 안전하지 않은 실행 형식이 포함되어 있어 수행하지 않았다.",
                "invalid_volume": "요청한 볼륨 값이 0에서 100 사이가 아니어서 수행하지 않았다.",
                "unsupported_app": "요청한 앱이 허용된 앱 목록에 없어 실행하지 않았다.",
                "unsupported_action": "요청한 PC 동작은 현재 지원하지 않아 수행하지 않았다.",
            }.get(reason, "요청한 PC 동작을 완료하지 못했다.")
            return (
                f"검증된 PC 실행 결과: {safe_reason} "
                "내부 오류, 명령, 도구나 시스템 구현을 언급하지 말고 결과만 자연스럽게 답한다."
            )
        compact = json.dumps(result.data, ensure_ascii=False, separators=(",", ":"))
        return (
            f"검증된 PC 실행 결과: {compact}\n"
            "실제로 성공한 action만 완료되었다고 말한다. 구조를 낭독하거나 실행하지 않은 동작을 "
            "추측하지 말고 크리스의 말투로 짧고 자연스럽게 답한다."
        )
