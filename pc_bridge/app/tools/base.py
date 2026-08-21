from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Protocol


@dataclass(frozen=True)
class ToolResult:
    name: str
    ok: bool
    data: dict[str, Any]
    error: str = ""

    def llm_context(self) -> str:
        if not self.ok:
            return (
                "요청한 외부 정보를 지금 확인하지 못했다. 내부 오류, API, 도구나 시스템을 "
                "언급하지 말고 지금은 확인할 수 없다고 크리스의 말투로 짧게 답한다."
            )
        compact = json.dumps(self.data, ensure_ascii=False, separators=(",", ":"))
        return (
            f"검증된 외부 정보: {compact}\n"
            "이 데이터만 사실 근거로 사용하고 구조를 낭독하지 않는다. location은 조회 지점의 "
            "이름일 뿐 지리나 주변 환경의 근거가 아니다. 데이터에 없는 하늘 상태, 체감, 지역 특성, "
            "활동 적합성이나 추천을 절대 만들어내지 않는다. 강수가 없다는 사실만으로 맑다고 "
            "표현하지 말고 sky 값이 있을 때만 하늘 상태를 말한다. 현재 날씨 질문에는 제공된 기온, "
            "습도, 강수, 바람, sky만 자연스러운 한두 문장으로 답한다. interaction_mode가 "
            "casual_observation이면 정보 요청이 아니라 일상적인 말에 대한 반응이므로 수치를 나열하거나 "
            "예보를 보고하지 말고, 관련된 실제 날씨를 반영해 친구처럼 짧고 자연스럽게 받아준다. "
            "이때 데이터에 없는 장소, 풍경, 외출 활동, 적합성, 추천을 연상해서 덧붙이지 않는다."
        )


class Tool(Protocol):
    name: str

    def matches(self, user_text: str) -> bool: ...

    def run(self, user_text: str) -> ToolResult: ...
