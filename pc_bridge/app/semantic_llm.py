from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class SemanticLlmRequest:
    task: str
    system_prompt: str
    input: dict[str, object]
    schema_name: str
    schema: dict[str, object]


@dataclass(frozen=True)
class SemanticLlmMetrics:
    provider: str = ""
    model: str = ""
    elapsed_seconds: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    request_id: str = ""


@dataclass(frozen=True)
class SemanticLlmResponse:
    data: dict[str, object]
    metrics: SemanticLlmMetrics = SemanticLlmMetrics()


class SemanticLlmError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class SemanticLlmClient(Protocol):
    def complete(self, request: SemanticLlmRequest) -> SemanticLlmResponse: ...
