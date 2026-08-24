from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from ..models import ChatMessage


@dataclass(frozen=True)
class RoutingRequest:
    text: str
    history: tuple[ChatMessage, ...] = ()


@dataclass(frozen=True)
class CapabilityMatch:
    capability: str
    confidence: float | None = None

    def __post_init__(self) -> None:
        if not self.capability.strip():
            raise ValueError("capability must not be empty")
        if self.confidence is not None and not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")


@dataclass(frozen=True)
class RouteDecision:
    matches: tuple[CapabilityMatch, ...] = ()

    def __post_init__(self) -> None:
        capabilities = [match.capability for match in self.matches]
        if len(capabilities) != len(set(capabilities)):
            raise ValueError("route decision contains duplicate capabilities")

    @property
    def required_capabilities(self) -> frozenset[str]:
        return frozenset(match.capability for match in self.matches)


class SemanticRouter(Protocol):
    def route(self, request: RoutingRequest) -> RouteDecision: ...
