"""Structured and testable Windows PC control domain."""

from .actions import (
    PcAction,
    PcActionParseResult,
    PcActionParser,
    PcActionResult,
    PcActionType,
    PcController,
)
from .parser import RuleBasedPcActionParser
from .registry import AppDefinition, AppRegistry, default_app_registry

__all__ = [
    "AppDefinition",
    "AppRegistry",
    "PcAction",
    "PcActionParseResult",
    "PcActionParser",
    "PcActionResult",
    "PcActionType",
    "PcController",
    "RuleBasedPcActionParser",
    "default_app_registry",
]
