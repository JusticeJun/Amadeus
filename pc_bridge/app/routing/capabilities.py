from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CapabilityDefinition:
    name: str
    side_effecting: bool
    description: str
    ml_fallback_enabled: bool


CAPABILITIES = (
    CapabilityDefinition("weather", False, "Current weather and forecast lookup", True),
    CapabilityDefinition("music_control", True, "Music playback and transport control", False),
    CapabilityDefinition("pc_control", True, "Local PC application and system control", False),
)

CAPABILITY_NAMES = tuple(item.name for item in CAPABILITIES)
CAPABILITY_BY_NAME = {item.name: item for item in CAPABILITIES}
