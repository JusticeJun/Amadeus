"""Structured Apple Music control domain."""

from .actions import (
    MusicAction, MusicActionParseResult, MusicActionParser, MusicActionResult,
    MusicActionSequence, MusicActionSequenceResult, MusicActionType, MusicController,
)
from .cdp import CdpAppleMusicBackend
from .controller import (
    AppleMusicBackend, AppleMusicPwaController, MusicControlError, MusicItem,
    PersonalMusicItem, PersonalMusicSnapshot, PlaylistItem, PlaylistSnapshot,
)
from .parser import RuleBasedMusicActionParser
from .sequence import MusicSequenceExecutor

__all__ = [
    "AppleMusicBackend", "AppleMusicPwaController", "CdpAppleMusicBackend",
    "MusicAction", "MusicActionParseResult", "MusicActionParser", "MusicActionResult",
    "MusicActionSequence", "MusicActionSequenceResult", "MusicActionType",
    "MusicControlError", "MusicController", "MusicItem",
    "PersonalMusicItem", "PersonalMusicSnapshot", "PlaylistItem", "PlaylistSnapshot",
    "RuleBasedMusicActionParser", "MusicSequenceExecutor",
]
