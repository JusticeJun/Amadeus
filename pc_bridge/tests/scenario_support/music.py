from __future__ import annotations

from dataclasses import dataclass
from itertools import product

from app.music_control import MusicAction, MusicActionResult, MusicActionType


@dataclass(frozen=True)
class MusicExecutionScenario:
    name: str
    actions: tuple[MusicAction, ...]
    fail_at: int | None = None
    failure_reason: str = "backend_unavailable"


class StatefulFakeMusicController:
    """Small state machine for sequence invariants; it never performs real playback."""

    def __init__(self, fail_at: int | None = None, reason: str = "backend_unavailable"):
        self.fail_at = fail_at
        self.reason = reason
        self.calls: list[MusicAction] = []
        self.playing = True
        self.track_index = 0
        self.title = "Initial Track"

    def execute(self, action: MusicAction) -> MusicActionResult:
        index = len(self.calls)
        self.calls.append(action)
        if index == self.fail_at:
            return MusicActionResult(action, False, {"reason": self.reason}, self.reason)
        if action.action_type is MusicActionType.PAUSE:
            self.playing = False
        elif action.action_type is MusicActionType.PLAY:
            self.playing = True
        elif action.action_type is MusicActionType.NEXT:
            self.track_index += 1
            self.title = f"Track {self.track_index}"
        elif action.action_type is MusicActionType.PLAY_SONG:
            self.playing = True
            self.title = action.title
        return MusicActionResult(action, True, {
            "now_playing": {"id": str(self.track_index), "title": self.title,
                            "artist": action.artist or "Artist"},
            "playing": self.playing,
        })


def generated_sequence_scenarios() -> tuple[MusicExecutionScenario, ...]:
    actions = (
        MusicAction(MusicActionType.PAUSE),
        MusicAction(MusicActionType.NEXT),
        MusicAction(MusicActionType.PLAY_SONG, title="Requested Song", artist="Artist"),
    )
    reasons = ("no_match", "ambiguous", "backend_unavailable")
    scenarios: list[MusicExecutionScenario] = []
    for first, second in product(actions, repeat=2):
        pair_name = f"{first.action_type.value}-{second.action_type.value}"
        scenarios.append(MusicExecutionScenario(f"{pair_name}-success", (first, second)))
        for fail_at, reason in product((0, 1), reasons):
            scenarios.append(MusicExecutionScenario(
                f"{pair_name}-fail-{fail_at}-{reason}",
                (first, second), fail_at, reason,
            ))
    return tuple(scenarios)
