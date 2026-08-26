from __future__ import annotations

from .actions import (
    MusicActionResult, MusicActionSequence, MusicActionSequenceResult,
    MusicController,
)


class MusicSequenceExecutor:
    """Execute a flat, same-capability sequence without planning dependencies."""

    def __init__(self, controller: MusicController) -> None:
        self._controller = controller

    def execute(self, sequence: MusicActionSequence) -> MusicActionSequenceResult:
        results: list[MusicActionResult] = []
        blocked = False
        for action in sequence.actions:
            if blocked:
                results.append(MusicActionResult(
                    action,
                    False,
                    {"reason": "skipped", "status": "skipped"},
                    "skipped after a previous action failed",
                ))
                continue
            result = self._controller.execute(action)
            results.append(result)
            blocked = not result.ok
        return MusicActionSequenceResult(sequence, tuple(results))
