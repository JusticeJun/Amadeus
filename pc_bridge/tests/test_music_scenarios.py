from __future__ import annotations

import pytest

from app.music_control import (
    MusicAction, MusicActionSequence, MusicActionType, MusicSequenceExecutor,
    RuleBasedMusicActionParser,
)
from app.tools import MusicControlTool
from tests.scenario_support.music import (
    MusicExecutionScenario, StatefulFakeMusicController, generated_sequence_scenarios,
)


@pytest.mark.parametrize(("text", "expected"), (
    (
        "지금 재생중인 노래 일시정지하고 백넘버의 크리스마스송 틀어줘",
        (MusicAction(MusicActionType.PAUSE), MusicAction(
            MusicActionType.PLAY_SONG, title="크리스마스송", artist="백넘버",
        )),
    ),
    (
        "다음 곡으로 넘기고 일시정지해줘",
        (MusicAction(MusicActionType.NEXT), MusicAction(MusicActionType.PAUSE)),
    ),
    (
        "마리골드 틀고 바로 일시정지해줘",
        (MusicAction(MusicActionType.PLAY_SONG, title="마리골드"),
         MusicAction(MusicActionType.PAUSE)),
    ),
))
def test_interpretation_scenarios_produce_ordered_actions(text, expected) -> None:
    parsed = RuleBasedMusicActionParser().parse(text)
    assert parsed.actions == expected


def test_song_title_containing_conjunction_is_not_split() -> None:
    parsed = RuleBasedMusicActionParser().parse("아이묭 사랑을 전하고 싶다든가 틀어줘")
    assert len(parsed.actions) == 1
    assert parsed.actions[0].title == "사랑을 전하고 싶다든가"


@pytest.mark.parametrize(
    "scenario", generated_sequence_scenarios(), ids=lambda scenario: scenario.name,
)
def test_generated_execution_scenarios_preserve_invariants(
    scenario: MusicExecutionScenario,
) -> None:
    controller = StatefulFakeMusicController(scenario.fail_at, scenario.failure_reason)
    result = MusicSequenceExecutor(controller).execute(
        MusicActionSequence(scenario.actions),
    )

    assert len(result.results) == len(scenario.actions)
    if scenario.fail_at is None:
        assert result.ok
        assert all(item.ok for item in result.results)
        assert controller.calls == list(scenario.actions)
    else:
        assert not result.ok
        assert result.results[scenario.fail_at].data["reason"] == scenario.failure_reason
        assert controller.calls == list(scenario.actions[:scenario.fail_at + 1])
        for skipped in result.results[scenario.fail_at + 1:]:
            assert not skipped.ok
            assert skipped.data["status"] == "skipped"


def test_tool_aggregates_partial_failure_without_claiming_full_success() -> None:
    controller = StatefulFakeMusicController(fail_at=1, reason="no_match")
    tool = MusicControlTool(RuleBasedMusicActionParser(), controller)

    result = tool.run(
        "지금 재생중인 노래 일시정지하고 백넘버의 크리스마스송 틀어줘",
    )

    assert not result.ok
    assert result.data["status"] == "partial_failure"
    assert [item["status"] for item in result.data["actions"]] == [
        "success", "failed",
    ]
    context = tool.build_llm_context(result)
    assert "success인 동작만 실제로 완료" in context
    assert "전체가 성공했다고" in context
