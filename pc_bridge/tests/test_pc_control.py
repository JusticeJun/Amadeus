from __future__ import annotations

from pathlib import Path

import pytest

from app.pc_control import (
    AppDefinition,
    AppRegistry,
    PcAction,
    PcActionType,
    RuleBasedPcActionParser,
)
from app.pc_control.windows import WindowsMediaKeySender, WindowsPcController, _launch_process
from app.tools import PcControlTool


@pytest.fixture
def app_registry(tmp_path: Path) -> AppRegistry:
    chrome = tmp_path / "chrome.exe"
    chrome.touch()
    return AppRegistry((
        AppDefinition("chrome", frozenset({"chrome", "크롬"}), (chrome,)),
        AppDefinition("notepad", frozenset({"notepad", "메모장"}), (tmp_path / "missing.exe",)),
    ))


class FakeVolume:
    def __init__(self, volume: float = 50, muted: bool = False) -> None:
        self.volume = volume
        self.muted = muted

    def get_volume(self) -> float:
        return self.volume

    def get_muted(self) -> bool:
        return self.muted

    def set_volume(self, percent: float) -> None:
        self.volume = percent

    def set_muted(self, muted: bool) -> None:
        self.muted = muted


class FakeMediaSender:
    def __init__(self) -> None:
        self.actions: list[PcActionType] = []

    def send(self, action_type: PcActionType) -> None:
        self.actions.append(action_type)


def test_parser_extracts_only_supported_structured_actions(app_registry: AppRegistry) -> None:
    parser = RuleBasedPcActionParser(app_registry)

    assert parser.parse("크롬 켜줘").actions == (
        PcAction(PcActionType.LAUNCH_APP, target="chrome"),
    )
    assert parser.parse("소리 좀 키워줘").actions == (
        PcAction(PcActionType.ADJUST_VOLUME, amount=10),
    )
    assert parser.parse("볼륨 30으로 해줘").actions == (
        PcAction(PcActionType.SET_VOLUME, amount=30),
    )
    assert parser.parse("음소거해줘").actions == (PcAction(PcActionType.MUTE),)
    assert parser.parse("음소거 풀어줘").actions == (PcAction(PcActionType.UNMUTE),)
    assert parser.parse("다음 곡 넘겨줘").actions == (PcAction(PcActionType.MEDIA_NEXT),)
    assert parser.parse("이전 노래로 바꿔줘").actions == (PcAction(PcActionType.MEDIA_PREVIOUS),)
    assert parser.parse("음악 일시정지해줘").actions == (
        PcAction(PcActionType.MEDIA_PLAY_PAUSE),
    )
    assert parser.parse("크롬 좀 띄워줘").actions == (
        PcAction(PcActionType.LAUNCH_APP, target="chrome"),
    )
    assert parser.parse("소리 아예 꺼줘").actions == (PcAction(PcActionType.MUTE),)
    assert parser.parse("한 곡 넘겨줘").actions == (PcAction(PcActionType.MEDIA_NEXT),)
    assert parser.parse("아까 노래로 돌아가줘").actions == (
        PcAction(PcActionType.MEDIA_PREVIOUS),
    )


def test_parser_rejects_unknown_apps_injection_and_invalid_volume(
    app_registry: AppRegistry,
) -> None:
    parser = RuleBasedPcActionParser(app_registry)

    assert parser.parse("포토샵 켜줘").error_code == "unsupported_app"
    assert parser.parse("크롬 켜줘 & calc.exe").error_code == "unsafe_input"
    assert parser.parse("cmd /c calc.exe").error_code == "unsafe_input"
    assert parser.parse("메모장 켜고 C:\\password.txt 열어줘").error_code == "unsafe_input"
    assert parser.parse("크롬 켜면서 --incognito 붙여줘").error_code == "unsafe_input"
    assert parser.parse("크롬 켜고 example.com도 열어줘").error_code == "unsafe_input"
    assert parser.parse("볼륨 101로 해줘").error_code == "invalid_volume"


def test_parser_does_not_treat_discussion_as_pc_actions(app_registry: AppRegistry) -> None:
    parser = RuleBasedPcActionParser(app_registry)

    assert not parser.parse("크롬이랑 엣지 중 뭐가 좋아?").ok
    assert not parser.parse("이 노래 볼륨이 작은 이유가 뭐야?").ok
    assert not parser.parse("다음 곡은 어떤 분위기일까?").ok


def test_controller_uses_allowlisted_path_and_fake_backends(app_registry: AppRegistry) -> None:
    launched: list[Path] = []
    volume = FakeVolume()
    media = FakeMediaSender()
    controller = WindowsPcController(
        app_registry,
        process_launcher=launched.append,
        volume_factory=lambda: volume,
        media_sender=media,
    )

    assert controller.execute(PcAction(PcActionType.LAUNCH_APP, target="chrome")).ok
    assert launched == [app_registry.executable_for("chrome")]
    assert not controller.execute(PcAction(PcActionType.LAUNCH_APP, target="notepad")).ok

    assert controller.execute(PcAction(PcActionType.ADJUST_VOLUME, amount=10)).data == {
        "volume_percent": 60,
    }
    volume.volume = 95
    controller.execute(PcAction(PcActionType.ADJUST_VOLUME, amount=10))
    assert volume.volume == 100
    volume.volume = 5
    controller.execute(PcAction(PcActionType.ADJUST_VOLUME, amount=-10))
    assert volume.volume == 0
    controller.execute(PcAction(PcActionType.SET_VOLUME, amount=30))
    assert volume.volume == 30

    controller.execute(PcAction(PcActionType.MUTE))
    assert volume.muted
    controller.execute(PcAction(PcActionType.UNMUTE))
    assert not volume.muted

    controller.execute(PcAction(PcActionType.MEDIA_PLAY_PAUSE))
    controller.execute(PcAction(PcActionType.MEDIA_NEXT))
    controller.execute(PcAction(PcActionType.MEDIA_PREVIOUS))
    assert media.actions == [
        PcActionType.MEDIA_PLAY_PAUSE,
        PcActionType.MEDIA_NEXT,
        PcActionType.MEDIA_PREVIOUS,
    ]


def test_pc_tool_returns_structured_results_for_character_llm(app_registry: AppRegistry) -> None:
    volume = FakeVolume()
    tool = PcControlTool(
        RuleBasedPcActionParser(app_registry),
        WindowsPcController(
            app_registry,
            process_launcher=lambda path: None,
            volume_factory=lambda: volume,
            media_sender=FakeMediaSender(),
        ),
    )

    result = tool.run("볼륨 30으로 해줘")

    assert result.ok
    assert result.data["actions"] == [{
        "type": "set_volume", "ok": True, "volume_percent": 30,
    }]
    assert '"volume_percent":30' in tool.build_llm_context(result)
    assert volume.volume == 30


def test_windows_media_sender_uses_fixed_key_down_and_key_up_inputs() -> None:
    calls: list[tuple[int, int]] = []

    def fake_send_input(count, inputs, input_size):
        calls.append((count, input_size))
        assert inputs[0].ki.wVk == 0xB3
        assert inputs[0].ki.dwFlags == 0
        assert inputs[1].ki.wVk == 0xB3
        assert inputs[1].ki.dwFlags == 0x0002
        return count

    WindowsMediaKeySender(fake_send_input).send(PcActionType.MEDIA_PLAY_PAUSE)

    assert calls == [(2, 40)]


def test_process_launcher_never_invokes_a_shell(monkeypatch: pytest.MonkeyPatch) -> None:
    observed: list[tuple[list[str], dict[str, object]]] = []

    def fake_popen(args, **kwargs):
        observed.append((args, kwargs))

    monkeypatch.setattr("app.pc_control.windows.subprocess.Popen", fake_popen)

    _launch_process(Path(r"C:\Program Files\Allowed App\app.exe"))

    assert observed == [(
        [r"C:\Program Files\Allowed App\app.exe"],
        {"shell": False, "close_fds": True},
    )]
