from __future__ import annotations

import ctypes
from ctypes import wintypes
from pathlib import Path
import subprocess
from typing import Callable, Protocol

from .actions import PcAction, PcActionResult, PcActionType, PcController
from .registry import AppRegistry


class VolumeBackend(Protocol):
    def get_volume(self) -> float: ...

    def get_muted(self) -> bool: ...

    def set_volume(self, percent: float) -> None: ...

    def set_muted(self, muted: bool) -> None: ...


class MediaKeySender(Protocol):
    def send(self, action_type: PcActionType) -> None: ...


ProcessLauncher = Callable[[Path], None]
VolumeFactory = Callable[[], VolumeBackend]


class WindowsPcController(PcController):
    def __init__(
        self,
        apps: AppRegistry,
        *,
        process_launcher: ProcessLauncher | None = None,
        volume_factory: VolumeFactory | None = None,
        media_sender: MediaKeySender | None = None,
    ) -> None:
        self._apps = apps
        self._process_launcher = process_launcher or _launch_process
        self._volume_factory = volume_factory or PycawVolumeBackend
        self._media_sender = media_sender or WindowsMediaKeySender()
        self._volume: VolumeBackend | None = None

    def execute(self, action: PcAction) -> PcActionResult:
        try:
            if action.action_type is PcActionType.LAUNCH_APP:
                return self._launch_app(action)
            if action.action_type is PcActionType.ADJUST_VOLUME:
                return self._adjust_volume(action)
            if action.action_type is PcActionType.SET_VOLUME:
                return self._set_volume(action)
            if action.action_type is PcActionType.MUTE:
                self._volume_backend().set_muted(True)
                return PcActionResult(action, True, {"muted": True})
            if action.action_type is PcActionType.UNMUTE:
                self._volume_backend().set_muted(False)
                return PcActionResult(action, True, {"muted": False})
            if action.action_type in {
                PcActionType.MEDIA_PLAY_PAUSE,
                PcActionType.MEDIA_NEXT,
                PcActionType.MEDIA_PREVIOUS,
            }:
                self._media_sender.send(action.action_type)
                return PcActionResult(action, True, {"signal_sent": True})
            return PcActionResult(action, False, {}, "unsupported action")
        except Exception as exc:
            return PcActionResult(action, False, {}, str(exc))

    def _launch_app(self, action: PcAction) -> PcActionResult:
        executable = self._apps.executable_for(action.target)
        if executable is None:
            return PcActionResult(action, False, {}, "allowlisted application is unavailable")
        self._process_launcher(executable)
        return PcActionResult(action, True, {"app": action.target, "launched": True})

    def _adjust_volume(self, action: PcAction) -> PcActionResult:
        backend = self._volume_backend()
        current = backend.get_volume()
        target = _clamp_volume(current + int(action.amount or 0))
        backend.set_volume(target)
        return PcActionResult(action, True, {"volume_percent": round(target)})

    def _set_volume(self, action: PcAction) -> PcActionResult:
        target = _clamp_volume(float(action.amount or 0))
        self._volume_backend().set_volume(target)
        return PcActionResult(action, True, {"volume_percent": round(target)})

    def _volume_backend(self) -> VolumeBackend:
        if self._volume is None:
            self._volume = self._volume_factory()
        return self._volume


class PycawVolumeBackend:
    def __init__(self) -> None:
        try:
            from pycaw.pycaw import AudioUtilities
        except ImportError as exc:
            raise RuntimeError("Windows volume backend is unavailable") from exc
        self._device = AudioUtilities.GetSpeakers()
        self._endpoint = self._device.EndpointVolume

    def get_volume(self) -> float:
        return float(self._endpoint.GetMasterVolumeLevelScalar()) * 100.0

    def get_muted(self) -> bool:
        return bool(self._endpoint.GetMute())

    def set_volume(self, percent: float) -> None:
        self._endpoint.SetMasterVolumeLevelScalar(_clamp_volume(percent) / 100.0, None)

    def set_muted(self, muted: bool) -> None:
        self._endpoint.SetMute(int(muted), None)


class _KEYBDINPUT(ctypes.Structure):
    _fields_ = (
        ("wVk", wintypes.WORD),
        ("wScan", wintypes.WORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", wintypes.WPARAM),
    )


class _MOUSEINPUT(ctypes.Structure):
    _fields_ = (
        ("dx", wintypes.LONG),
        ("dy", wintypes.LONG),
        ("mouseData", wintypes.DWORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", wintypes.WPARAM),
    )


class _HARDWAREINPUT(ctypes.Structure):
    _fields_ = (
        ("uMsg", wintypes.DWORD),
        ("wParamL", wintypes.WORD),
        ("wParamH", wintypes.WORD),
    )


class _INPUT_UNION(ctypes.Union):
    _fields_ = (
        ("mi", _MOUSEINPUT),
        ("ki", _KEYBDINPUT),
        ("hi", _HARDWAREINPUT),
    )


class _INPUT(ctypes.Structure):
    _anonymous_ = ("value",)
    _fields_ = (("type", wintypes.DWORD), ("value", _INPUT_UNION))


class WindowsMediaKeySender:
    _KEYS = {
        PcActionType.MEDIA_PLAY_PAUSE: 0xB3,
        PcActionType.MEDIA_NEXT: 0xB0,
        PcActionType.MEDIA_PREVIOUS: 0xB1,
    }

    def __init__(self, send_input=None) -> None:
        if send_input is None:
            user32 = ctypes.WinDLL("user32", use_last_error=True)
            send_input = user32.SendInput
            send_input.argtypes = (wintypes.UINT, ctypes.POINTER(_INPUT), ctypes.c_int)
            send_input.restype = wintypes.UINT
        self._send_input = send_input

    def send(self, action_type: PcActionType) -> None:
        virtual_key = self._KEYS.get(action_type)
        if virtual_key is None:
            raise ValueError("unsupported media action")
        inputs = (_INPUT * 2)(
            _INPUT(type=1, ki=_KEYBDINPUT(wVk=virtual_key)),
            _INPUT(type=1, ki=_KEYBDINPUT(wVk=virtual_key, dwFlags=0x0002)),
        )
        sent = self._send_input(2, inputs, ctypes.sizeof(_INPUT))
        if sent != 2:
            raise OSError(ctypes.get_last_error(), "SendInput failed")


def _launch_process(executable: Path) -> None:
    subprocess.Popen([str(executable)], shell=False, close_fds=True)


def _clamp_volume(value: float) -> float:
    return min(100.0, max(0.0, value))
