from __future__ import annotations

import json
from typing import Any

from .models import ALLOWED_EMOTIONS


def serialize_state(emotion: str, event: str = "state") -> bytes:
    safe_emotion = emotion if emotion in ALLOWED_EMOTIONS else "neutral"
    return (json.dumps({"type": event, "emotion": safe_emotion},
                       ensure_ascii=True, separators=(",", ":")) + "\n").encode("ascii")


class SerialBridge:
    def __init__(self, enabled: bool, port: str, baud: int) -> None:
        self._enabled = enabled
        self._port = port
        self._baud = baud
        self._serial: Any = None

    def open(self) -> None:
        if not self._enabled:
            return
        try:
            import serial
            self._serial = serial.Serial(self._port, self._baud, timeout=0.2)
        except Exception as exc:
            self._serial = None
            print(f"[serial] {self._port}를 열지 못했습니다: {exc}")
            print("[serial] PlatformIO 시리얼 모니터가 열려 있다면 먼저 닫으세요.")

    def send_state(self, emotion: str) -> None:
        if self._serial is None:
            return
        try:
            self._serial.write(serialize_state(emotion))
            self._serial.flush()
        except Exception as exc:
            print(f"[serial] 상태 전송 실패: {exc}")

    def close(self) -> None:
        if self._serial is not None:
            self._serial.close()
            self._serial = None

    def __enter__(self) -> "SerialBridge":
        self.open()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

