from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import time

import serial

from app.voice_capture import (
    parse_audio_begin,
    parse_audio_end,
    validate_capture,
    write_wav,
)


def read_line(connection: serial.Serial, deadline: float) -> str:
    while time.monotonic() < deadline:
        raw = connection.readline()
        if raw:
            line = raw.decode("ascii", errors="replace").strip()
            if line:
                print(line)
                return line
    raise TimeoutError("timed out waiting for firmware response")


def read_until_prefix(connection: serial.Serial, prefix: str, deadline: float) -> str:
    while True:
        line = read_line(connection, deadline)
        if line.startswith("AUDIO_ERROR"):
            raise RuntimeError(line)
        if line.startswith(prefix):
            return line


def wait_for_ready(connection: serial.Serial, deadline: float) -> bool:
    while time.monotonic() < deadline:
        raw = connection.readline()
        if not raw:
            continue
        line = raw.decode("ascii", errors="replace").strip()
        if not line:
            continue
        print(line)
        if line.startswith("AUDIO_ERROR"):
            raise RuntimeError(line)
        if line.startswith("INMP441_READY"):
            return True
    return False


def read_exact(connection: serial.Serial, byte_count: int, deadline: float) -> bytes:
    data = bytearray()
    while len(data) < byte_count and time.monotonic() < deadline:
        chunk = connection.read(byte_count - len(data))
        if chunk:
            data.extend(chunk)
    if len(data) != byte_count:
        raise TimeoutError(f"expected {byte_count} PCM bytes, received {len(data)}")
    return bytes(data)


def default_output() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return Path("captures") / f"inmp441-{timestamp}.wav"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Capture INMP441 diagnostic PCM from the ESP32-S3 into a WAV file",
    )
    parser.add_argument("--port", default="COM3")
    parser.add_argument("--baud", type=int, default=921600)
    parser.add_argument("--seconds", type=int, default=5, choices=range(1, 31))
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    output = args.output or default_output()

    with serial.Serial(args.port, args.baud, timeout=0.2) as connection:
        if not wait_for_ready(connection, time.monotonic() + 4.0):
            print("[voice] readiness banner not observed; requesting capture from running firmware")
        connection.write(f"CAPTURE {args.seconds}\n".encode("ascii"))
        connection.flush()
        deadline = time.monotonic() + args.seconds + 10.0
        begin_line = read_until_prefix(connection, "AUDIO_BEGIN", deadline)
        frame = parse_audio_begin(begin_line)
        pcm = read_exact(connection, frame.byte_count, deadline)
        end_line = read_until_prefix(connection, "AUDIO_END", deadline)
        summary = parse_audio_end(end_line)

    validate_capture(frame, pcm, summary)
    write_wav(output, frame, pcm)
    print(f"WAV: {output.resolve()}")
    print(
        "Capture: "
        f"{frame.sample_rate} Hz, mono PCM{frame.bits}, {frame.samples} samples, "
        f"min={summary.minimum}, max={summary.maximum}, clipped={summary.clipped}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
