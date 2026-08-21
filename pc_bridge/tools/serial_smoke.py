from __future__ import annotations

import argparse
import json
import time

import serial


def main() -> int:
    parser = argparse.ArgumentParser(description="Exercise the ESP32 line JSON protocol")
    parser.add_argument("--port", default="COM3")
    parser.add_argument("--baud", type=int, default=115200)
    args = parser.parse_args()
    commands = [
        {"type": "state", "emotion": "wondering"},
        {"type": "state", "emotion": "happy"},
        {"type": "state", "emotion": "unsupported"},
    ]
    with serial.Serial(args.port, args.baud, timeout=0.1) as connection:
        time.sleep(2.8)
        connection.reset_input_buffer()
        for command in commands:
            connection.write((json.dumps(command, separators=(",", ":")) + "\n").encode("ascii"))
            connection.flush()
            time.sleep(0.15)
        connection.write(b"not-json\n")
        connection.write(b"{" + b"x" * 200 + b"}\n")
        connection.flush()
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            line = connection.readline()
            if line:
                print(line.decode("utf-8", errors="replace").rstrip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

