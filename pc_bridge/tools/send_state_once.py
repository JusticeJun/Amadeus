"""Send one newline-delimited Amadeus display state over USB UART."""

from __future__ import annotations

import argparse
import json
import time

import serial


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("emotion")
    parser.add_argument("--port", default="COM3")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--startup-wait", type=float, default=3.0)
    args = parser.parse_args()

    payload = json.dumps(
        {"type": "state", "emotion": args.emotion},
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("ascii") + b"\n"

    with serial.Serial(args.port, args.baud, timeout=0.2) as connection:
        time.sleep(args.startup_wait)
        connection.write(payload)
        connection.flush()
        time.sleep(0.5)
        received = connection.read(connection.in_waiting).decode(
            "utf-8", errors="replace"
        )

    print(f"sent: {payload.decode('ascii').strip()}")
    if received:
        print(received, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
