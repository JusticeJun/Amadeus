"""Query the configured KMA weather tool without exposing credentials."""

from __future__ import annotations

import json
from pathlib import Path
import sys


BRIDGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BRIDGE_ROOT))

from app.config import Settings  # noqa: E402
from app.tools.weather import KmaWeatherTool  # noqa: E402


QUESTIONS = (
    "지금 날씨 어때?",
    "오늘 날씨 어때?",
    "오늘 저녁에 비 와?",
    "내일 날씨 어때?",
)


def main() -> int:
    settings = Settings.from_env()
    tool = KmaWeatherTool(
        settings.kma_service_key,
        settings.amadeus_location_name,
        settings.amadeus_latitude,
        settings.amadeus_longitude,
        settings.weather_timeout_seconds,
    )
    failed = False
    for question in QUESTIONS:
        result = tool.run(question)
        print(f"QUESTION={question}")
        if result.ok:
            print(json.dumps(result.data, ensure_ascii=False, indent=2))
        else:
            failed = True
            print(f"ERROR={result.error}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
