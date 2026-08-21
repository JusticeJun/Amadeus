from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
from urllib.error import URLError

from app.tools.weather import (
    KmaWeatherTool,
    latlon_to_grid,
    parse_weather_query,
    ultra_forecast_release_candidates,
    ultra_release_candidates,
    village_release_candidates,
)
from app.tools.base import ToolResult


KST = timezone(timedelta(hours=9), "KST")


class FakeResponse:
    def __init__(self, items: list[dict] | None = None, *, code: str = "00") -> None:
        payload = {
            "response": {
                "header": {"resultCode": code, "resultMsg": "OK" if code == "00" else "ERROR"},
                "body": {"items": {"item": items or []}},
            }
        }
        self._body = json.dumps(payload).encode()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self) -> bytes:
        return self._body


def test_routes_weather_questions_without_matching_mood_chat() -> None:
    tool = KmaWeatherTool("key", "Busan", 35.1796, 129.0756)
    for text in (
        "지금 날씨 어때?", "지금 몇 도야?", "오늘 비 와?", "오늘 저녁에 비 와?",
        "내일 날씨 어때?", "내일 추워?", "안녕 크리스 오늘 날씨가 좋네",
        "안녕 크리스 지금 밖에 날씨 좀 알려줄래",
        "현재 기온을 다시 확인해 줘",
        "안녕 크리스 혹시 지금 밖에 좀 더운지 알려줄래",
        "내일 많이 추울까?",
        "바깥 온도가 어느 정도인지 봐줄래?",
        "오늘 오후는 더우려나?",
        "퇴근할 때쯤 쌀쌀할까?",
        "밖에 습한지 궁금해",
        "오늘 비가 오는지 확인해줘",
        "저녁에는 눈이 올까?",
        "내일 강수 확률 알려줘",
        "나갈 때 우산 챙겨야 돼?",
        "오늘 우산 필요할까?",
        "지금 바깥은 선선하니?",
        "오늘은 꽤 덥네",
        "현재 날씨가 흐리네",
        "일기예보 좀 봐줘",
    ):
        assert tool.matches(text), text
    for text in (
        "오늘 기분 어때?", "너 뭐 좋아해?", "비 오는 노래 좋아해?",
        "눈 오는 장면이 예쁜 영화 추천해줘",
        "바람의 노래 들어봤어?",
        "오늘 매운 음식 먹을까?",
        "그 사람 성격이 좀 차갑네",
        "요즘 분위기가 쌀쌀맞아",
        "비와 눈의 차이가 뭐야?",
        "온도라는 단어의 뜻이 뭐야?",
        "내일 일정 알려줘",
        "지금 뭐 하고 있어?",
        "우산이라는 소설 알아?",
        "오늘 기분이 맑아",
    ):
        assert not tool.matches(text), text


def test_query_periods_are_understood() -> None:
    temperature = parse_weather_query("지금 몇 도야?")
    weather = parse_weather_query("지금 날씨 어때?")
    assert temperature.period == "now"
    assert not temperature.include_current_sky
    assert weather.include_current_sky
    assert parse_weather_query("오늘 날씨 어때?").period == "today"
    assert parse_weather_query("오늘 저녁에 비 와?").period == "tonight"
    assert parse_weather_query("내일 비 와?").period == "tomorrow"
    observation = parse_weather_query("안녕 크리스 오늘 날씨가 좋네")
    assert observation.period == "today"
    assert observation.interaction_mode == "casual_observation"


def test_casual_observation_omits_location_that_can_trigger_scene_inference() -> None:
    tool = KmaWeatherTool(
        "key", "Busan_Haeundae", 35.1796, 129.0756,
        opener=lambda *args, **kwargs: FakeResponse([
            {"fcstDate": "20260822", "fcstTime": "1200", "category": "TMP", "fcstValue": "28"},
            {"fcstDate": "20260822", "fcstTime": "1200", "category": "POP", "fcstValue": "20"},
            {"fcstDate": "20260822", "fcstTime": "1200", "category": "PTY", "fcstValue": "0"},
            {"fcstDate": "20260822", "fcstTime": "1200", "category": "SKY", "fcstValue": "3"},
        ]),
        clock=lambda: datetime(2026, 8, 22, 12, 0, tzinfo=KST),
    )
    result = tool.run("안녕 크리스 오늘 날씨가 좋네")
    assert result.ok
    assert result.data["interaction_mode"] == "casual_observation"
    assert "location" not in result.data


def test_busan_coordinates_convert_to_kma_grid() -> None:
    assert latlon_to_grid(35.1796, 129.0756) == (98, 76)


def test_release_candidates_do_not_request_unpublished_runs() -> None:
    early = datetime(2026, 8, 22, 0, 5, tzinfo=KST)
    assert ultra_release_candidates(early)[0] == ("20260821", "2300")
    assert village_release_candidates(early)[0] == ("20260821", "2300")

    morning = datetime(2026, 8, 22, 8, 7, tzinfo=KST)
    # The 08:00 village forecast is not assumed available before 08:10.
    assert village_release_candidates(morning)[0] == ("20260822", "0500")
    assert ultra_forecast_release_candidates(morning)[0] == ("20260822", "0700")


def test_current_observation_is_compacted_for_the_llm() -> None:
    captured_urls = []

    def opener(request, timeout):
        captured_urls.append(request.full_url)
        return FakeResponse([
            {"category": "T1H", "obsrValue": "27.3"},
            {"category": "REH", "obsrValue": "72"},
            {"category": "PTY", "obsrValue": "0"},
            {"category": "RN1", "obsrValue": "0"},
            {"category": "WSD", "obsrValue": "2.1"},
        ])

    tool = KmaWeatherTool(
        "encoded%2Bkey", "Busan", 35.1796, 129.0756,
        opener=opener, clock=lambda: datetime(2026, 8, 22, 14, 25, tzinfo=KST),
    )
    result = tool.run("지금 날씨 어때?")

    assert result.ok
    assert result.data["current"] == {
        "precipitation": "없음", "temperature_c": 27.3,
        "humidity_percent": 72, "wind_speed_mps": 2.1,
        "precipitation_1h_mm": 0,
    }
    assert "forecast" not in result.data
    assert "ServiceKey=encoded%2Bkey" in captured_urls[0]
    assert "getUltraSrtNcst" in captured_urls[0]


def test_tomorrow_forecast_is_aggregated_without_raw_items() -> None:
    def opener(request, timeout):
        return FakeResponse([
            {"fcstDate": "20260823", "fcstTime": "0900", "category": "TMP", "fcstValue": "24"},
            {"fcstDate": "20260823", "fcstTime": "1500", "category": "TMP", "fcstValue": "30"},
            {"fcstDate": "20260823", "fcstTime": "0900", "category": "POP", "fcstValue": "20"},
            {"fcstDate": "20260823", "fcstTime": "1800", "category": "POP", "fcstValue": "70"},
            {"fcstDate": "20260823", "fcstTime": "1800", "category": "PTY", "fcstValue": "1"},
            {"fcstDate": "20260823", "fcstTime": "0900", "category": "SKY", "fcstValue": "3"},
            {"fcstDate": "20260823", "fcstTime": "1800", "category": "SKY", "fcstValue": "4"},
        ])

    tool = KmaWeatherTool(
        "key", "Busan", 35.1796, 129.0756,
        opener=opener, clock=lambda: datetime(2026, 8, 22, 14, 25, tzinfo=KST),
    )
    result = tool.run("내일 비 와?")

    assert result.ok
    forecast = result.data["forecast"]
    assert forecast["temperature_min_c"] == 24
    assert forecast["temperature_max_c"] == 30
    assert forecast["rain_probability_max_percent"] == 70
    assert forecast["precipitation"] == "비"
    assert "items" not in result.data


def test_forecast_falls_back_when_latest_release_lacks_target_period() -> None:
    calls = 0

    def opener(request, timeout):
        nonlocal calls
        calls += 1
        if calls == 1:
            return FakeResponse([
                {"fcstDate": "20260822", "fcstTime": "1200", "category": "TMP", "fcstValue": "28"}
            ])
        return FakeResponse([
            {"fcstDate": "20260823", "fcstTime": "1200", "category": "TMP", "fcstValue": "29"},
            {"fcstDate": "20260823", "fcstTime": "1200", "category": "PTY", "fcstValue": "0"},
        ])

    tool = KmaWeatherTool(
        "key", "Busan", 35.1796, 129.0756,
        opener=opener, clock=lambda: datetime(2026, 8, 22, 14, 25, tzinfo=KST),
    )
    result = tool.run("내일 날씨 어때?")
    assert result.ok
    assert calls == 2
    assert result.data["forecast"]["temperature_max_c"] == 29


def test_generic_current_weather_includes_nearest_ultra_forecast_sky() -> None:
    def opener(request, timeout):
        if "getUltraSrtNcst" in request.full_url:
            return FakeResponse([
                {"category": "T1H", "obsrValue": "26.4"},
                {"category": "PTY", "obsrValue": "0"},
            ])
        return FakeResponse([
            {"fcstDate": "20260822", "fcstTime": "0200", "category": "SKY", "fcstValue": "3"},
            {"fcstDate": "20260822", "fcstTime": "0300", "category": "SKY", "fcstValue": "4"},
        ])

    tool = KmaWeatherTool(
        "key", "Busan", 35.1796, 129.0756,
        opener=opener, clock=lambda: datetime(2026, 8, 22, 2, 37, tzinfo=KST),
    )
    result = tool.run("지금 날씨 어때?")
    assert result.ok
    assert result.data["current"]["sky"] == "흐림"


def test_missing_configuration_and_network_errors_are_isolated() -> None:
    missing = KmaWeatherTool("", "Busan", None, None).run("오늘 날씨 어때?")
    assert not missing.ok
    assert "KMA_SERVICE_KEY" in missing.error

    def failing_opener(request, timeout):
        raise URLError("offline")

    offline = KmaWeatherTool(
        "key", "Busan", 35.1796, 129.0756, opener=failing_opener,
        clock=lambda: datetime(2026, 8, 22, 14, 25, tzinfo=KST),
    ).run("지금 몇 도야?")
    assert not offline.ok
    assert "network" in offline.error


def test_failed_tool_context_does_not_expose_internal_error() -> None:
    result = ToolResult("weather", False, {}, "HTTP 500: private diagnostic")
    context = result.llm_context()
    assert "HTTP" not in context
    assert "private diagnostic" not in context


def test_success_context_forbids_unprovided_weather_inference() -> None:
    context = ToolResult(
        "weather", True, {"location": "Busan", "temperature_c": 27}
    ).llm_context()
    assert "절대 만들어내지 않는다" in context
    assert '"temperature_c":27' in context


def test_invalid_api_payload_isolated_as_failure() -> None:
    class InvalidResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return b"not-json"

    tool = KmaWeatherTool(
        "key", "Busan", 35.1796, 129.0756,
        opener=lambda request, timeout: InvalidResponse(),
        clock=lambda: datetime(2026, 8, 22, 14, 25, tzinfo=KST),
    )
    result = tool.run("오늘 날씨 어때?")
    assert not result.ok
    assert "invalid JSON" in result.error
