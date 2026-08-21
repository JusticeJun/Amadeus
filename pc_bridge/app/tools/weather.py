from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import json
import math
import re
import socket
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Callable

from .base import ToolResult


KST = timezone(timedelta(hours=9), "KST")
KMA_BASE_URL = "https://apis.data.go.kr/1360000/VilageFcstInfoService_2.0"
VILLAGE_RELEASES = ("0200", "0500", "0800", "1100", "1400", "1700", "2000", "2300")
SKY_CODES = {"1": "맑음", "3": "구름많음", "4": "흐림"}
PTY_CODES = {"0": "없음", "1": "비", "2": "비/눈", "3": "눈", "4": "소나기"}


class WeatherError(RuntimeError):
    pass


@dataclass(frozen=True)
class WeatherQuery:
    period: str
    include_current: bool
    include_current_sky: bool = False
    interaction_mode: str = "information_request"


class KmaWeatherTool:
    name = "weather"

    def __init__(
        self,
        service_key: str,
        location_name: str,
        latitude: float | None,
        longitude: float | None,
        timeout_seconds: float = 8.0,
        *,
        opener: Callable[..., object] = urllib.request.urlopen,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._service_key = service_key.strip()
        self._location_name = location_name.strip() or "default"
        self._timeout = timeout_seconds
        self._opener = opener
        self._clock = clock or (lambda: datetime.now(KST))
        self._grid = (
            latlon_to_grid(latitude, longitude)
            if latitude is not None and longitude is not None else None
        )

    def matches(self, user_text: str) -> bool:
        text = "".join(user_text.lower().split())
        time_context = any(word in text for word in (
            "지금", "현재", "오늘", "내일", "저녁", "밤", "아침", "오후", "퇴근",
        ))
        outdoor_context = any(word in text for word in ("밖", "바깥", "외부", "나갈", "외출"))
        request_intent = any(word in text for word in (
            "어때", "어떻", "알려", "봐줘", "확인", "궁금", "몇도",
            "할까", "올까", "오는지", "내릴까", "필요", "챙겨", "입어야",
        )) or text.endswith(("?", "니", "냐", "지"))
        observation_intent = any(word in text for word in (
            "좋네", "좋다", "맑네", "흐리네", "덥네", "춥네", "쌀쌀하네",
            "따뜻하네", "선선하네", "습하네", "후덥지근",
        ))
        conceptual_context = any(word in text for word in (
            "단어", "뜻", "의미", "차이", "개념", "원리", "설명해",
        ))

        explicit_subject = any(word in text for word in (
            "날씨", "기온", "온도", "습도", "강수", "일기예보",
        ))
        temperature_expression = any(word in text for word in (
            "덥", "더워", "더운", "더우", "춥", "추워", "추울", "추우", "쌀쌀", "따뜻",
            "선선", "습", "후덥", "몇도",
        ))
        precipitation_expression = bool(re.search(
            r"(?:비|눈)(?:가|는|이)?(?:와|오|올|내리|예보|확률)", text
        ))
        precipitation_request = any(word in text for word in (
            "비올까", "비가올까", "비오는지", "비가오는지", "비와?", "비가와?",
            "눈올까", "눈이올까", "눈오는지", "눈이오는지", "눈와?", "눈이와?",
            "내릴까", "강수확률", "비예보", "눈예보",
        ))
        umbrella_expression = "우산" in text and any(
            word in text for word in ("필요", "챙", "가져", "써야")
        )

        if explicit_subject and (request_intent or observation_intent) \
                and not (conceptual_context and not (time_context or outdoor_context)):
            return True
        if (time_context or outdoor_context) and temperature_expression \
                and (request_intent or observation_intent):
            return True
        if precipitation_expression and (time_context or outdoor_context or precipitation_request):
            return True
        return umbrella_expression and (time_context or outdoor_context or request_intent)

    def run(self, user_text: str) -> ToolResult:
        if not self._service_key:
            return ToolResult(self.name, False, {}, "KMA_SERVICE_KEY is empty")
        if self._grid is None:
            return ToolResult(self.name, False, {}, "default location coordinates are missing")
        try:
            now = self._now()
            query = parse_weather_query(user_text)
            data: dict[str, Any] = {
                "requested_period": query.period,
                "interaction_mode": query.interaction_mode,
                "reference_time": now.isoformat(timespec="minutes"),
            }
            # A casual weather remark does not need a place name. Omitting it also
            # prevents the LLM from inventing nearby scenery or activities.
            if query.interaction_mode != "casual_observation":
                data["location"] = self._location_name
            if query.include_current:
                data["current"] = self._current(now, query.include_current_sky)
            if query.period != "now":
                data["forecast"] = self._forecast(now, query.period)
            return ToolResult(self.name, True, data)
        except WeatherError as exc:
            return ToolResult(self.name, False, {}, str(exc))
        except Exception as exc:  # Never let a tool terminate the bridge.
            return ToolResult(self.name, False, {}, f"unexpected weather error: {exc}")

    def _now(self) -> datetime:
        value = self._clock()
        return value.astimezone(KST) if value.tzinfo else value.replace(tzinfo=KST)

    def _current(self, now: datetime, include_sky: bool) -> dict[str, Any]:
        items = self._request_latest("getUltraSrtNcst", ultra_release_candidates(now))
        values = {str(item.get("category")): item.get("obsrValue") for item in items}
        result: dict[str, Any] = {
            "precipitation": PTY_CODES.get(str(values.get("PTY", "0")), "알 수 없음")
        }
        _put_number(result, "temperature_c", values.get("T1H"))
        _put_number(result, "humidity_percent", values.get("REH"))
        _put_number(result, "wind_speed_mps", values.get("WSD"))
        _put_number(result, "precipitation_1h_mm", values.get("RN1"))
        if include_sky:
            forecast_items = self._request_latest(
                "getUltraSrtFcst", ultra_forecast_release_candidates(now)
            )
            sky_value = _nearest_forecast_value(forecast_items, "SKY", now)
            if sky_value is not None:
                result["sky"] = SKY_CODES.get(str(sky_value), "알 수 없음")
        if len(result) == 1 and result["precipitation"] == "알 수 없음":
            raise WeatherError("current observation data is missing")
        return result

    def _forecast(self, now: datetime, period: str) -> dict[str, Any]:
        target = now + timedelta(days=1) if period == "tomorrow" else now
        target_date = target.strftime("%Y%m%d")
        allowed_hours = range(18, 24) if period == "tonight" else range(24)
        selected: list[dict[str, Any]] = []
        last_error = f"forecast data is missing for {target_date}/{period}"
        for base_date, base_time in village_release_candidates(now):
            try:
                items = self._request("getVilageFcst", base_date, base_time)
            except WeatherError as exc:
                last_error = str(exc)
                if last_error.startswith("authentication error"):
                    break
                continue
            selected = _select_forecast(items, target_date, allowed_hours)
            if selected:
                break
        if not selected:
            raise WeatherError(last_error)

        grouped: dict[str, list[Any]] = {}
        for item in selected:
            grouped.setdefault(str(item.get("category")), []).append(item.get("fcstValue"))
        result: dict[str, Any] = {"date": target_date, "part_of_day": period}
        temperatures = _numbers(grouped.get("TMP", []))
        if temperatures:
            result["temperature_min_c"] = min(temperatures)
            result["temperature_max_c"] = max(temperatures)
        rain_probabilities = _numbers(grouped.get("POP", []))
        if rain_probabilities:
            result["rain_probability_max_percent"] = max(rain_probabilities)
        precipitation = [PTY_CODES.get(str(value), "알 수 없음") for value in grouped.get("PTY", [])]
        wet = [value for value in precipitation if value not in {"없음", "알 수 없음"}]
        result["precipitation"] = Counter(wet).most_common(1)[0][0] if wet else "없음"
        skies = [SKY_CODES.get(str(value), "알 수 없음") for value in grouped.get("SKY", [])]
        known_skies = [value for value in skies if value != "알 수 없음"]
        if known_skies:
            result["sky"] = Counter(known_skies).most_common(1)[0][0]
        return result

    def _request_latest(self, endpoint: str,
                        candidates: list[tuple[str, str]]) -> list[dict[str, Any]]:
        last_error = "no release candidate"
        for base_date, base_time in candidates:
            try:
                items = self._request(endpoint, base_date, base_time)
                if items:
                    return items
                last_error = f"empty response for {base_date} {base_time}"
            except WeatherError as exc:
                last_error = str(exc)
                if last_error.startswith("authentication error"):
                    break
        raise WeatherError(last_error)

    def _request(self, endpoint: str, base_date: str,
                 base_time: str) -> list[dict[str, Any]]:
        if self._grid is None:
            raise WeatherError("weather grid is unavailable")
        nx, ny = self._grid
        params = urllib.parse.urlencode({
            "pageNo": 1, "numOfRows": 1000, "dataType": "JSON",
            "base_date": base_date, "base_time": base_time, "nx": nx, "ny": ny,
        })
        # Accept either the decoded or percent-encoded key shown by data.go.kr.
        key = urllib.parse.quote(self._service_key, safe="%")
        url = f"{KMA_BASE_URL}/{endpoint}?ServiceKey={key}&{params}"
        request = urllib.request.Request(url, headers={"User-Agent": "Amadeus-PC-Bridge/0.1"})
        try:
            with self._opener(request, timeout=self._timeout) as response:
                raw = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:200]
            raise WeatherError(f"HTTP {exc.code}: {detail}") from exc
        except (urllib.error.URLError, TimeoutError, socket.timeout) as exc:
            raise WeatherError(f"network timeout/error: {exc}") from exc
        try:
            payload = json.loads(raw)
            header = payload["response"]["header"]
            code = str(header.get("resultCode"))
            if code != "00":
                message = str(header.get("resultMsg") or "unknown API error")
                kind = "authentication" if code in {"20", "22", "30", "31"} else "API"
                raise WeatherError(f"{kind} error {code}: {message}")
            items = payload["response"]["body"]["items"]["item"]
        except WeatherError:
            raise
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            raise WeatherError("invalid JSON response") from exc
        if not isinstance(items, list):
            raise WeatherError("invalid weather item list")
        return [item for item in items if isinstance(item, dict)]


def parse_weather_query(text: str) -> WeatherQuery:
    compact = "".join(text.lower().split())
    if "내일" in compact:
        return WeatherQuery("tomorrow", False)
    if "저녁" in compact or "밤" in compact:
        return WeatherQuery("tonight", False)
    current = any(word in compact for word in ("지금", "현재", "몇도"))
    observation = (
        "날씨" in compact
        and any(word in compact for word in (
            "좋네", "좋다", "좋아", "맑네", "흐리네", "덥네", "춥네",
            "따뜻", "쌀쌀", "선선", "습하", "후덥",
        ))
    )
    return WeatherQuery(
        "now" if current else "today",
        current,
        current and "날씨" in compact,
        "casual_observation" if observation else "information_request",
    )


def ultra_release_candidates(now: datetime, count: int = 3) -> list[tuple[str, str]]:
    latest = now.replace(minute=0, second=0, microsecond=0) - timedelta(hours=1)
    return [
        ((latest - timedelta(hours=index)).strftime("%Y%m%d"),
         (latest - timedelta(hours=index)).strftime("%H00"))
        for index in range(count)
    ]


def ultra_forecast_release_candidates(now: datetime,
                                      count: int = 4) -> list[tuple[str, str]]:
    safe = now - timedelta(minutes=45)
    minute = 30 if safe.minute >= 30 else 0
    latest = safe.replace(minute=minute, second=0, microsecond=0)
    return [
        ((latest - timedelta(minutes=30 * index)).strftime("%Y%m%d"),
         (latest - timedelta(minutes=30 * index)).strftime("%H%M"))
        for index in range(count)
    ]


def village_release_candidates(now: datetime, count: int = 4) -> list[tuple[str, str]]:
    safe_now = now - timedelta(minutes=10)
    candidates: list[tuple[str, str]] = []
    cursor = safe_now
    while len(candidates) < count:
        date = cursor.strftime("%Y%m%d")
        available = [release for release in VILLAGE_RELEASES if release <= cursor.strftime("%H%M")]
        candidates.extend((date, release) for release in reversed(available))
        cursor = (cursor - timedelta(days=1)).replace(hour=23, minute=59)
    return candidates[:count]


def latlon_to_grid(latitude: float, longitude: float) -> tuple[int, int]:
    """Convert WGS84 coordinates to the KMA 5 km Lambert grid."""
    radius = 6371.00877 / 5.0
    slat1, slat2, olon, olat = map(math.radians, (30.0, 60.0, 126.0, 38.0))
    sn = math.log(math.cos(slat1) / math.cos(slat2)) / math.log(
        math.tan(math.pi * 0.25 + slat2 * 0.5) /
        math.tan(math.pi * 0.25 + slat1 * 0.5)
    )
    sf = math.tan(math.pi * 0.25 + slat1 * 0.5) ** sn * math.cos(slat1) / sn
    ro = radius * sf / math.tan(math.pi * 0.25 + olat * 0.5) ** sn
    ra = radius * sf / math.tan(math.pi * 0.25 + math.radians(latitude) * 0.5) ** sn
    theta = math.radians(longitude) - olon
    if theta > math.pi:
        theta -= 2.0 * math.pi
    if theta < -math.pi:
        theta += 2.0 * math.pi
    theta *= sn
    return (
        math.floor(ra * math.sin(theta) + 43.0 + 0.5),
        math.floor(ro - ra * math.cos(theta) + 136.0 + 0.5),
    )


def _numbers(values: list[Any]) -> list[float]:
    result = []
    for value in values:
        try:
            result.append(float(value))
        except (TypeError, ValueError):
            continue
    return result


def _put_number(target: dict[str, Any], key: str, value: Any) -> None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return
    target[key] = int(number) if number.is_integer() else number


def _nearest_forecast_value(items: list[dict[str, Any]], category: str,
                            now: datetime) -> Any | None:
    candidates = []
    for item in items:
        if str(item.get("category")) != category:
            continue
        try:
            forecast_at = datetime.strptime(
                str(item["fcstDate"]) + str(item["fcstTime"]), "%Y%m%d%H%M"
            ).replace(tzinfo=KST)
        except (KeyError, ValueError):
            continue
        candidates.append((abs((forecast_at - now).total_seconds()), item.get("fcstValue")))
    return min(candidates, default=(0, None), key=lambda pair: pair[0])[1]


def _select_forecast(items: list[dict[str, Any]], target_date: str,
                     allowed_hours: range) -> list[dict[str, Any]]:
    selected = []
    for item in items:
        if str(item.get("fcstDate")) != target_date:
            continue
        try:
            hour = int(str(item.get("fcstTime", "0000"))[:2])
        except ValueError:
            continue
        if hour in allowed_hours:
            selected.append(item)
    return selected
