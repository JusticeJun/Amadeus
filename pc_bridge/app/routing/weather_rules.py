from __future__ import annotations

import re

from .base import RoutingRequest


def matches_weather_request(request: RoutingRequest) -> bool:
    """Preserve the existing Weather fast-path decision boundary."""
    text = "".join(request.text.lower().split())
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
    media_context = any(word in text for word in (
        "앨범", "영화", "드라마", "소설", "제목", "ost", "뮤직비디오",
    ))

    explicit_subject = any(word in text for word in (
        "날씨", "기온", "온도", "습도", "강수", "일기예보",
    ))
    temperature_expression = any(word in text for word in (
        "덥", "더워", "더운", "더우", "춥", "추워", "추울", "추우", "쌀쌀", "따뜻",
        "선선", "습", "후덥", "몇도",
    ))
    implicit_temperature_question = bool(re.search(
        r"(?:덥|춥|더우|추우|쌀쌀|따뜻|습하|후덥)(?:나|냐|ㄴ가|은가|려나)", text,
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

    if explicit_subject and not media_context and (request_intent or observation_intent) \
            and not (conceptual_context and not (time_context or outdoor_context)):
        return True
    if (time_context or outdoor_context) and temperature_expression \
            and (request_intent or observation_intent or implicit_temperature_question):
        return True
    if precipitation_expression and (time_context or outdoor_context or precipitation_request):
        return True
    return umbrella_expression and (time_context or outdoor_context or request_intent)


def supports_weather_ml_promotion(request: RoutingRequest) -> bool:
    """Reject known non-weather semantic uses before promoting an ML-only weather match."""
    history = " ".join(message.content for message in request.history[-2:])
    text = "".join((history + " " + request.text).lower().split())
    device_damage = any(word in text for word in (
        "컴퓨터", "노트북", "휴대폰", "태블릿", "키보드", "마우스", "스피커", "모니터",
    )) and any(word in text for word in ("젖", "비맞", "비를맞", "빗물", "물들어"))
    media_reference = any(word in text for word in (
        "노래", "음악", "앨범", "곡", "영화", "드라마", "소설", "제목", "ost", "뮤직비디오",
    )) and not any(word in text for word in ("틀어", "재생", "들려"))
    outdoor_activity = any(word in text for word in ("밖에서", "야외", "공원", "광장")) \
        and any(word in text for word in (
            "밥", "식사", "공부", "회의", "친구", "만나", "인터뷰", "발표", "게임", "사진",
        )) and not any(word in text for word in (
            "덥", "춥", "쌀쌀", "따뜻", "비", "눈", "바람", "기온", "날씨", "옷", "우산",
        ))
    mood_metaphor = any(word in text for word in (
        "기분", "마음", "감정", "분위기", "우리사이",
    )) and any(word in text for word in (
        "우중충", "먹구름", "흐린", "흐려", "장마", "싸늘", "찬바람", "축축", "눅눅",
    ))
    unrelated_today = any(word in text for word in (
        "학교가기싫", "수업", "숙제", "할일", "저녁메뉴",
    )) and not any(word in text for word in (
        "날씨", "기온", "덥", "춥", "비", "눈", "바람", "옷", "우산",
    ))
    weather_object = "우산" in text and any(word in text for word in ("부러", "고치", "수리"))
    return not (
        device_damage or media_reference or outdoor_activity or mood_metaphor
        or unrelated_today or weather_object
    )
