import json

from app.models import LlmResult


def test_valid_json_and_clamping() -> None:
    result = LlmResult.parse(json.dumps({
        "reply": "괜찮아요.", "emotion": "happy",
        "speech_style": {"speed": 99, "pitch": -99, "energy": 0},
    }))
    assert result.reply == "괜찮아요."
    assert result.emotion == "happy"
    assert result.speech_style.speed == 1.25
    assert result.speech_style.pitch == -6.0
    assert result.speech_style.energy == 0.7


def test_invalid_emotion_falls_back_to_neutral() -> None:
    result = LlmResult.parse('{"reply":"네.","emotion":"angry"}')
    assert result.emotion == "neutral"


def test_malformed_json_recovers_text() -> None:
    result = LlmResult.parse("그냥 대답입니다.")
    assert result.reply == "그냥 대답입니다."
    assert result.emotion == "neutral"

