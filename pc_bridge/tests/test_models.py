import json

from app.models import ChatMessage, LlmResult, is_repetitive_reply


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


def test_device_only_state_is_not_accepted_as_reply_emotion() -> None:
    result = LlmResult.parse('{"reply":"잘 자요.","emotion":"sleep"}')
    assert result.emotion == "neutral"


def test_reply_is_cleaned_and_safely_shortened() -> None:
    result = LlmResult.parse(json.dumps({
        "reply": "**알겠어요.** " + ("아주 긴 설명 " * 50),
        "emotion": "neutral",
    }))
    assert "*" not in result.reply
    assert len(result.reply) <= 241


def test_reply_is_limited_to_two_spoken_sentences() -> None:
    result = LlmResult.parse(json.dumps({
        "reply": "첫 번째 문장입니다. 두 번째 문장입니다. 세 번째 문장입니다.",
        "emotion": "neutral",
    }))
    assert result.reply == "첫 번째 문장입니다. 두 번째 문장입니다."


def test_direct_user_echo_is_detected() -> None:
    assert is_repetitive_reply("지금은 아주 좋은데?", "지금은 아주 좋은데?", [])


def test_recent_assistant_repetition_is_detected() -> None:
    history = [ChatMessage("assistant", "지금은 사용자를 기다리고 있어요.")]
    assert is_repetitive_reply("지금은 사용자님을 기다리고 있어요.", "무슨 생각 해?", history)


def test_genuinely_new_reply_is_allowed() -> None:
    history = [ChatMessage("assistant", "지금은 사용자를 기다리고 있어요.")]
    assert not is_repetitive_reply("애니메이션 이야기를 조금 더 해 볼까요?", "뭐 해?", history)

