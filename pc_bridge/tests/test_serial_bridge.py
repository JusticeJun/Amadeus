import json

from app.serial_bridge import serialize_state


def test_serial_message_is_newline_delimited_json() -> None:
    encoded = serialize_state("wondering")
    assert encoded.endswith(b"\n")
    assert json.loads(encoded) == {"type": "state", "emotion": "wondering"}


def test_unknown_emotion_becomes_neutral() -> None:
    assert json.loads(serialize_state("surprised"))["emotion"] == "neutral"


def test_all_response_emotions_are_serializable() -> None:
    for emotion in ("answering", "happy", "angry", "shy", "sad", "wondering"):
        assert json.loads(serialize_state(emotion))["emotion"] == emotion

