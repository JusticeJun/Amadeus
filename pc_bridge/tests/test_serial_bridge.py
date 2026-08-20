import json

from app.serial_bridge import serialize_state


def test_serial_message_is_newline_delimited_json() -> None:
    encoded = serialize_state("thinking")
    assert encoded.endswith(b"\n")
    assert json.loads(encoded) == {"type": "state", "emotion": "thinking"}


def test_unknown_emotion_becomes_neutral() -> None:
    assert json.loads(serialize_state("angry"))["emotion"] == "neutral"

