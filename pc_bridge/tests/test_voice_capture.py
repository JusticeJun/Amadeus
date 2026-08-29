from pathlib import Path
import struct
import wave

import pytest

from app.voice_capture import (
    CaptureSummary,
    parse_audio_begin,
    parse_audio_end,
    validate_capture,
    write_wav,
)


def test_capture_protocol_writes_standard_pcm16_wav(tmp_path: Path) -> None:
    frame = parse_audio_begin(
        "AUDIO_BEGIN rate=16000 channels=1 bits=16 samples=4 bytes=8",
    )
    pcm = struct.pack("<hhhh", -32768, -10, 10, 32767)
    summary = parse_audio_end(
        "AUDIO_END samples=4 bytes=8 read_errors=0 overruns=0 nonzero=4 "
        "min=-32768 max=32767 clipped=2",
    )

    validate_capture(frame, pcm, summary)
    output = tmp_path / "capture.wav"
    write_wav(output, frame, pcm)

    with wave.open(str(output), "rb") as captured:
        assert captured.getnchannels() == 1
        assert captured.getsampwidth() == 2
        assert captured.getframerate() == 16000
        assert captured.getnframes() == 4
        assert captured.readframes(4) == pcm


@pytest.mark.parametrize(
    "line",
    [
        "AUDIO_BEGIN rate=16000 channels=2 bits=16 samples=4 bytes=8",
        "AUDIO_BEGIN rate=16000 channels=1 bits=24 samples=4 bytes=8",
        "AUDIO_BEGIN rate=16000 channels=1 bits=16 samples=4 bytes=7",
        "AUDIO_BEGIN rate=16000 channels=1 bits=16 samples=-1 bytes=-2",
    ],
)
def test_capture_protocol_rejects_unsupported_or_inconsistent_frames(line: str) -> None:
    with pytest.raises(ValueError):
        parse_audio_begin(line)


def test_capture_protocol_rejects_transport_or_i2s_errors() -> None:
    frame = parse_audio_begin(
        "AUDIO_BEGIN rate=16000 channels=1 bits=16 samples=2 bytes=4",
    )
    with pytest.raises(ValueError, match="PCM length mismatch"):
        validate_capture(
            frame,
            b"\0\0",
            CaptureSummary(2, 4, 0, 0, 2, 0, 0, 0),
        )
    with pytest.raises(ValueError, match="read_errors=1, overruns=2"):
        validate_capture(
            frame,
            b"\0\0\0\0",
            CaptureSummary(2, 4, 1, 2, 2, 0, 0, 0),
        )


def test_capture_protocol_rejects_dma_sample_loss_when_reads_succeed() -> None:
    frame = parse_audio_begin(
        "AUDIO_BEGIN rate=16000 channels=1 bits=16 samples=2 bytes=4",
    )

    with pytest.raises(ValueError, match="read_errors=0, overruns=5"):
        validate_capture(
            frame,
            b"\0\0\0\0",
            CaptureSummary(2, 4, 0, 5, 2, -10, 10, 0),
        )


def test_capture_protocol_rejects_exactly_zero_dead_input() -> None:
    frame = parse_audio_begin(
        "AUDIO_BEGIN rate=16000 channels=1 bits=16 samples=2 bytes=4",
    )

    with pytest.raises(ValueError, match="dead input: all PCM samples are zero"):
        validate_capture(
            frame,
            b"\0\0\0\0",
            CaptureSummary(2, 4, 0, 0, 0, 0, 0, 0),
        )


def test_capture_protocol_verifies_nonzero_count_against_pcm() -> None:
    frame = parse_audio_begin(
        "AUDIO_BEGIN rate=16000 channels=1 bits=16 samples=2 bytes=4",
    )

    with pytest.raises(ValueError, match="firmware=2, pcm=1"):
        validate_capture(
            frame,
            b"\1\0\0\0",
            CaptureSummary(2, 4, 0, 0, 2, 0, 1, 0),
        )
