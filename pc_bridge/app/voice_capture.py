from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import wave


@dataclass(frozen=True)
class AudioFrame:
    sample_rate: int
    channels: int
    bits: int
    samples: int
    byte_count: int


@dataclass(frozen=True)
class CaptureSummary:
    samples: int
    byte_count: int
    read_errors: int
    overruns: int
    minimum: int
    maximum: int
    clipped: int


def _parse_fields(line: str, prefix: str) -> dict[str, int]:
    if not line.startswith(prefix + " "):
        raise ValueError(f"expected {prefix}, received {line!r}")
    fields: dict[str, int] = {}
    for item in line[len(prefix) + 1:].split():
        key, separator, value = item.partition("=")
        if not separator or not key:
            raise ValueError(f"malformed {prefix} field: {item!r}")
        fields[key] = int(value)
    return fields


def parse_audio_begin(line: str) -> AudioFrame:
    fields = _parse_fields(line, "AUDIO_BEGIN")
    required = {"rate", "channels", "bits", "samples", "bytes"}
    if fields.keys() != required:
        raise ValueError(f"unexpected AUDIO_BEGIN fields: {sorted(fields)}")
    frame = AudioFrame(
        fields["rate"], fields["channels"], fields["bits"],
        fields["samples"], fields["bytes"],
    )
    if frame.sample_rate <= 0 or frame.channels != 1 or frame.bits != 16:
        raise ValueError(f"unsupported PCM format: {frame}")
    if frame.samples < 0 or frame.byte_count != frame.samples * 2:
        raise ValueError(f"inconsistent PCM length: {frame}")
    return frame


def parse_audio_end(line: str) -> CaptureSummary:
    fields = _parse_fields(line, "AUDIO_END")
    required = {
        "samples", "bytes", "read_errors", "overruns", "min", "max", "clipped",
    }
    if fields.keys() != required:
        raise ValueError(f"unexpected AUDIO_END fields: {sorted(fields)}")
    return CaptureSummary(
        fields["samples"], fields["bytes"], fields["read_errors"],
        fields["overruns"], fields["min"], fields["max"], fields["clipped"],
    )


def validate_capture(frame: AudioFrame, pcm: bytes, summary: CaptureSummary) -> None:
    if len(pcm) != frame.byte_count:
        raise ValueError(f"PCM length mismatch: expected {frame.byte_count}, got {len(pcm)}")
    if summary.samples != frame.samples or summary.byte_count != frame.byte_count:
        raise ValueError("capture summary does not match AUDIO_BEGIN")
    if summary.read_errors or summary.overruns:
        raise ValueError(
            f"firmware reported read_errors={summary.read_errors}, "
            f"overruns={summary.overruns}",
        )


def write_wav(path: Path, frame: AudioFrame, pcm: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as output:
        output.setnchannels(frame.channels)
        output.setsampwidth(frame.bits // 8)
        output.setframerate(frame.sample_rate)
        output.writeframes(pcm)
