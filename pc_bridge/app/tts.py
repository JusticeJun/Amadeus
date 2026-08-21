from __future__ import annotations

from abc import ABC, abstractmethod
import hashlib
import json
from pathlib import Path
import re
import unicodedata
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
import winsound

from .config import PROJECT_ROOT, Settings
from .models import LlmResult


_TTS_PUNCTUATION_TRANSLATION = str.maketrans({
    "\u00a0": " ",
    "\u1680": " ",
    "\u2000": " ",
    "\u2001": " ",
    "\u2002": " ",
    "\u2003": " ",
    "\u2004": " ",
    "\u2005": " ",
    "\u2006": " ",
    "\u2007": " ",
    "\u2008": " ",
    "\u2009": " ",
    "\u200a": " ",
    "\u202f": " ",
    "\u205f": " ",
    "\u3000": " ",
    "\u2010": "-",
    "\u2011": "-",
    "\u2012": "-",
    "\u2013": "-",
    "\u2014": "-",
    "\u2015": "-",
    "\u2018": "'",
    "\u2019": "'",
    "\u201a": "'",
    "\u201b": "'",
    "\u201c": '"',
    "\u201d": '"',
    "\u201e": '"',
    "\u2032": "'",
    "\u2033": '"',
    "\u2026": "...",
    "\u2022": "-",
    "\u2023": "-",
    "\u2043": "-",
    "\u2212": "-",
    "\u00d7": " x ",
    "\u00f7": " / ",
    "\u2260": " != ",
    "\u2264": " <= ",
    "\u2265": " >= ",
    "\u2248": " ~ ",
    "\u00b1": " +/- ",
    "\u2190": " <- ",
    "\u2192": " -> ",
    "\u2194": " <-> ",
    "\u21d2": " -> ",
    "\u2044": "/",
})


def normalize_tts_text(text: str) -> str:
    """Make LLM typography safe for Windows G2P without changing letters."""
    normalized = unicodedata.normalize("NFKC", text).translate(_TTS_PUNCTUATION_TRANSLATION)
    normalized = _normalize_spoken_measurements(normalized)
    normalized = "".join(
        character
        for character in normalized
        if unicodedata.category(character) not in {"Cc", "Cf", "Cs", "So"}
        or character in "\n\t"
    )
    return re.sub(r"[ \t\r\f\v]+", " ", normalized).strip()


_NUMBER = r"-?\d+(?:\.\d+)?"


def _normalize_spoken_measurements(text: str) -> str:
    replacements = (
        (rf"({_NUMBER})\s*[~\-]\s*({_NUMBER})\s*(?:°?C(?![A-Za-z])|도)",
         lambda match: f"{_number_to_korean(match.group(1))} 도에서 "
                       f"{_number_to_korean(match.group(2))} 도"),
        (rf"({_NUMBER})\s*(?:°?C(?![A-Za-z])|도)",
         lambda match: f"{_number_to_korean(match.group(1))} 도"),
        (rf"({_NUMBER})\s*%",
         lambda match: f"{_number_to_korean(match.group(1))} 퍼센트"),
        (rf"({_NUMBER})\s*m\s*/\s*s(?![A-Za-z])",
         lambda match: f"초속 {_number_to_korean(match.group(1))} 미터"),
        (rf"({_NUMBER})\s*km\s*/\s*h(?![A-Za-z])",
         lambda match: f"시속 {_number_to_korean(match.group(1))} 킬로미터"),
        (rf"({_NUMBER})\s*mm(?![A-Za-z])",
         lambda match: f"{_number_to_korean(match.group(1))} 밀리미터"),
    )
    for pattern, replacement in replacements:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text


def _number_to_korean(value: str) -> str:
    negative = value.startswith("-")
    unsigned = value[1:] if negative else value
    integer, dot, fraction = unsigned.partition(".")
    spoken = _integer_to_korean(int(integer or "0"), speech_spacing=True)
    if dot:
        digits = "영일이삼사오육칠팔구"
        spoken += " 점 " + " ".join(digits[int(digit)] for digit in fraction)
    return ("마이너스 " if negative else "") + spoken


def _integer_to_korean(value: int, *, speech_spacing: bool = False) -> str:
    if value == 0:
        return "영"
    digits = "영일이삼사오육칠팔구"
    small_units = ("", "십", "백", "천")
    large_units = ("", "만", "억", "조")
    groups = []
    while value:
        groups.append(value % 10000)
        value //= 10000
    parts = []
    for group_index in range(len(groups) - 1, -1, -1):
        group = groups[group_index]
        if not group:
            continue
        group_parts = []
        for position in range(3, -1, -1):
            divisor = 10 ** position
            digit = group // divisor % 10
            if not digit:
                continue
            if digit != 1 or position == 0:
                group_parts.append(digits[digit])
            group_parts.append(small_units[position])
        parts.append("".join(group_parts) + large_units[group_index])
    spoken = "".join(parts)
    if speech_spacing:
        # GPT-SoVITS can collapse a glued form such as "이십오" into "이오".
        # A small morpheme boundary is natural when spoken and stabilizes G2P.
        spoken = re.sub(r"([십백천만억조])(?=[일이삼사오육칠팔구])", r"\1 ", spoken)
    return spoken


class TtsError(RuntimeError):
    pass


class TtsEngine(ABC):
    @abstractmethod
    def synthesize(self, result: LlmResult) -> Path | None: ...

    def play(self, audio_path: Path | None) -> None:
        if audio_path is not None:
            winsound.PlaySound(str(audio_path), winsound.SND_FILENAME)


class SilentTtsEngine(TtsEngine):
    def synthesize(self, result: LlmResult) -> Path | None:
        del result
        return None


class GptSovitsEngine(TtsEngine):
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        settings.generated_dir.mkdir(parents=True, exist_ok=True)

    def _references(self) -> list[Path]:
        supported = {".wav", ".flac", ".mp3"}
        references = [
            path.resolve()
            for path in sorted(self._settings.voice_reference_dir.glob("*"))
            if path.suffix.lower() in supported
        ]
        configured = self._settings.gpt_sovits_primary_reference
        if configured:
            configured_path = Path(configured)
            candidates = [configured_path] if configured_path.is_absolute() else [
                PROJECT_ROOT / configured_path,
                self._settings.voice_reference_dir / configured_path,
            ]
            primary = next((path.resolve() for path in candidates if path.is_file()), None)
            if primary is None:
                primary = next((path for path in references if path.name == configured), None)
            if primary is None:
                raise TtsError(
                    f"주 기준 음성을 찾을 수 없습니다: {configured} "
                    f"({self._settings.voice_reference_dir})"
                )
            references = [primary] + [path for path in references if path != primary]
        explicit_aux = [
            item.strip()
            for item in self._settings.gpt_sovits_aux_references.split("|")
            if item.strip()
        ]
        if references and explicit_aux:
            primary = references[0]
            auxiliaries: list[Path] = []
            for configured_aux in explicit_aux:
                match = next((path for path in references[1:] if path.name == configured_aux), None)
                if match is None:
                    candidate = PROJECT_ROOT / configured_aux
                    match = candidate.resolve() if candidate.is_file() else None
                if match is None:
                    raise TtsError(f"보조 기준 음성을 찾을 수 없습니다: {configured_aux}")
                if match != primary and match not in auxiliaries:
                    auxiliaries.append(match)
            references = [primary] + auxiliaries
        elif references and not self._settings.gpt_sovits_use_aux_references:
            references = references[:1]
        return references

    def synthesize(self, result: LlmResult) -> Path:
        references = self._references()
        if not references:
            raise TtsError(f"기준 음성이 없습니다: {self._settings.voice_reference_dir}")
        request_body = {
            "text": normalize_tts_text(result.reply),
            "text_lang": self._settings.gpt_sovits_text_language,
            "ref_audio_path": str(references[0]),
            "aux_ref_audio_paths": [str(path) for path in references[1:]],
            "prompt_text": self._settings.gpt_sovits_prompt_text,
            "prompt_lang": self._settings.gpt_sovits_prompt_language,
            "media_type": "wav",
            "streaming_mode": False,
            # Preserve the approved reference voice's natural prosody. Dynamic
            # LLM speed values made some short replies sound unnatural.
            "speed_factor": self._settings.gpt_sovits_speed_factor,
            "text_split_method": self._settings.gpt_sovits_text_split_method,
            "seed": self._settings.gpt_sovits_seed,
            "top_k": self._settings.gpt_sovits_top_k,
            "top_p": self._settings.gpt_sovits_top_p,
            "temperature": self._settings.gpt_sovits_temperature,
        }
        cache_key = json.dumps(
            {
                "backend": "gpt_sovits",
                "api": self._settings.gpt_sovits_api_url,
                "request": request_body,
                "references": [
                    (str(path), path.stat().st_mtime_ns) for path in references
                ],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        digest = hashlib.sha256(cache_key.encode("utf-8")).hexdigest()[:24]
        output = self._settings.generated_dir / f"gpt_sovits_{digest}.wav"
        if output.exists() and output.stat().st_size >= 44 and output.read_bytes()[:4] == b"RIFF":
            print(f"[gpt-sovits] 캐시 사용: {output.name}")
            return output
        payload = json.dumps(request_body, ensure_ascii=False).encode("utf-8")
        request = Request(
            self._settings.gpt_sovits_api_url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self._settings.gpt_sovits_timeout_seconds) as response:
                audio = response.read()
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise TtsError(f"GPT-SoVITS API 오류 {exc.code}: {detail}") from exc
        except (URLError, TimeoutError) as exc:
            raise TtsError(
                f"GPT-SoVITS 서버에 연결할 수 없습니다: {self._settings.gpt_sovits_api_url}"
            ) from exc
        if len(audio) < 44 or audio[:4] != b"RIFF":
            raise TtsError("GPT-SoVITS가 유효한 WAV를 반환하지 않았습니다.")
        temporary = output.with_suffix(".wav.part")
        temporary.write_bytes(audio)
        temporary.replace(output)
        return output


def create_tts_engine(settings: Settings) -> TtsEngine:
    if settings.tts_engine == "silent":
        return SilentTtsEngine()
    if settings.tts_engine == "gpt_sovits":
        return GptSovitsEngine(settings)
    raise TtsError(f"지원하지 않는 TTS 엔진: {settings.tts_engine}")
