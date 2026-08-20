from __future__ import annotations

from abc import ABC, abstractmethod
import hashlib
import json
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
import winsound

from .config import PROJECT_ROOT, Settings
from .models import LlmResult


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
            "text": result.reply,
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
