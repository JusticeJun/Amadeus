"""Generate conservative emotion samples without changing the adopted voice defaults."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


BRIDGE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = BRIDGE_ROOT.parent
sys.path.insert(0, str(BRIDGE_ROOT))

from app.config import Settings  # noqa: E402


SAMPLES = (
    ("01_answering", "응, 무슨 말인지 알겠어. 그 얘기 조금 흥미롭네.", 1.00),
    ("02_happy", "정말 잘됐네. 나도 기분이 좋아.", 1.02),
    ("03_shy", "갑자기 그렇게 말하면, 조금 부끄럽잖아.", 0.96),
    ("04_angry", "그건 좀 기분 나쁜데. 장난이라도 적당히 해.", 0.97),
    ("05_sad", "그랬구나. 그건 정말 속상했겠다.", 0.96),
    ("06_wondering", "그건 나도 궁금한데. 왜 그렇게 생각했어?", 0.98),
)


def resolve_primary(settings: Settings) -> Path:
    configured = Path(settings.gpt_sovits_primary_reference)
    candidates = [configured] if configured.is_absolute() else [
        PROJECT_ROOT / configured,
        settings.voice_reference_dir / configured,
    ]
    primary = next((path.resolve() for path in candidates if path.is_file()), None)
    if primary is None:
        raise RuntimeError(f"주 기준 음성을 찾을 수 없습니다: {settings.gpt_sovits_primary_reference}")
    return primary


def main() -> int:
    settings = Settings.from_env()
    primary = resolve_primary(settings)
    output_dir = PROJECT_ROOT / "voice" / "generated" / "emotion_voice_test"
    output_dir.mkdir(parents=True, exist_ok=True)
    report = [
        "# 크리스 감정 음성 시험",
        "",
        "기본 음색은 동일하게 유지하고 문장과 속도만 아주 약하게 바꾼 시험본입니다.",
        "이 파일들은 청취 승인 전까지 기본 엔진에 적용되지 않습니다.",
        "",
        "| 파일 | 속도 | 시험 감정 |",
        "|---|---:|---|",
    ]
    for name, text, speed in SAMPLES:
        body = json.dumps({
            "text": text,
            "text_lang": settings.gpt_sovits_text_language,
            "ref_audio_path": str(primary),
            "aux_ref_audio_paths": [],
            "prompt_text": settings.gpt_sovits_prompt_text,
            "prompt_lang": settings.gpt_sovits_prompt_language,
            "media_type": "wav",
            "streaming_mode": False,
            "speed_factor": speed,
            "text_split_method": settings.gpt_sovits_text_split_method,
            "seed": settings.gpt_sovits_seed,
            "top_k": settings.gpt_sovits_top_k,
            "top_p": settings.gpt_sovits_top_p,
            "temperature": settings.gpt_sovits_temperature,
        }, ensure_ascii=False).encode("utf-8")
        request = Request(
            settings.gpt_sovits_api_url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        started = time.perf_counter()
        try:
            with urlopen(request, timeout=settings.gpt_sovits_timeout_seconds) as response:
                audio = response.read()
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            raise RuntimeError(f"GPT-SoVITS HTTP {exc.code}: {detail}") from exc
        except (URLError, TimeoutError) as exc:
            raise RuntimeError(f"GPT-SoVITS 연결 실패: {exc}") from exc
        if len(audio) < 44 or audio[:4] != b"RIFF":
            raise RuntimeError(f"유효하지 않은 WAV: {name}")
        output = output_dir / f"{name}.wav"
        temporary = output.with_suffix(".wav.part")
        temporary.write_bytes(audio)
        temporary.replace(output)
        elapsed = time.perf_counter() - started
        print(f"{name}: {elapsed:.2f}s")
        report.append(f"| `{output.name}` | {speed:.2f} | {name.split('_', 1)[1]} |")
    report.extend([
        "",
        "강한 감정이 필요하면 속도를 더 크게 바꾸기보다 감정별 승인 앵커를 추가하는 것이 안전합니다.",
    ])
    (output_dir / "REPORT.md").write_text("\n".join(report), encoding="utf-8")
    print(f"OUTPUT={output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
