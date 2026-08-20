"""Generate deterministic GPT-SoVITS reference/configuration comparison WAVs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import librosa
import numpy as np
import soundfile as sf


BRIDGE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = BRIDGE_ROOT.parent
sys.path.insert(0, str(BRIDGE_ROOT))

from app.config import Settings  # noqa: E402


TEST_SENTENCES = (
    "안녕하세요, 오늘은 무엇을 도와드릴까요?",
    "지금은 조용히 생각을 정리하고 있었어요.",
    "그렇게 말씀하시면 조금 부끄럽네요. 그래도 고마워요.",
    "갑자기 큰 소리가 나서 조금 놀랐어요. 괜찮으세요?",
    "오늘은 천천히 쉬어 가도 괜찮아요.",
)


@dataclass(frozen=True)
class Variant:
    name: str
    primary: Path
    auxiliaries: tuple[Path, ...]
    top_k: int
    top_p: float
    temperature: float
    prompt_language: str = "ja"
    prompt_text: str = ""


def audio_metrics(path: Path) -> dict[str, float]:
    audio, rate = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    duration = len(audio) / rate
    rms = float(np.sqrt(np.mean(np.square(audio)))) if len(audio) else 0.0
    f0, voiced_flag, _ = librosa.pyin(
        audio,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C6"),
        sr=rate,
    )
    voiced = f0[np.isfinite(f0)] if f0 is not None else np.array([])
    return {
        "duration_seconds": round(duration, 3),
        "rms_dbfs": round(20 * math.log10(max(rms, 1e-9)), 2),
        "median_f0_hz": round(float(np.median(voiced)), 2) if len(voiced) else 0.0,
        "f0_std_hz": round(float(np.std(voiced)), 2) if len(voiced) else 0.0,
        "voiced_ratio": round(float(np.mean(voiced_flag)), 3) if voiced_flag is not None else 0.0,
    }


def synthesize(settings: Settings, variant: Variant, text: str, output: Path) -> float:
    if output.exists() and output.stat().st_size >= 44 and output.read_bytes()[:4] == b"RIFF":
        return 0.0
    payload = json.dumps({
        "text": text,
        "text_lang": settings.gpt_sovits_text_language,
        "ref_audio_path": str(variant.primary),
        "aux_ref_audio_paths": [str(path) for path in variant.auxiliaries],
        "prompt_text": variant.prompt_text,
        "prompt_lang": variant.prompt_language,
        "media_type": "wav",
        "streaming_mode": False,
        "speed_factor": 1.0,
        "text_split_method": "cut1",
        "seed": 42,
        "top_k": variant.top_k,
        "top_p": variant.top_p,
        "temperature": variant.temperature,
    }, ensure_ascii=False).encode("utf-8")
    request = Request(
        settings.gpt_sovits_api_url,
        data=payload,
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
        raise RuntimeError("GPT-SoVITS가 유효한 WAV를 반환하지 않았습니다.")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".wav.part")
    temporary.write_bytes(audio)
    temporary.replace(output)
    return time.perf_counter() - started


def variants_for(references: list[Path]) -> list[Variant]:
    primary = references[0]
    variants = [
        Variant("01_all_refs_default", primary, tuple(references[1:]), 15, 1.0, 1.0),
        Variant("02_all_refs_stable", primary, tuple(references[1:]), 5, 0.85, 0.7),
    ]
    variants.extend(
        Variant(f"{index + 10:02d}_ref_{index + 1:02d}_stable", reference, (), 5, 0.85, 0.7)
        for index, reference in enumerate(references)
    )
    return variants


def prepare_primary_clips(references: list[Path]) -> list[Path]:
    """Create non-destructive <=9.5 s derivatives for references too long as primary input."""
    output_dir = PROJECT_ROOT / "voice" / "cache" / "reference_clips"
    output_dir.mkdir(parents=True, exist_ok=True)
    prepared: list[Path] = []
    for index, reference in enumerate(references, 1):
        info = sf.info(reference)
        if 3.0 <= info.duration <= 9.5:
            prepared.append(reference)
            continue
        audio, rate = sf.read(reference, dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        intervals = librosa.effects.split(audio, top_db=32)
        max_frames = int(rate * 9.5)
        best_start, best_end, best_length = 0, 0, 0
        for start_index, (start, _) in enumerate(intervals):
            for _, end in intervals[start_index:]:
                if end - start > max_frames:
                    break
                if end - start > best_length:
                    best_start, best_end = int(start), int(end)
                    best_length = best_end - best_start
        if best_length == 0:
            first_voiced = int(intervals[0][0]) if len(intervals) else 0
            best_start = first_voiced
            best_end = min(len(audio), first_voiced + max_frames)
        clip = np.array(audio[best_start:best_end], copy=True)
        fade = min(int(rate * 0.01), len(clip) // 2)
        if fade:
            clip[:fade] *= np.linspace(0.0, 1.0, fade, dtype=np.float32)
            clip[-fade:] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)
        output = output_dir / f"reference_{index:02d}_primary_clip.wav"
        sf.write(output, clip, rate, subtype="PCM_16")
        prepared.append(output)
    return prepared


def write_report(output_dir: Path, records: list[dict], references: list[Path]) -> None:
    grouped: dict[str, list[dict]] = {}
    for record in records:
        grouped.setdefault(record["variant"], []).append(record)
    lines = [
        "# 크리스 음색 일관성 비교",
        "",
        "음높이 수치는 진단 단서일 뿐이므로 수치만으로 최종 음성을 결정하지 않습니다.",
        "",
        "## 먼저 들어볼 순서",
        "",
        "1. `30_calm_05_anchor_default/` (최종 선호한 15번/05.wav의 차분한 음색 앵커)",
        "2. `31_calm_05_anchor_stable/` (동일 앵커 + 저변동 샘플링)",
        "3. `32_calm_05_anchor_conservative/` (동일 앵커 + 더 보수적인 샘플링)",
        "4. `20_approved_ko_anchor_default/` (이전에 선택한 01.wav 앵커 비교)",
        "",
        "## 기준 음성",
        "",
    ]
    lines.extend(f"- `{path.name}`" for path in references)
    lines.extend(["", "## 후보 요약", "", "| 후보 | 평균 중앙 F0 | 문장 간 F0 편차 | 평균 생성시간 |", "|---|---:|---:|---:|"])
    for variant, items in grouped.items():
        medians = [item["metrics"]["median_f0_hz"] for item in items]
        times = [item["generation_seconds"] for item in items]
        lines.append(
            f"| `{variant}` | {np.mean(medians):.1f} Hz | {np.std(medians):.1f} Hz | {np.mean(times):.2f} s |"
        )
    lines.extend([
        "",
        "## 청취 확인표",
        "",
        "1. 첫 인사에서 쉼표 전후로 동일한 여성 음색이 유지되는가?",
        "2. 어느 문장에서든 낮은 소년 같은 음색으로 변하는가?",
        "3. 두 문장 답변에서도 문장 경계 전후 음색이 일정한가?",
        "4. 발음이 로봇 같거나 지나치게 숨 섞인 소리로 변하지 않는가?",
        "",
    ])
    (output_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    (output_dir / "manifest.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> int:
    settings = Settings.from_env()
    references = sorted(settings.voice_reference_dir.glob("*.wav"))
    if not references:
        print(f"No WAV references: {settings.voice_reference_dir}")
        return 2
    prepared = prepare_primary_clips(references)
    output_dir = PROJECT_ROOT / "voice" / "generated" / "voice_consistency"
    records: list[dict] = []
    variants = variants_for(references)[:2]
    variants.extend(
        Variant(f"{index + 10:02d}_ref_{index + 1:02d}_stable", primary, (), 5, 0.85, 0.7)
        for index, primary in enumerate(prepared)
    )
    if len(prepared) >= 4:
        variants.extend([
            Variant("14_ref_02_plus_04_stable", prepared[1], (references[3],), 5, 0.85, 0.7),
            Variant("15_ref_04_plus_02_stable", prepared[3], (references[1],), 5, 0.85, 0.7),
        ])
    anchor = PROJECT_ROOT / "voice" / "cache" / "anchors" / "chris_approved_ko_anchor.wav"
    if anchor.is_file():
        anchor_text = TEST_SENTENCES[0]
        variants.extend([
            Variant("20_approved_ko_anchor_default", anchor, (), 15, 1.0, 1.0, "ko", anchor_text),
            Variant("21_approved_ko_anchor_stable", anchor, (), 5, 0.85, 0.7, "ko", anchor_text),
            Variant("22_approved_ko_anchor_plus_originals", anchor, tuple(references), 15, 1.0, 1.0, "ko", anchor_text),
            Variant("23_approved_ko_anchor_plus_originals_stable", anchor, tuple(references), 5, 0.85, 0.7, "ko", anchor_text),
        ])
    calm_anchor = PROJECT_ROOT / "voice" / "cache" / "anchors" / "chris_approved_calm_ko_anchor_3s.wav"
    if calm_anchor.is_file():
        calm_text = TEST_SENTENCES[4]
        variants.extend([
            Variant("30_calm_05_anchor_default", calm_anchor, (), 15, 1.0, 1.0, "ko", calm_text),
            Variant("31_calm_05_anchor_stable", calm_anchor, (), 5, 0.85, 0.7, "ko", calm_text),
            Variant("32_calm_05_anchor_conservative", calm_anchor, (), 3, 0.8, 0.6, "ko", calm_text),
        ])
    for variant in variants:
        for sentence_index, text in enumerate(TEST_SENTENCES, 1):
            output = output_dir / variant.name / f"{sentence_index:02d}.wav"
            elapsed = synthesize(settings, variant, text, output)
            metrics = audio_metrics(output)
            records.append({
                "variant": variant.name,
                "sentence": sentence_index,
                "text": text,
                "file": str(output.relative_to(output_dir)),
                "generation_seconds": round(elapsed, 3),
                "configuration": {
                    **asdict(variant),
                    "primary": variant.primary.name,
                    "auxiliaries": [path.name for path in variant.auxiliaries],
                    "seed": 42,
                    "text_split_method": "cut1",
                },
                "metrics": metrics,
            })
            print(f"[{variant.name}] {sentence_index}/{len(TEST_SENTENCES)} -> {output.name}")
    write_report(output_dir, records, references)
    print(f"REPORT={output_dir / 'REPORT.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
