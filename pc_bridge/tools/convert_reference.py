from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert a voice source to PCM 16-bit mono WAV")
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    source = args.input.resolve()
    if not source.is_file():
        parser.error(f"input does not exist: {source}")
    output = args.output or (Path(__file__).resolve().parents[2] / "voice" / "references" / "chris" /
                             f"{source.stem}_reference.wav")
    output = output.resolve()
    if output.exists():
        parser.error(f"output already exists; refusing to overwrite: {output}")
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        parser.error("ffmpeg is not installed or not on PATH")
    output.parent.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run([
        ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error",
        "-i", str(source), "-vn", "-ac", "1", "-ar", "24000",
        "-c:a", "pcm_s16le", str(output),
    ], check=False)
    if completed.returncode != 0:
        if output.exists():
            output.unlink()
        return completed.returncode
    print(f"Created {output}; original preserved at {source}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

