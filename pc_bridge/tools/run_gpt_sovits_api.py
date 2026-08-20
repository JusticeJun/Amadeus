from __future__ import annotations

import argparse
import os
from pathlib import Path
import runpy
import sys

import soundfile as sf
import torch
import torchaudio


def load_pcm_reference(path, *args, **kwargs):
    del args, kwargs
    samples, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    return torch.from_numpy(samples.T.copy()), sample_rate


def main() -> None:
    parser = argparse.ArgumentParser(description="Run GPT-SoVITS API with Windows PCM loader")
    parser.add_argument("repository", type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9880)
    args = parser.parse_args()
    repository = args.repository.resolve()
    api = repository / "api_v2.py"
    if not api.is_file():
        parser.error(f"GPT-SoVITS api_v2.py not found: {api}")
    torchaudio.load = load_pcm_reference
    os.chdir(repository)
    sys.path.insert(0, str(repository))
    sys.argv = [str(api), "-a", args.host, "-p", str(args.port),
                "-c", "GPT_SoVITS/configs/tts_infer.yaml"]
    runpy.run_path(str(api), run_name="__main__")


if __name__ == "__main__":
    main()
