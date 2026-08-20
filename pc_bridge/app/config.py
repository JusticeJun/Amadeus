from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path


BRIDGE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = BRIDGE_ROOT.parent


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    return default if value is None else value.lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    mode: str
    groq_api_key: str
    groq_api_url: str
    groq_model: str
    groq_timeout_seconds: float
    groq_temperature: float
    groq_max_completion_tokens: int
    groq_max_retries: int
    groq_reasoning_effort: str
    tts_engine: str
    serial_enabled: bool
    serial_port: str
    serial_baud: int
    idle_sleep_seconds: int
    voice_reference_dir: Path
    generated_dir: Path
    cache_dir: Path
    gpt_sovits_api_url: str
    gpt_sovits_prompt_language: str
    gpt_sovits_prompt_text: str
    gpt_sovits_text_language: str
    gpt_sovits_timeout_seconds: float
    gpt_sovits_speed_factor: float
    gpt_sovits_text_split_method: str
    gpt_sovits_seed: int
    gpt_sovits_primary_reference: str
    gpt_sovits_use_aux_references: bool
    gpt_sovits_aux_references: str
    gpt_sovits_top_k: int
    gpt_sovits_top_p: float
    gpt_sovits_temperature: float

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv(BRIDGE_ROOT / ".env")
        return cls(
            mode=os.getenv("AMADEUS_MODE", "mock").lower(),
            groq_api_key=os.getenv("GROQ_API_KEY", ""),
            groq_api_url=os.getenv("GROQ_API_URL", "https://api.groq.com/openai/v1/chat/completions"),
            groq_model=os.getenv("GROQ_MODEL", "openai/gpt-oss-20b"),
            groq_timeout_seconds=float(os.getenv("GROQ_TIMEOUT_SECONDS", "30")),
            groq_temperature=float(os.getenv("GROQ_TEMPERATURE", "0.45")),
            groq_max_completion_tokens=int(os.getenv("GROQ_MAX_COMPLETION_TOKENS", "256")),
            groq_max_retries=int(os.getenv("GROQ_MAX_RETRIES", "2")),
            groq_reasoning_effort=os.getenv("GROQ_REASONING_EFFORT", "low").lower(),
            tts_engine=os.getenv("TTS_ENGINE", "gpt_sovits").lower(),
            serial_enabled=_bool("AMADEUS_SERIAL_ENABLED", False),
            serial_port=os.getenv("AMADEUS_SERIAL_PORT", "COM3"),
            serial_baud=int(os.getenv("AMADEUS_SERIAL_BAUD", "115200")),
            idle_sleep_seconds=int(os.getenv("AMADEUS_IDLE_SLEEP_SECONDS", "300")),
            voice_reference_dir=PROJECT_ROOT / "voice" / "references" / "chris",
            generated_dir=PROJECT_ROOT / "voice" / "generated",
            cache_dir=PROJECT_ROOT / "voice" / "cache",
            gpt_sovits_api_url=os.getenv("GPT_SOVITS_API_URL", "http://127.0.0.1:9880/tts"),
            gpt_sovits_prompt_language=os.getenv("GPT_SOVITS_PROMPT_LANGUAGE", "ja"),
            gpt_sovits_prompt_text=os.getenv("GPT_SOVITS_PROMPT_TEXT", ""),
            gpt_sovits_text_language=os.getenv("GPT_SOVITS_TEXT_LANGUAGE", "ko"),
            gpt_sovits_timeout_seconds=float(os.getenv("GPT_SOVITS_TIMEOUT_SECONDS", "120")),
            gpt_sovits_speed_factor=float(os.getenv("GPT_SOVITS_SPEED_FACTOR", "1.0")),
            gpt_sovits_text_split_method=os.getenv("GPT_SOVITS_TEXT_SPLIT_METHOD", "cut1"),
            gpt_sovits_seed=int(os.getenv("GPT_SOVITS_SEED", "42")),
            gpt_sovits_primary_reference=os.getenv("GPT_SOVITS_PRIMARY_REFERENCE", ""),
            gpt_sovits_use_aux_references=_bool("GPT_SOVITS_USE_AUX_REFERENCES", True),
            gpt_sovits_aux_references=os.getenv("GPT_SOVITS_AUX_REFERENCES", ""),
            gpt_sovits_top_k=int(os.getenv("GPT_SOVITS_TOP_K", "5")),
            gpt_sovits_top_p=float(os.getenv("GPT_SOVITS_TOP_P", "0.85")),
            gpt_sovits_temperature=float(os.getenv("GPT_SOVITS_TEMPERATURE", "0.7")),
        )
