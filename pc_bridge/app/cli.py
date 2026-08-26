from __future__ import annotations

import argparse

from .config import Settings
from .conversation import ConversationManager
from .input_provider import TextInputProvider
from .llm import GroqLlmClient, LlmError, MockLlmClient
from .music_control import (
    AppleMusicPwaController, CdpAppleMusicBackend, RuleBasedMusicActionParser,
)
from .pc_control import RuleBasedPcActionParser, default_app_registry
from .pc_control.windows import WindowsPcController
from .routing import create_default_semantic_router
from .serial_bridge import SerialBridge
from .tts import TtsError, create_tts_engine
from .tools import KmaWeatherTool, MusicControlTool, PcControlTool, ToolExecutor


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Amadeus PC conversation bridge")
    parser.add_argument("--mode", choices=["mock", "groq"], help="환경변수 AMADEUS_MODE 덮어쓰기")
    parser.add_argument("--tts", choices=["silent", "gpt_sovits"], help="TTS 엔진 덮어쓰기")
    parser.add_argument("--serial", action="store_true", help="USB 시리얼 상태 전송 활성화")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    settings = Settings.from_env()
    if args.mode or args.tts or args.serial:
        values = dict(settings.__dict__)
        if args.mode:
            values["mode"] = args.mode
        if args.tts:
            values["tts_engine"] = args.tts
        if args.serial:
            values["serial_enabled"] = True
        settings = Settings(**values)
    try:
        llm = MockLlmClient() if settings.mode == "mock" else GroqLlmClient(settings)
        tts = create_tts_engine(settings)
    except (LlmError, TtsError) as exc:
        print(f"[config] {exc}")
        return 2
    apps = default_app_registry()
    tools = [KmaWeatherTool(
        settings.kma_service_key,
        settings.amadeus_location_name,
        settings.amadeus_latitude,
        settings.amadeus_longitude,
        settings.weather_timeout_seconds,
    ), PcControlTool(
        RuleBasedPcActionParser(apps),
        WindowsPcController(apps),
    ), MusicControlTool(
        RuleBasedMusicActionParser(),
        AppleMusicPwaController(
            CdpAppleMusicBackend(
                port=settings.apple_music_cdp_port,
                timeout_seconds=settings.apple_music_timeout_seconds,
                chrome_path=settings.apple_music_chrome_path,
                profile_dir=settings.apple_music_profile_dir,
            ),
            playlist_cache_seconds=settings.apple_music_playlist_cache_seconds,
        ),
    )]
    semantic_router = create_default_semantic_router(apps)
    with SerialBridge(settings.serial_enabled, settings.serial_port, settings.serial_baud) as bridge:
        ConversationManager(
            TextInputProvider(), llm, tts, bridge,
            semantic_router=semantic_router,
            tool_executor=ToolExecutor(tools),
        ).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
