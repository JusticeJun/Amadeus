# Roadmap

1. Upload board diagnostics and verify 16 MB Flash / 8 MB PSRAM in UART logs.
2. ST7796 solid-color test at 10 MHz SPI (verified on hardware).
3. Approved neutral image converted to RGB565 and displayed from LittleFS (verified on hardware: orientation, size, crop, and colors correct).
4. Default display profile: persistent scanline/edge CRT base, intermittent small tears, and rare wide horizontal-sync displacement bands (verified on hardware; active whenever the display is on).
5. Later: microphone, networking, speech/LLM, emotions, externally powered servos, sleep.

## PC bridge progress

- Keyboard input, bounded conversation memory, mock LLM, and structured emotion output implemented.
- Groq LLM client implemented; live verification awaits a user-provided API key.
- GPT-SoVITS v2 selected as the default TTS engine after Korean zero-shot validation; silent mode remains for tests.
- Isolated Python 3.11 CUDA runtime and RTX 4060 inference verified for GPT-SoVITS.
- ESP32 newline JSON emotion protocol built, uploaded, and tested on COM3 with neutral-image fallback.
- Authorized Japanese references and multiple Korean zero-shot validation sentences verified.
