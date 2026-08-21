# Amadeus PC Bridge

The PC bridge keeps network, conversation, and speech workloads on Windows. The ESP32 remains responsible for the LCD, character state, CRT effects, and future servos.

```text
keyboard -> mock/Groq LLM -> validated reply + emotion
         -> GPT-SoVITS speech -> PC speaker
         -> newline JSON over COM3 -> ESP32 character state
```

INMP441 capture and Groq Whisper STT are intentionally placeholders until the microphone arrives.

## First installation

From the project root in PowerShell:

```powershell
winget install --id astral-sh.uv --exact
uv python install 3.14
uv sync --project pc_bridge --extra dev
```

## Activate the virtual environment

```powershell
pc_bridge\.venv\Scripts\Activate.ps1
```

Activation is optional when using the explicit interpreter path shown below.

## Mock mode

Mock mode needs no API key, voice reference, network, or ESP32:

```powershell
pc_bridge\.venv\Scripts\python.exe -m app.cli --mode mock --tts silent
```

Type `/quit` or press Ctrl+C to stop. Run the same command again to restart.

## Groq configuration

Create a Groq account and API key yourself. Copy `.env.example` to `pc_bridge/.env`, then put the key only in the untracked file:

```dotenv
AMADEUS_MODE=groq
GROQ_API_KEY=your_key_here
GROQ_MODEL=openai/gpt-oss-20b
```

The response model uses low reasoning effort to reduce spoken-response latency. A single request generates both the reply and its emotion, and each turn prints model time, TTS time, time-to-audio, and token usage. Override `GROQ_REASONING_EFFORT` only after comparing latency and response quality.

Never put the key in source code, `platformio.ini`, screenshots, commits, or issue reports. The API URL, model and timeout can all be changed in `.env`. HTTP errors, timeouts, rate limits, and malformed replies are reported without terminating the interactive bridge.

## Voice references

Use only recordings that you own or have explicit permission to clone.

1. Preserve Japanese M4A originals in `voice/source/`.
2. Install FFmpeg: `winget install --id Gyan.FFmpeg --exact`.
3. Convert without overwriting the original:

```powershell
pc_bridge\.venv\Scripts\python.exe pc_bridge\tools\convert_reference.py voice\source\sample.m4a
```

The tool creates 24kHz PCM 16-bit mono WAV in `voice/references/chris/` and refuses to overwrite an existing output. Multiple WAV files in that folder are passed together to GPT-SoVITS.

## GPT-SoVITS zero-shot speech

GPT-SoVITS v2 is the approved default TTS engine. It runs as an isolated local API because its supported Python environment differs from the PC bridge. The local third-party repository, virtual environment, model weights, private references, and generated speech are all ignored by Git. Amadeus uses prompt-free cross-language inference with Japanese references (`prompt_lang=ja`, empty `prompt_text`) and Korean output (`text_lang=ko`). No fine-tuning is performed.

Start the locally installed API from the repository root:

```powershell
tools\gpt-sovits-runtime\.venv\Scripts\python.exe pc_bridge\tools\run_gpt_sovits_api.py tools\GPT-SoVITS
```

Keep that terminal open. In a second terminal, select the backend in `pc_bridge/.env`:

```dotenv
TTS_ENGINE=gpt_sovits
GPT_SOVITS_API_URL=http://127.0.0.1:9880/tts
GPT_SOVITS_PROMPT_LANGUAGE=ja
GPT_SOVITS_PROMPT_TEXT=
GPT_SOVITS_TEXT_LANGUAGE=ko
GPT_SOVITS_SPEED_FACTOR=1.0
GPT_SOVITS_TEXT_SPLIT_METHOD=cut1
GPT_SOVITS_SEED=42
GPT_SOVITS_PRIMARY_REFERENCE=
GPT_SOVITS_USE_AUX_REFERENCES=true
GPT_SOVITS_AUX_REFERENCES=
GPT_SOVITS_TOP_K=5
GPT_SOVITS_TOP_P=0.85
GPT_SOVITS_TEMPERATURE=0.7
```

Short replies are kept in one synthesis group (`cut1`) so a comma does not split the voice into noticeably different tones. A fixed seed makes inference reproducible. Add more reference recordings only after comparing this deterministic baseline; more files are not automatically better.

The locally approved production voice may use a derived Korean anchor with its exact transcript. Keep that private anchor under `voice/references/anchors/`; it is excluded from Git just like the original recordings. The public `.env.example` intentionally contains no private filename or transcript.

## Voice stability

Early prompt-free cross-language tests occasionally shifted from the intended soft female timbre to a sharper or lower boy-like timbre depending on the Korean sentence. The adopted local configuration fixes that regression by using a user-approved Korean anchor and its exact transcript, one continuous synthesis group, a fixed seed, and no auxiliary-reference mixing. This keeps the selected Chris voice reproducible across the tested sentence set.

The approved recording, derived anchor, transcript, comparison WAVs, and generated cache remain private and are never committed. The repository contains only the engine integration, configurable controls, diagnostics, and regression tests. Conservative emotion samples use small text and speed changes; stronger emotions should use separately approved emotion anchors rather than large pitch or speed shifts.

Generated speech is cached in `voice/generated/`. Stop the GPT-SoVITS server with Ctrl+C when finished to release GPU memory.

## ESP32 serial state output

Close the PlatformIO serial monitor first, because it and the bridge cannot open COM3 simultaneously. In `pc_bridge/.env`:

```dotenv
AMADEUS_SERIAL_ENABLED=true
AMADEUS_SERIAL_PORT=COM3
AMADEUS_SERIAL_BAUD=115200
```

Or run once with `--serial`. Change `AMADEUS_SERIAL_PORT` if Windows assigns a different port. The display stays neutral while idle, switches to listening only while input is active, then shows `answering`, `happy`, `angry`, `shy`, `sad`, or `wondering` for the reply before returning to neutral. `/quit` sends sleep.

Protocol diagnostic (close PlatformIO Monitor first):

```powershell
pc_bridge\.venv\Scripts\python.exe pc_bridge\tools\serial_smoke.py --port COM3
```

## Tests

```powershell
pc_bridge\.venv\Scripts\python.exe -m compileall -q pc_bridge\app pc_bridge\tools pc_bridge\tests
pc_bridge\.venv\Scripts\python.exe -m pytest -q pc_bridge\tests
```

## KMA weather tool

Apply for **기상청_단기예보 조회서비스** on data.go.kr and add its service key,
the fixed device location, and WGS84 coordinates to `pc_bridge/.env`:

```dotenv
KMA_SERVICE_KEY=your_service_key
AMADEUS_LOCATION_NAME=Busan
AMADEUS_LATITUDE=35.1796
AMADEUS_LONGITUDE=129.0756
WEATHER_TIMEOUT_SECONDS=8
```

Run the bridge normally. Weather questions such as `지금 날씨 어때?`,
`오늘 저녁에 비 와?`, and `내일 날씨 어때?` use the KMA tool. Ordinary
conversation bypasses it. The tool returns compact facts only; Groq still writes
Chris's final reply and selects the emotion. See `docs/tools.md` for the extension
contract used by future tools.

## Common problems

- `COM3 access denied`: close PlatformIO Monitor or another serial program.
- `GROQ_API_KEY가 없습니다`: use mock mode or create `pc_bridge/.env` from the example.
- No reference WAV: place an authorized source in `voice/source/` and run the conversion tool.
- `ffmpeg is not installed`: install FFmpeg and open a new terminal so PATH refreshes.
- CUDA unavailable: confirm the NVIDIA driver and isolated GPT-SoVITS runtime installation.
- GPT-SoVITS connection refused: start the local API before the PC bridge.
- No speaker sound: check the Windows output device and play a generated WAV manually.

## Future microphone path

After the INMP441 is installed and its current hardware stage is verified, `VoiceInputProvider` and `GroqWhisperEngine` will replace keyboard input and the mock STT. The LLM, TTS, cache, serial protocol, and ESP32 character path remain unchanged.
