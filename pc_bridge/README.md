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
GROQ_MODEL=llama-3.3-70b-versatile
```

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
GPT_SOVITS_TEXT_LANGUAGE=ko
```

Generated speech is cached in `voice/generated/`. Stop the GPT-SoVITS server with Ctrl+C when finished to release GPU memory.

## ESP32 serial state output

Close the PlatformIO serial monitor first, because it and the bridge cannot open COM3 simultaneously. In `pc_bridge/.env`:

```dotenv
AMADEUS_SERIAL_ENABLED=true
AMADEUS_SERIAL_PORT=COM3
AMADEUS_SERIAL_BAUD=115200
```

Or run once with `--serial`. Change `AMADEUS_SERIAL_PORT` if Windows assigns a different port. The state sequence is listening -> thinking -> response emotion -> neutral; `/quit` sends sleep. Missing emotion assets safely keep the neutral image.

Protocol diagnostic (close PlatformIO Monitor first):

```powershell
pc_bridge\.venv\Scripts\python.exe pc_bridge\tools\serial_smoke.py --port COM3
```

## Tests

```powershell
pc_bridge\.venv\Scripts\python.exe -m compileall -q pc_bridge\app pc_bridge\tools pc_bridge\tests
pc_bridge\.venv\Scripts\python.exe -m pytest -q pc_bridge\tests
```

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
