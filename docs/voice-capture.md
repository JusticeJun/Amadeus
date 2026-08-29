# INMP441 Phase 1 capture diagnostic

This diagnostic validates only the physical INMP441-to-ESP32-S3 I2S path and
PC WAV capture. It does not implement AEC, VAD, STT, wake-word detection, or
conversation integration.

## Phase 1 status

Phase 1 hardware acceptance was completed on 2026-08-29 using the wiring in
this document. A 15-second spoken capture produced:

- `read_errors=0`
- `overruns=0`
- `nonzero=239909` of 240000 samples
- `min=-8721`, `max=7360`
- `clipped=0`

The generated 16 kHz mono PCM16 WAV was manually reviewed. Speech was clear
and intelligible with correct speed and pitch, and there were no audible
dropouts, repeated segments, severe digital noise, or sample-format corruption.
This result validates the Phase 1 raw-capture path only; it does not add or
validate AEC, VAD, STT, wake-word detection, or production voice input.

## Audio format

- INMP441 source: Philips I2S, signed 24-bit data in a 32-bit slot
- Channel: left (`L/R` connected to GND)
- Sample rate: 16 kHz; generated directly by the ESP32-S3 I2S clock
- WAV output: mono signed PCM16 at 16 kHz

The firmware receives each 32-bit slot, sign-extends the valid upper 24 bits,
then discards the least-significant eight valid bits to produce PCM16. No
sample-rate conversion or digital gain is applied.

The diagnostic installs and starts a fresh I2S driver immediately before each
capture, then stops and uninstalls it before PCM is sent to the PC. The RX queue
therefore exists only during active capture. `overruns` counts RX DMA buffers
actually discarded during that interval, not overflow events accumulated
while idle.

Mono PCM16 at 16 kHz is 32,000 bytes/s. The I2S driver reads the microphone's
32-bit slots at 64,000 bytes/s, while 921600-baud UART with 8N1 framing has a
theoretical payload ceiling of 92,160 bytes/s. PCM is buffered in PSRAM and
transmitted only after capture, so serial backpressure is outside the I2S read
loop.

## Wiring

Power the board off before wiring.

| INMP441 | ESP32-S3 DevKitC-1 |
|---|---|
| VDD | 3V3 |
| GND | GND |
| L/R | GND |
| SCK / BCLK | GPIO4 |
| WS / LRCLK | GPIO5 |
| SD / DOUT | GPIO6 |

## Build, upload, and record

From the repository root in PowerShell:

```powershell
$env:PLATFORMIO_SETTING_ENABLE_TELEMETRY='No'
& "$env:USERPROFILE\.platformio\penv\Scripts\pio.exe" run -e esp32-s3-inmp441-capture
& "$env:USERPROFILE\.platformio\penv\Scripts\pio.exe" run -e esp32-s3-inmp441-capture -t upload
cd pc_bridge
.\.venv\Scripts\python.exe tools\capture_inmp441.py `
  --port COM3 `
  --seconds 15 `
  --output captures\inmp441-voice-15s.wav
```

Close every serial monitor and stop the normal PC bridge before running the
capture utility. Only one process can own COM3. The utility resets the UART
input, requests a bounded capture, reads the exact advertised PCM byte count,
checks the firmware summary, and writes:

```text
pc_bridge\captures\inmp441-voice-15s.wav
```

Use `--output <path>` to select an explicit WAV path. Captures are limited to
1 through 30 seconds. The diagnostic UART runs at 921600 baud; the production
firmware and JSON protocol remain at 115200 baud.

## Observability and acceptance

The firmware reports the selected pins, 16 kHz sample rate, 24-bit data in a
32-bit left slot, requested and captured sample counts, PCM byte count, I2S
read errors, DMA overruns, minimum/maximum PCM values, and clipped samples.
It also reports the number of samples that are not exactly zero. The PC checks
that count against the received PCM and rejects an all-zero capture as a dead
diagnostic input. This is an exact transport/data-path integrity check, not an
amplitude threshold or a production silence detector: a functioning digital
microphone produces some noise codes even in a quiet environment.

A healthy spoken capture has all of the following properties:

- the utility completes without a timeout or protocol error;
- `read_errors=0` and `overruns=0`;
- `nonzero` is greater than zero and agrees with the received PCM payload;
- the WAV is mono PCM16 at 16 kHz and matches the requested duration;
- silence has low-amplitude noise rather than a constant full-scale value;
- speech is intelligible at normal playback volume;
- the waveform is centered around zero, responds clearly to speech, and is
  not continuously flat, rail-clipped, byte-swapped, or dominated by periodic
  digital noise;
- `min` and `max` normally have both negative and positive values, while
  `clipped` remains zero or very small during ordinary speech.

These checks require a physical microphone recording. A successful build or
deterministic PC-side test does not establish that the microphone wiring,
channel selection, or sample alignment works on hardware.

After the Phase 1 recording, restore the normal firmware before running the PC
bridge again:

```powershell
cd ..
$env:PLATFORMIO_SETTING_ENABLE_TELEMETRY='No'
& "$env:USERPROFILE\.platformio\penv\Scripts\pio.exe" run -e esp32-s3-n16r8-diagnostic -t upload
```
