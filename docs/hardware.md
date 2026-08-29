# Hardware

- MCU: ESP32-S3-WROOM-1-N16R8 (16 MB Flash, 8 MB Octal PSRAM)
- Board: ESP32-S3 DevKitC-1
- Upload/serial: CH343P UART USB-C `COM` port, 115200 baud
- Display: ST7796, SPI, 320×480 portrait; touch is not used
- Phase 1 microphone: INMP441 at 3.3 V; GPIO4 BCLK, GPIO5 WS, GPIO6 data
- Future servos: two MG996R; GPIO 16/17 reserved and external power required

Hardware capacities must be accepted only after upload and runtime diagnostics.

## Verified diagnostic result (2026-08-20)

- Upload port: COM3, USB VID:PID `1A86:55D3`, CH343
- Chip: ESP32-S3, 2 cores, 240 MHz
- Flash: 16,777,216 bytes (16.00 MiB)
- PSRAM: detected, 8,386,279 bytes reported (8.00 MiB)
- Free PSRAM at startup: 8,386,035 bytes (8.00 MiB)
- Free internal heap at startup: 372,868 bytes
- Largest free internal block: 335,860 bytes

These values were captured from the uploaded firmware over the CH343 UART, not inferred from the build configuration.

## Verified display result (2026-08-20)

- ST7796 initialized over HSPI at 10 MHz
- 320×480 portrait geometry verified
- Solid red, green, blue, white, and black screens verified
- Approved neutral character loaded from LittleFS as RGB565
- Image orientation, dimensions, cropping, and colors verified on the physical LCD
