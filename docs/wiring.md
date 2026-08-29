# Wiring

## ST7796 LCD

| LCD | ESP32-S3 | Note |
|---|---|---|
| GND | GND | Common ground |
| VCC | 3V3 | Module logic/power per stated 3–5 V specification |
| SCL | GPIO12 | SPI SCK |
| SDA | GPIO11 | SPI MOSI |
| RST | GPIO8 | Reset |
| DC | GPIO9 | Data/command |
| CS | GPIO10 | Chip select |
| BL | 3V3 | Backlight always on |
| SDA-O | Not connected | LCD MISO unused |

Do not connect power until every row has been visually traced on both ends.

## INMP441 microphone (Phase 1 diagnostic)

| INMP441 | ESP32-S3 | Note |
|---|---|---|
| VDD | 3V3 | Do not use 5 V |
| GND | GND | Common ground |
| L/R | GND | Selects the left I2S slot |
| SCK / BCLK | GPIO4 | I2S bit clock from ESP32-S3 |
| WS / LRCLK | GPIO5 | I2S word select from ESP32-S3 |
| SD / DOUT | GPIO6 | I2S microphone data into ESP32-S3 |

GPIO4/5/6 do not overlap the LCD, CH343 UART, native USB/JTAG, boot
strapping pins, future servo reservation, or the N16R8 module's octal-memory
connections. Power off the board before changing microphone wiring.
