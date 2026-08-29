# Wiring

This is the current ESP32-S3 DevKitC-1 wiring baseline. The LCD and INMP441
assignments are implemented in `include/pins.h`; GPIO16/17 are reserved for
future servos and are not part of the verified microphone capture path.

## ESP32-S3 pin allocation

| GPIO | Current assignment | Direction | Status |
|---|---|---|---|
| GPIO4 | INMP441 BCLK / SCK | ESP32-S3 output | Connected and capture-verified |
| GPIO5 | INMP441 WS / LRCLK | ESP32-S3 output | Connected and capture-verified |
| GPIO6 | INMP441 SD / DOUT | ESP32-S3 input | Connected and capture-verified |
| GPIO8 | ST7796 reset | ESP32-S3 output | Connected |
| GPIO9 | ST7796 data/command | ESP32-S3 output | Connected |
| GPIO10 | ST7796 chip select | ESP32-S3 output | Connected |
| GPIO11 | ST7796 SPI MOSI / SDA | ESP32-S3 output | Connected |
| GPIO12 | ST7796 SPI SCK / SCL | ESP32-S3 output | Connected |
| GPIO16 | Servo 1 | ESP32-S3 output | Reserved; not connected by this phase |
| GPIO17 | Servo 2 | ESP32-S3 output | Reserved; not connected by this phase |

The CH343 USB-C serial/upload interface is onboard and requires no external
GPIO wiring. Native USB/JTAG pins are not reassigned by the connections below.

## ST7796 LCD

| LCD | ESP32-S3 | Note |
|---|---|---|
| GND | GND | Common ground |
| VCC | 3V3 | Module logic and power |
| SCL | GPIO12 | SPI SCK |
| SDA | GPIO11 | SPI MOSI |
| RST | GPIO8 | Reset |
| DC | GPIO9 | Data/command |
| CS | GPIO10 | Chip select |
| BL | 3V3 | Backlight always on |
| SDA-O | Not connected | LCD MISO unused |

Do not connect power until every row has been visually traced on both ends.

## INMP441 microphone

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
connections. This exact left-channel wiring produced a clear 15-second,
16 kHz mono PCM16 hardware capture with no I2S read errors, DMA overruns, or
clipping. Power off the board before changing microphone wiring.
