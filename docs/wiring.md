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

