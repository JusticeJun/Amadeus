#pragma once

#include <Arduino.h>

namespace Pins {
// ST7796 SPI LCD (do not change without rewiring and verification).
constexpr gpio_num_t LCD_SCLK = GPIO_NUM_12;
constexpr gpio_num_t LCD_MOSI = GPIO_NUM_11;
constexpr gpio_num_t LCD_RST  = GPIO_NUM_8;
constexpr gpio_num_t LCD_DC   = GPIO_NUM_9;
constexpr gpio_num_t LCD_CS   = GPIO_NUM_10;

// Reserved for a future INMP441; not initialized in the current firmware.
constexpr gpio_num_t MIC_BCLK = GPIO_NUM_4;
constexpr gpio_num_t MIC_LRCL = GPIO_NUM_5;
constexpr gpio_num_t MIC_DATA = GPIO_NUM_6;

// Reserved for future externally powered servos; not initialized.
constexpr gpio_num_t SERVO_1 = GPIO_NUM_16;
constexpr gpio_num_t SERVO_2 = GPIO_NUM_17;

constexpr gpio_num_t BUILTIN_RGB = GPIO_NUM_48;
}  // namespace Pins

