#pragma once

#include <Arduino.h>

namespace ProjectConfig {
constexpr char PROJECT_NAME[] = "Amadeus";
constexpr uint32_t SERIAL_BAUD = 115200;
constexpr uint32_t STARTUP_SERIAL_WAIT_MS = 1500;

namespace Crt {
constexpr bool ENABLED = true;
constexpr uint16_t REFRESH_INTERVAL_MS = 45;
constexpr uint16_t GLITCH_CHANCE_PER_THOUSAND = 45;
constexpr uint8_t GLITCH_MAX_REGIONS = 4;
constexpr uint8_t GLITCH_MIN_WIDTH = 40;
constexpr uint16_t GLITCH_MAX_WIDTH = 190;
constexpr uint8_t GLITCH_MAX_HEIGHT = 8;
constexpr uint8_t GLITCH_MAX_SHIFT = 9;
constexpr uint8_t SCANLINE_DIM_PERCENT = 94;
constexpr uint8_t EDGE_DIM_PERCENT = 94;
constexpr uint8_t EDGE_WIDTH_PIXELS = 14;
constexpr uint16_t SYNC_TEAR_CHANCE_PER_THOUSAND = 8;
constexpr uint8_t SYNC_TEAR_MIN_HEIGHT = 8;
constexpr uint8_t SYNC_TEAR_MAX_HEIGHT = 22;
constexpr uint8_t SYNC_TEAR_MIN_SHIFT = 18;
constexpr uint8_t SYNC_TEAR_MAX_SHIFT = 48;
}  // namespace Crt

namespace HappyHearts {
constexpr uint16_t FRAME_INTERVAL_MS = 800;
constexpr int16_t DIRTY_SIZE = 48;
}  // namespace HappyHearts
}  // namespace ProjectConfig
