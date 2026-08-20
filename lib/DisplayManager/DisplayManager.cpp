#include "DisplayManager.h"

void DisplayManager::begin() {
  display_.init();
  display_.setRotation(0);
  // RGB565 assets are stored as native little-endian uint16_t values.
  // ST7796 expects the high byte first on SPI, so swap during image writes.
  display_.setSwapBytes(true);
  display_.fillScreen(TFT_BLACK);
}

void DisplayManager::showSolidColor(uint16_t color) {
  display_.fillScreen(color);
}

void DisplayManager::drawRgb565Row(int16_t y, const uint16_t* pixels,
                                   int16_t count) {
  display_.pushImage(0, y, count, 1, pixels);
}

void DisplayManager::drawRgb565Span(int16_t x, int16_t y,
                                    const uint16_t* pixels, int16_t count) {
  display_.pushImage(x, y, count, 1, pixels);
}

int16_t DisplayManager::width() {
  return display_.width();
}

int16_t DisplayManager::height() {
  return display_.height();
}
