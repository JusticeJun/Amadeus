#pragma once

#include <Arduino.h>
#include <TFT_eSPI.h>

class DisplayManager {
 public:
  static constexpr int16_t WIDTH = 320;
  static constexpr int16_t HEIGHT = 480;

  void begin();
  void showSolidColor(uint16_t color);
  void drawRgb565Row(int16_t y, const uint16_t* pixels, int16_t count);
  void drawRgb565Span(int16_t x, int16_t y, const uint16_t* pixels,
                      int16_t count);
  int16_t width();
  int16_t height();

 private:
  TFT_eSPI display_;
};
