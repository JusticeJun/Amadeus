#pragma once

#include <Arduino.h>

#include "DisplayManager.h"

class AssetManager {
 public:
  bool begin();
  bool drawRgb565(DisplayManager& display, const char* path);
  const uint16_t* imagePixels() const;

 private:
  bool loadRgb565(const char* path);
  uint16_t* imagePixels_ = nullptr;
};
