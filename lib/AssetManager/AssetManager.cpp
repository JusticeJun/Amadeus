#include "AssetManager.h"

#include <LittleFS.h>
#include <esp_heap_caps.h>

bool AssetManager::begin() {
  return LittleFS.begin(false);
}

bool AssetManager::loadRgb565(const char* path) {
  File file = LittleFS.open(path, FILE_READ);
  if (!file) {
    Serial.printf("Asset open failed:       %s\n", path);
    return false;
  }
  constexpr size_t expectedBytes =
      DisplayManager::WIDTH * DisplayManager::HEIGHT * sizeof(uint16_t);
  if (file.size() != expectedBytes) {
    Serial.printf("Asset size invalid:      %u (expected %u)\n",
                  static_cast<unsigned>(file.size()),
                  static_cast<unsigned>(expectedBytes));
    file.close();
    return false;
  }
  if (imagePixels_ == nullptr) {
    imagePixels_ = static_cast<uint16_t*>(
        heap_caps_malloc(expectedBytes, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT));
  }
  if (imagePixels_ == nullptr) {
    Serial.println("PSRAM image allocation failed.");
    file.close();
    return false;
  }
  const size_t bytesRead =
      file.read(reinterpret_cast<uint8_t*>(imagePixels_), expectedBytes);
  file.close();
  if (bytesRead != expectedBytes) {
    Serial.printf("Asset read incomplete:   %u (expected %u)\n",
                  static_cast<unsigned>(bytesRead),
                  static_cast<unsigned>(expectedBytes));
    return false;
  }
  return true;
}

bool AssetManager::drawRgb565(DisplayManager& display, const char* path) {
  if (!loadRgb565(path)) return false;
  for (int16_t y = 0; y < DisplayManager::HEIGHT; ++y) {
    display.drawRgb565Row(
        y, imagePixels_ + static_cast<size_t>(y) * DisplayManager::WIDTH,
        DisplayManager::WIDTH);
  }
  return true;
}

const uint16_t* AssetManager::imagePixels() const {
  return imagePixels_;
}
