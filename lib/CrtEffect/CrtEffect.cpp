#include "CrtEffect.h"

#include <esp_heap_caps.h>
#include <esp_system.h>

bool CrtEffect::begin(DisplayManager& display, const uint16_t* source,
                      const CrtEffectConfig& config) {
  display_ = &display;
  source_ = source;
  config_ = config;
  bandY_ = -1;
  lastUpdateMs_ = millis();
  if (source_ == nullptr || !buildAmbientBase()) return false;
  for (int16_t y = 0; y < DisplayManager::HEIGHT; ++y) {
    display_->drawRgb565Row(
        y, ambientPixels_ + static_cast<size_t>(y) * DisplayManager::WIDTH,
        DisplayManager::WIDTH);
  }
  return true;
}

const uint16_t* CrtEffect::effectSource() const {
  return ambientPixels_ != nullptr ? ambientPixels_ : source_;
}

bool CrtEffect::buildAmbientBase() {
  constexpr size_t pixelCount =
      static_cast<size_t>(DisplayManager::WIDTH) * DisplayManager::HEIGHT;
  if (ambientPixels_ == nullptr) {
    ambientPixels_ = static_cast<uint16_t*>(heap_caps_malloc(
        pixelCount * sizeof(uint16_t), MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT));
  }
  if (ambientPixels_ == nullptr) return false;

  for (int16_t y = 0; y < DisplayManager::HEIGHT; ++y) {
    for (int16_t x = 0; x < DisplayManager::WIDTH; ++x) {
      const uint16_t pixel =
          source_[static_cast<size_t>(y) * DisplayManager::WIDTH + x];
      uint16_t scale = (y & 1) ? config_.scanlineDimPercent : 100;
      if (x < config_.edgeWidthPixels ||
          x >= DisplayManager::WIDTH - config_.edgeWidthPixels) {
        scale = static_cast<uint16_t>(scale * config_.edgeDimPercent / 100U);
      }
      const uint16_t red = ((pixel >> 11) & 0x1F) * scale / 100U;
      const uint16_t green = ((pixel >> 5) & 0x3F) * scale / 100U;
      const uint16_t blue = (pixel & 0x1F) * scale / 100U;
      ambientPixels_[static_cast<size_t>(y) * DisplayManager::WIDTH + x] =
          static_cast<uint16_t>((red << 11) | (green << 5) | blue);
    }
  }
  return true;
}

void CrtEffect::setEnabled(bool enabled) {
  if (!enabled) {
    restoreGlitches();
    restoreStatic();
    restoreBlockGlitch();
    restoreSyncTear();
  }
  config_.enabled = enabled;
  if (!enabled) bandY_ = -1;
}

void CrtEffect::update() {
  if (!config_.enabled || display_ == nullptr || source_ == nullptr ||
      config_.mode != CrtMode::Ambient) {
    return;
  }
  const uint32_t now = millis();
  if (now - lastUpdateMs_ < config_.refreshIntervalMs) return;
  lastUpdateMs_ = now;

  restoreGlitches();
  restoreStatic();
  restoreBlockGlitch();
  restoreSyncTear();
  drawStaticFrame();

  if (glitchFramesRemaining_ == 0 &&
      (esp_random() % 1000U) < config_.glitchChancePerThousand) {
    glitchFramesRemaining_ = 2 + (esp_random() % 3U);
  }
  if (glitchFramesRemaining_ > 0) {
    drawGlitchBurstFrame();
    --glitchFramesRemaining_;
  }

  if (blockFramesRemaining_ == 0 &&
      (esp_random() % 1000U) < config_.blockGlitchChancePerThousand) {
    blockFramesRemaining_ = 2 + (esp_random() % 2U);
  }
  if (blockFramesRemaining_ > 0) {
    drawBlockGlitchFrame();
    --blockFramesRemaining_;
  }

  if (syncTearFramesRemaining_ == 0 &&
      (esp_random() % 1000U) < config_.syncTearChancePerThousand) {
    syncTearFramesRemaining_ = 1 + (esp_random() % 2U);
  }
  if (syncTearFramesRemaining_ > 0) {
    drawSyncTearFrame();
    --syncTearFramesRemaining_;
  }
}

void CrtEffect::restoreBand(int16_t startY) {
  for (uint8_t row = 0; row < config_.rollingBandHeight; ++row) {
    const int16_t y = (startY + row) % DisplayManager::HEIGHT;
    display_->drawRgb565Row(
        y, effectSource() + static_cast<size_t>(y) * DisplayManager::WIDTH,
        DisplayManager::WIDTH);
  }
}

void CrtEffect::drawAmbientBand(int16_t startY) {
  int8_t shift = 0;
  if (config_.horizontalJitterPixels > 0 && (esp_random() % 24U) == 0U) {
    shift = (esp_random() & 1U) ? config_.horizontalJitterPixels
                                : -config_.horizontalJitterPixels;
  }
  for (uint8_t row = 0; row < config_.rollingBandHeight; ++row) {
    const int16_t y = (startY + row) % DisplayManager::HEIGHT;
    const uint16_t* sourceRow =
        effectSource() + static_cast<size_t>(y) * DisplayManager::WIDTH;
    const bool scanline = (y & 1) != 0;
    for (int16_t x = 0; x < DisplayManager::WIDTH; ++x) {
      const int16_t sourceX = constrain(x - shift, 0, DisplayManager::WIDTH - 1);
      rowBuffer_[x] = alterPixel(sourceRow[sourceX], scanline);
    }
    display_->drawRgb565Row(y, rowBuffer_, DisplayManager::WIDTH);
  }
}

void CrtEffect::restoreGlitches() {
  for (uint8_t index = 0; index < glitchRegionCount_; ++index) {
    const GlitchRegion& region = glitchRegions_[index];
    for (int16_t row = 0; row < region.height; ++row) {
      const int16_t y = region.y + row;
      const uint16_t* sourceRow =
          effectSource() + static_cast<size_t>(y) * DisplayManager::WIDTH + region.x;
      display_->drawRgb565Span(region.x, y, sourceRow, region.width);
    }
  }
  glitchRegionCount_ = 0;
}

void CrtEffect::drawGlitchBurstFrame() {
  const uint8_t configuredMax =
      min(config_.glitchMaxRegions, MAX_GLITCH_REGIONS);
  glitchRegionCount_ = 1 + (esp_random() % configuredMax);

  for (uint8_t index = 0; index < glitchRegionCount_; ++index) {
    GlitchRegion& region = glitchRegions_[index];
    const int16_t widthRange =
        max<int16_t>(1, config_.glitchMaxWidth - config_.glitchMinWidth + 1);
    region.width = config_.glitchMinWidth + (esp_random() % widthRange);
    region.width = min<int16_t>(region.width, DisplayManager::WIDTH);
    region.x = esp_random() % (DisplayManager::WIDTH - region.width + 1);
    region.height = 2 + (esp_random() % max<uint8_t>(1, config_.glitchMaxHeight - 1));
    region.height = min<int16_t>(region.height, DisplayManager::HEIGHT);
    region.y = esp_random() % (DisplayManager::HEIGHT - region.height + 1);

    const int8_t magnitude = 2 + (esp_random() % max<uint8_t>(1, config_.glitchMaxShift - 1));
    const int8_t shift = (esp_random() & 1U) ? magnitude : -magnitude;
    for (int16_t row = 0; row < region.height; ++row) {
      const int16_t y = region.y + row;
      const uint16_t* sourceRow =
          effectSource() + static_cast<size_t>(y) * DisplayManager::WIDTH;
      for (int16_t offset = 0; offset < region.width; ++offset) {
        const int16_t screenX = region.x + offset;
        const int16_t sourceX =
            constrain(screenX - shift, 0, DisplayManager::WIDTH - 1);
        uint16_t pixel = sourceRow[sourceX];
        if ((esp_random() % 1000U) < 90U) {
          const uint8_t level = 8 + (esp_random() % 24U);
          pixel = static_cast<uint16_t>((level << 11) |
                                        (min<uint8_t>(63, level * 2) << 5) |
                                        level);
        }
        rowBuffer_[offset] = pixel;
      }
      display_->drawRgb565Span(region.x, y, rowBuffer_, region.width);
    }
  }
}

void CrtEffect::restoreStatic() {
  for (uint8_t index = 0; index < staticRegionCount_; ++index) {
    const GlitchRegion& region = staticRegions_[index];
    for (int16_t row = 0; row < region.height; ++row) {
      const int16_t y = region.y + row;
      const uint16_t* sourceRow =
          effectSource() + static_cast<size_t>(y) * DisplayManager::WIDTH + region.x;
      display_->drawRgb565Span(region.x, y, sourceRow, region.width);
    }
  }
  staticRegionCount_ = 0;
}

void CrtEffect::drawStaticFrame() {
  staticRegionCount_ = min(config_.staticRegionCount, MAX_STATIC_REGIONS);
  for (uint8_t index = 0; index < staticRegionCount_; ++index) {
    GlitchRegion& region = staticRegions_[index];
    region.width = 2 + (esp_random() % 13U);
    region.height = 1 + (esp_random() % 2U);
    region.x = esp_random() % (DisplayManager::WIDTH - region.width + 1);
    region.y = esp_random() % (DisplayManager::HEIGHT - region.height + 1);
    for (int16_t row = 0; row < region.height; ++row) {
      for (int16_t x = 0; x < region.width; ++x) {
        const uint8_t level = 10 + (esp_random() % 34U);
        const bool greenTint = (esp_random() & 3U) == 0U;
        const uint8_t red = greenTint ? level / 2 : level;
        const uint8_t green = min<uint8_t>(63, greenTint ? level * 2 : level);
        const uint8_t blue = greenTint ? level / 2 : level;
        rowBuffer_[x] = static_cast<uint16_t>(
            ((red & 0x1F) << 11) | ((green & 0x3F) << 5) | (blue & 0x1F));
      }
      display_->drawRgb565Span(region.x, region.y + row, rowBuffer_,
                               region.width);
    }
  }
}

void CrtEffect::restoreBlockGlitch() {
  if (!blockRegionActive_) return;
  for (int16_t row = 0; row < blockRegion_.height; ++row) {
    const int16_t y = blockRegion_.y + row;
    const uint16_t* sourceRow =
        effectSource() + static_cast<size_t>(y) * DisplayManager::WIDTH + blockRegion_.x;
    display_->drawRgb565Span(blockRegion_.x, y, sourceRow, blockRegion_.width);
  }
  blockRegionActive_ = false;
}

void CrtEffect::drawBlockGlitchFrame() {
  blockRegion_.width = config_.blockMinWidth +
      (esp_random() % (config_.blockMaxWidth - config_.blockMinWidth + 1));
  blockRegion_.height = config_.blockMinHeight +
      (esp_random() % (config_.blockMaxHeight - config_.blockMinHeight + 1));
  blockRegion_.x =
      esp_random() % (DisplayManager::WIDTH - blockRegion_.width + 1);
  blockRegion_.y =
      esp_random() % (DisplayManager::HEIGHT - blockRegion_.height + 1);

  const int8_t shiftX = 3 + (esp_random() % max<uint8_t>(1, config_.blockMaxShift - 2));
  const int8_t shiftY = 2 + (esp_random() % max<uint8_t>(1, config_.blockMaxShift - 1));
  const int8_t signedX = (esp_random() & 1U) ? shiftX : -shiftX;
  const int8_t signedY = (esp_random() & 1U) ? shiftY : -shiftY;
  for (int16_t row = 0; row < blockRegion_.height; ++row) {
    const int16_t sourceY =
        constrain(blockRegion_.y + row - signedY, 0, DisplayManager::HEIGHT - 1);
    const uint16_t* sourceRow =
        effectSource() + static_cast<size_t>(sourceY) * DisplayManager::WIDTH;
    for (int16_t offset = 0; offset < blockRegion_.width; ++offset) {
      const int16_t sourceX = constrain(blockRegion_.x + offset - signedX, 0,
                                        DisplayManager::WIDTH - 1);
      rowBuffer_[offset] = sourceRow[sourceX];
    }
    display_->drawRgb565Span(blockRegion_.x, blockRegion_.y + row, rowBuffer_,
                             blockRegion_.width);
  }
  blockRegionActive_ = true;
}

void CrtEffect::restoreSyncTear() {
  if (!syncTearActive_) return;
  for (int16_t row = 0; row < syncTearRegion_.height; ++row) {
    const int16_t y = syncTearRegion_.y + row;
    const uint16_t* sourceRow = effectSource() +
        static_cast<size_t>(y) * DisplayManager::WIDTH + syncTearRegion_.x;
    display_->drawRgb565Span(syncTearRegion_.x, y, sourceRow,
                             syncTearRegion_.width);
  }
  syncTearActive_ = false;
}

void CrtEffect::drawSyncTearFrame() {
  const int16_t minWidth = DisplayManager::WIDTH * 3 / 4;
  syncTearRegion_.width =
      minWidth + (esp_random() % (DisplayManager::WIDTH - minWidth + 1));
  syncTearRegion_.height = config_.syncTearMinHeight +
      (esp_random() %
       (config_.syncTearMaxHeight - config_.syncTearMinHeight + 1));
  syncTearRegion_.x =
      esp_random() % (DisplayManager::WIDTH - syncTearRegion_.width + 1);
  syncTearRegion_.y =
      esp_random() % (DisplayManager::HEIGHT - syncTearRegion_.height + 1);

  const int16_t shift = config_.syncTearMinShift +
      (esp_random() %
       (config_.syncTearMaxShift - config_.syncTearMinShift + 1));
  const int16_t signedShift = (esp_random() & 1U) ? shift : -shift;
  for (int16_t row = 0; row < syncTearRegion_.height; ++row) {
    const int16_t y = syncTearRegion_.y + row;
    const uint16_t* sourceRow =
        effectSource() + static_cast<size_t>(y) * DisplayManager::WIDTH;
    for (int16_t offset = 0; offset < syncTearRegion_.width; ++offset) {
      const int16_t screenX = syncTearRegion_.x + offset;
      int16_t sourceX = screenX - signedShift;
      while (sourceX < 0) sourceX += DisplayManager::WIDTH;
      while (sourceX >= DisplayManager::WIDTH) sourceX -= DisplayManager::WIDTH;
      rowBuffer_[offset] = sourceRow[sourceX];
    }
    display_->drawRgb565Span(syncTearRegion_.x, y, rowBuffer_,
                             syncTearRegion_.width);
  }
  syncTearActive_ = true;
}

uint16_t CrtEffect::alterPixel(uint16_t pixel, bool scanline) const {
  int red = (pixel >> 11) & 0x1F;
  int green = (pixel >> 5) & 0x3F;
  int blue = pixel & 0x1F;

  const int delta = scanline ? 0 : config_.brightnessSteps;
  red = min(31, red + delta);
  green = min(63, green + delta);
  blue = min(31, blue + delta);

  if ((esp_random() % 1000U) < config_.noisePerThousand) {
    const bool greenNoise = (esp_random() & 3U) == 0U;
    red = greenNoise ? 8 : 18;
    green = greenNoise ? 28 : 36;
    blue = greenNoise ? 8 : 18;
  }
  return static_cast<uint16_t>((red << 11) | (green << 5) | blue);
}
