#include "CrtEffect.h"

#include <esp_heap_caps.h>
#include <esp_system.h>

namespace {

constexpr uint16_t TRANSITION_DURATION_MS = 120;
constexpr uint16_t TRANSITION_SWAP_MS = 60;

constexpr uint8_t TRANSITION_TEAR_MIN_COUNT = 8;
constexpr uint8_t TRANSITION_TEAR_MAX_COUNT = 12;

constexpr int16_t TRANSITION_TEAR_MIN_HEIGHT = 25;
constexpr int16_t TRANSITION_TEAR_MAX_HEIGHT = 100;

constexpr int16_t TRANSITION_TEAR_MIN_SHIFT = 70;
constexpr int16_t TRANSITION_TEAR_MAX_SHIFT = 220;

}  // namespace

bool CrtEffect::begin(DisplayManager& display, const uint16_t* source,
                      const CrtEffectConfig& config) {
  display_ = &display;
  source_ = source;
  config_ = config;
  bandY_ = -1;
  lastUpdateMs_ = millis();

  if (source_ == nullptr || !buildAmbientBase()) return false;

  for (int16_t y = 0; y < DisplayManager::HEIGHT; ++y) {
    display_->drawRgb565Row(y, ambientPixels_ + static_cast<size_t>(y) * DisplayManager::WIDTH,
                            DisplayManager::WIDTH);
  }

  return true;
}

const uint16_t* CrtEffect::effectSource() const {
  return ambientPixels_ != nullptr ? ambientPixels_ : source_;
}

bool CrtEffect::buildAmbientBase() {
  constexpr size_t pixelCount = static_cast<size_t>(DisplayManager::WIDTH) * DisplayManager::HEIGHT;

  if (ambientPixels_ == nullptr) {
    ambientPixels_ = static_cast<uint16_t*>(
        heap_caps_malloc(pixelCount * sizeof(uint16_t), MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT));
  }

  if (ambientPixels_ == nullptr) return false;

  for (int16_t y = 0; y < DisplayManager::HEIGHT; ++y) {
    for (int16_t x = 0; x < DisplayManager::WIDTH; ++x) {
      const size_t offset = static_cast<size_t>(y) * DisplayManager::WIDTH + x;
      ambientPixels_[offset] = makeAmbientPixel(x, y, source_[offset]);
    }
  }

  return true;
}

uint16_t CrtEffect::makeAmbientPixel(int16_t x, int16_t y, uint16_t pixel) const {
  uint16_t scale = (y & 1) ? config_.scanlineDimPercent : 100;

  if (x < config_.edgeWidthPixels || x >= DisplayManager::WIDTH - config_.edgeWidthPixels) {
    scale = static_cast<uint16_t>(scale * config_.edgeDimPercent / 100U);
  }

  const uint16_t red = ((pixel >> 11) & 0x1F) * scale / 100U;

  const uint16_t green = ((pixel >> 5) & 0x3F) * scale / 100U;

  const uint16_t blue = (pixel & 0x1F) * scale / 100U;

  return static_cast<uint16_t>((red << 11) | (green << 5) | blue);
}

bool CrtEffect::setSourceFull(const uint16_t* source) {
  if (display_ == nullptr || source == nullptr) return false;

  restoreGlitches();
  restoreStatic();
  restoreBlockGlitch();
  restoreSyncTear();
  restoreTransitionTears();

  source_ = source;

  if (!buildAmbientBase()) return false;

  for (int16_t y = 0; y < DisplayManager::HEIGHT; ++y) {
    display_->drawRgb565Row(y, ambientPixels_ + static_cast<size_t>(y) * DisplayManager::WIDTH,
                            DisplayManager::WIDTH);
  }

  return true;
}

bool CrtEffect::setSourceRegions(const uint16_t* source, const CrtUpdateRegion* regions,
                                 size_t regionCount) {
  if (display_ == nullptr || ambientPixels_ == nullptr || source == nullptr || regions == nullptr) {
    return false;
  }

  restoreGlitches();
  restoreStatic();
  restoreBlockGlitch();
  restoreSyncTear();
  restoreTransitionTears();

  source_ = source;

  for (size_t index = 0; index < regionCount; ++index) {
    const CrtUpdateRegion& requested = regions[index];

    const int16_t startX = constrain(requested.x, 0, DisplayManager::WIDTH);

    const int16_t startY = constrain(requested.y, 0, DisplayManager::HEIGHT);

    const int16_t endX = constrain(requested.x + requested.width, 0, DisplayManager::WIDTH);

    const int16_t endY = constrain(requested.y + requested.height, 0, DisplayManager::HEIGHT);

    if (startX >= endX || startY >= endY) continue;

    for (int16_t y = startY; y < endY; ++y) {
      const size_t rowOffset = static_cast<size_t>(y) * DisplayManager::WIDTH;

      for (int16_t x = startX; x < endX; ++x) {
        const size_t offset = rowOffset + x;
        ambientPixels_[offset] = makeAmbientPixel(x, y, source_[offset]);
      }

      display_->drawRgb565Span(startX, y, ambientPixels_ + rowOffset + startX, endX - startX);
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
    restoreTransitionTears();

    transitionActive_ = false;
    transitionSourceSwapped_ = false;
    pendingTransitionSource_ = nullptr;
  }

  config_.enabled = enabled;

  if (!enabled) {
    bandY_ = -1;
  }
}

void CrtEffect::triggerTransition() {
  if (!config_.enabled) return;

  glitchFramesRemaining_ = max<uint8_t>(glitchFramesRemaining_, 3);
}

bool CrtEffect::transitionToSource(const uint16_t* source) {
  if (display_ == nullptr || source == nullptr) {
    return false;
  }

  if (!config_.enabled) {
    return setSourceFull(source);
  }

  restoreGlitches();
  restoreStatic();
  restoreBlockGlitch();
  restoreSyncTear();
  restoreTransitionTears();

  glitchRegionCount_ = 0;
  glitchFramesRemaining_ = 0;
  blockFramesRemaining_ = 0;
  syncTearFramesRemaining_ = 0;

  transitionActive_ = true;
  transitionSourceSwapped_ = false;
  transitionTearDrawn_ = false;
  transitionStartMs_ = millis();
  pendingTransitionSource_ = source;

  return true;
}

void CrtEffect::update() {
  if (!config_.enabled || display_ == nullptr || source_ == nullptr ||
      config_.mode != CrtMode::Ambient) {
    return;
  }

  const uint32_t now = millis();

  if (now - lastUpdateMs_ < config_.refreshIntervalMs) {
    return;
  }

  lastUpdateMs_ = now;

  if (transitionActive_) {
    const uint32_t elapsed = now - transitionStartMs_;

    if (!transitionTearDrawn_) {
      restoreTransitionTears();
      drawTransitionSyncTearFrame();
      transitionTearDrawn_ = true;
    }

    if (!transitionSourceSwapped_ && elapsed >= TRANSITION_SWAP_MS) {
      restoreTransitionTears();

      source_ = pendingTransitionSource_;

      if (!buildAmbientBase()) {
        transitionActive_ = false;
        transitionSourceSwapped_ = false;
        transitionTearDrawn_ = false;
        pendingTransitionSource_ = nullptr;
        return;
      }

      for (int16_t y = 0; y < DisplayManager::HEIGHT; ++y) {
        display_->drawRgb565Row(y, ambientPixels_ + static_cast<size_t>(y) * DisplayManager::WIDTH,
                                DisplayManager::WIDTH);
      }

      transitionSourceSwapped_ = true;
    }

    if (elapsed >= TRANSITION_DURATION_MS) {
      restoreTransitionTears();

      if (!transitionSourceSwapped_ && pendingTransitionSource_ != nullptr) {
        source_ = pendingTransitionSource_;

        if (buildAmbientBase()) {
          for (int16_t y = 0; y < DisplayManager::HEIGHT; ++y) {
            display_->drawRgb565Row(y,
                                    ambientPixels_ + static_cast<size_t>(y) * DisplayManager::WIDTH,
                                    DisplayManager::WIDTH);
          }
        }
      }

      transitionActive_ = false;
      transitionSourceSwapped_ = false;
      transitionTearDrawn_ = false;
      pendingTransitionSource_ = nullptr;

      glitchFramesRemaining_ = 0;
      blockFramesRemaining_ = 0;
      syncTearFramesRemaining_ = 0;
    }

    return;
  }

  restoreGlitches();
  restoreStatic();
  restoreBlockGlitch();
  restoreSyncTear();

  drawStaticFrame();

  if (glitchFramesRemaining_ == 0 && (esp_random() % 1000U) < config_.glitchChancePerThousand) {
    glitchFramesRemaining_ = 2 + (esp_random() % 3U);
  }

  if (glitchFramesRemaining_ > 0) {
    drawGlitchBurstFrame();
    --glitchFramesRemaining_;
  }

  if (blockFramesRemaining_ == 0 && (esp_random() % 1000U) < config_.blockGlitchChancePerThousand) {
    blockFramesRemaining_ = 2 + (esp_random() % 2U);
  }

  if (blockFramesRemaining_ > 0) {
    drawBlockGlitchFrame();
    --blockFramesRemaining_;
  }

  if (syncTearFramesRemaining_ == 0 && (esp_random() % 1000U) < config_.syncTearChancePerThousand) {
    syncTearFramesRemaining_ = 1 + (esp_random() % 2U);
  }

  if (syncTearFramesRemaining_ > 0) {
    drawSyncTearFrame();
    --syncTearFramesRemaining_;
  }
}

void CrtEffect::restoreTransitionTears() {
  if (transitionTearCount_ == 0) return;

  for (uint8_t index = 0; index < transitionTearCount_; ++index) {
    const GlitchRegion& region = transitionTearRegions_[index];

    for (int16_t row = 0; row < region.height; ++row) {
      const int16_t y = region.y + row;

      const uint16_t* sourceRow =
          effectSource() + static_cast<size_t>(y) * DisplayManager::WIDTH + region.x;

      display_->drawRgb565Span(region.x, y, sourceRow, region.width);
    }
  }

  transitionTearCount_ = 0;
}

void CrtEffect::drawTransitionSyncTearFrame() {
  restoreTransitionTears();

  transitionTearCount_ =
      TRANSITION_TEAR_MIN_COUNT +
      (esp_random() % (TRANSITION_TEAR_MAX_COUNT - TRANSITION_TEAR_MIN_COUNT + 1));

  transitionTearCount_ = min<uint8_t>(transitionTearCount_, MAX_TRANSITION_TEARS);

  const int16_t sliceHeight = DisplayManager::HEIGHT / transitionTearCount_;

  for (uint8_t index = 0; index < transitionTearCount_; ++index) {
    GlitchRegion& region = transitionTearRegions_[index];

    const int16_t minWidth = DisplayManager::WIDTH * 9 / 10;

    region.width = minWidth + (esp_random() % (DisplayManager::WIDTH - minWidth + 1));

    region.height = TRANSITION_TEAR_MIN_HEIGHT +
                    (esp_random() % (TRANSITION_TEAR_MAX_HEIGHT - TRANSITION_TEAR_MIN_HEIGHT + 1));

    region.x = esp_random() % (DisplayManager::WIDTH - region.width + 1);

    const int16_t bandTop = index * sliceHeight;

    const int16_t bandBottom =
        (index == transitionTearCount_ - 1) ? DisplayManager::HEIGHT : (index + 1) * sliceHeight;

    const int16_t availableHeight = max<int16_t>(1, bandBottom - bandTop - region.height + 1);

    region.y = bandTop + (esp_random() % availableHeight);

    region.y = constrain(region.y, 0, DisplayManager::HEIGHT - region.height);

    const int16_t shift =
        TRANSITION_TEAR_MIN_SHIFT +
        (esp_random() % (TRANSITION_TEAR_MAX_SHIFT - TRANSITION_TEAR_MIN_SHIFT + 1));

    const int16_t signedShift = (esp_random() & 1U) ? shift : -shift;

    for (int16_t row = 0; row < region.height; ++row) {
      const int16_t y = region.y + row;

      const uint16_t* sourceRow = effectSource() + static_cast<size_t>(y) * DisplayManager::WIDTH;

      for (int16_t offset = 0; offset < region.width; ++offset) {
        const int16_t screenX = region.x + offset;

        int16_t sourceX = screenX - signedShift;

        while (sourceX < 0) {
          sourceX += DisplayManager::WIDTH;
        }

        while (sourceX >= DisplayManager::WIDTH) {
          sourceX -= DisplayManager::WIDTH;
        }

        rowBuffer_[offset] = sourceRow[sourceX];
      }

      display_->drawRgb565Span(region.x, y, rowBuffer_, region.width);
    }
  }
}

void CrtEffect::restoreBand(int16_t startY) {
  for (uint8_t row = 0; row < config_.rollingBandHeight; ++row) {
    const int16_t y = (startY + row) % DisplayManager::HEIGHT;

    display_->drawRgb565Row(y, effectSource() + static_cast<size_t>(y) * DisplayManager::WIDTH,
                            DisplayManager::WIDTH);
  }
}

void CrtEffect::drawAmbientBand(int16_t startY) {
  int8_t shift = 0;

  if (config_.horizontalJitterPixels > 0 && (esp_random() % 24U) == 0U) {
    shift = (esp_random() & 1U) ? config_.horizontalJitterPixels : -config_.horizontalJitterPixels;
  }

  for (uint8_t row = 0; row < config_.rollingBandHeight; ++row) {
    const int16_t y = (startY + row) % DisplayManager::HEIGHT;

    const uint16_t* sourceRow = effectSource() + static_cast<size_t>(y) * DisplayManager::WIDTH;

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
  const uint8_t configuredMax = min(config_.glitchMaxRegions, MAX_GLITCH_REGIONS);

  glitchRegionCount_ = 1 + (esp_random() % configuredMax);

  for (uint8_t index = 0; index < glitchRegionCount_; ++index) {
    GlitchRegion& region = glitchRegions_[index];

    const int16_t widthRange = max<int16_t>(1, config_.glitchMaxWidth - config_.glitchMinWidth + 1);

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

      const uint16_t* sourceRow = effectSource() + static_cast<size_t>(y) * DisplayManager::WIDTH;

      for (int16_t offset = 0; offset < region.width; ++offset) {
        const int16_t screenX = region.x + offset;

        const int16_t sourceX = constrain(screenX - shift, 0, DisplayManager::WIDTH - 1);

        uint16_t pixel = sourceRow[sourceX];

        if ((esp_random() % 1000U) < 90U) {
          const uint8_t level = 8 + (esp_random() % 24U);

          pixel = static_cast<uint16_t>((level << 11) | (min<uint8_t>(63, level * 2) << 5) | level);
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

        rowBuffer_[x] =
            static_cast<uint16_t>(((red & 0x1F) << 11) | ((green & 0x3F) << 5) | (blue & 0x1F));
      }

      display_->drawRgb565Span(region.x, region.y + row, rowBuffer_, region.width);
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
  blockRegion_.width =
      config_.blockMinWidth + (esp_random() % (config_.blockMaxWidth - config_.blockMinWidth + 1));

  blockRegion_.height = config_.blockMinHeight +
                        (esp_random() % (config_.blockMaxHeight - config_.blockMinHeight + 1));

  blockRegion_.x = esp_random() % (DisplayManager::WIDTH - blockRegion_.width + 1);

  blockRegion_.y = esp_random() % (DisplayManager::HEIGHT - blockRegion_.height + 1);

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
      const int16_t sourceX =
          constrain(blockRegion_.x + offset - signedX, 0, DisplayManager::WIDTH - 1);

      rowBuffer_[offset] = sourceRow[sourceX];
    }

    display_->drawRgb565Span(blockRegion_.x, blockRegion_.y + row, rowBuffer_, blockRegion_.width);
  }

  blockRegionActive_ = true;
}

void CrtEffect::restoreSyncTear() {
  if (!syncTearActive_) return;

  for (int16_t row = 0; row < syncTearRegion_.height; ++row) {
    const int16_t y = syncTearRegion_.y + row;

    const uint16_t* sourceRow =
        effectSource() + static_cast<size_t>(y) * DisplayManager::WIDTH + syncTearRegion_.x;

    display_->drawRgb565Span(syncTearRegion_.x, y, sourceRow, syncTearRegion_.width);
  }

  syncTearActive_ = false;
}

void CrtEffect::drawSyncTearFrame() {
  const int16_t minWidth = DisplayManager::WIDTH * 3 / 4;

  syncTearRegion_.width = minWidth + (esp_random() % (DisplayManager::WIDTH - minWidth + 1));

  syncTearRegion_.height =
      config_.syncTearMinHeight +
      (esp_random() % (config_.syncTearMaxHeight - config_.syncTearMinHeight + 1));

  syncTearRegion_.x = esp_random() % (DisplayManager::WIDTH - syncTearRegion_.width + 1);

  syncTearRegion_.y = esp_random() % (DisplayManager::HEIGHT - syncTearRegion_.height + 1);

  const int16_t shift = config_.syncTearMinShift +
                        (esp_random() % (config_.syncTearMaxShift - config_.syncTearMinShift + 1));

  const int16_t signedShift = (esp_random() & 1U) ? shift : -shift;

  for (int16_t row = 0; row < syncTearRegion_.height; ++row) {
    const int16_t y = syncTearRegion_.y + row;

    const uint16_t* sourceRow = effectSource() + static_cast<size_t>(y) * DisplayManager::WIDTH;

    for (int16_t offset = 0; offset < syncTearRegion_.width; ++offset) {
      const int16_t screenX = syncTearRegion_.x + offset;

      int16_t sourceX = screenX - signedShift;

      while (sourceX < 0) {
        sourceX += DisplayManager::WIDTH;
      }

      while (sourceX >= DisplayManager::WIDTH) {
        sourceX -= DisplayManager::WIDTH;
      }

      rowBuffer_[offset] = sourceRow[sourceX];
    }

    display_->drawRgb565Span(syncTearRegion_.x, y, rowBuffer_, syncTearRegion_.width);
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