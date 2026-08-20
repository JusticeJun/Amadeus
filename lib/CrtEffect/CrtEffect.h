#pragma once

#include <Arduino.h>

#include "DisplayManager.h"

enum class CrtMode : uint8_t { Ambient, Transition };

struct CrtEffectConfig {
  bool enabled = true;
  uint8_t rollingBandHeight = 16;
  uint8_t movementPixels = 2;
  uint8_t horizontalJitterPixels = 1;
  uint16_t noisePerThousand = 5;
  uint8_t brightnessSteps = 1;
  uint16_t refreshIntervalMs = 45;
  uint8_t staticRegionCount = 0;
  uint16_t glitchChancePerThousand = 35;
  uint8_t glitchMaxRegions = 4;
  uint8_t glitchMinWidth = 40;
  uint16_t glitchMaxWidth = 190;
  uint8_t glitchMaxHeight = 8;
  uint8_t glitchMaxShift = 9;
  uint16_t blockGlitchChancePerThousand = 0;
  uint8_t blockMinWidth = 22;
  uint8_t blockMaxWidth = 78;
  uint8_t blockMinHeight = 6;
  uint8_t blockMaxHeight = 28;
  uint8_t blockMaxShift = 12;
  uint8_t scanlineDimPercent = 94;
  uint8_t edgeDimPercent = 94;
  uint8_t edgeWidthPixels = 14;
  uint16_t syncTearChancePerThousand = 8;
  uint8_t syncTearMinHeight = 8;
  uint8_t syncTearMaxHeight = 22;
  uint8_t syncTearMinShift = 18;
  uint8_t syncTearMaxShift = 48;
  CrtMode mode = CrtMode::Ambient;
};

class CrtEffect {
 public:
  bool begin(DisplayManager& display, const uint16_t* source,
             const CrtEffectConfig& config);
  void update();
  void setEnabled(bool enabled);

 private:
  void restoreBand(int16_t startY);
  void drawAmbientBand(int16_t startY);
  void restoreGlitches();
  void drawGlitchBurstFrame();
  void restoreStatic();
  void drawStaticFrame();
  void restoreBlockGlitch();
  void drawBlockGlitchFrame();
  bool buildAmbientBase();
  const uint16_t* effectSource() const;
  void restoreSyncTear();
  void drawSyncTearFrame();
  uint16_t alterPixel(uint16_t pixel, bool scanline) const;

  struct GlitchRegion {
    int16_t x = 0;
    int16_t y = 0;
    int16_t width = 0;
    int16_t height = 0;
  };

  DisplayManager* display_ = nullptr;
  const uint16_t* source_ = nullptr;
  uint16_t* ambientPixels_ = nullptr;
  CrtEffectConfig config_{};
  uint16_t rowBuffer_[DisplayManager::WIDTH]{};
  int16_t bandY_ = -1;
  uint32_t lastUpdateMs_ = 0;
  static constexpr uint8_t MAX_GLITCH_REGIONS = 4;
  GlitchRegion glitchRegions_[MAX_GLITCH_REGIONS]{};
  uint8_t glitchRegionCount_ = 0;
  uint8_t glitchFramesRemaining_ = 0;
  static constexpr uint8_t MAX_STATIC_REGIONS = 8;
  GlitchRegion staticRegions_[MAX_STATIC_REGIONS]{};
  uint8_t staticRegionCount_ = 0;
  GlitchRegion blockRegion_{};
  bool blockRegionActive_ = false;
  uint8_t blockFramesRemaining_ = 0;
  GlitchRegion syncTearRegion_{};
  bool syncTearActive_ = false;
  uint8_t syncTearFramesRemaining_ = 0;
};
