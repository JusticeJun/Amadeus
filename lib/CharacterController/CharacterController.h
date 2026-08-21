#pragma once

#include <Arduino.h>

#include "CrtEffect.h"

struct CharacterAssets {
  const uint16_t* neutral = nullptr;
  const uint16_t* listening = nullptr;
  const uint16_t* answering1 = nullptr;
  const uint16_t* answering2 = nullptr;
  const uint16_t* answering3 = nullptr;
  const uint16_t* happyFrameA = nullptr;
  const uint16_t* happyFrameB = nullptr;
  const uint16_t* shy = nullptr;
  const uint16_t* angry = nullptr;
  const uint16_t* sad = nullptr;
  const uint16_t* wondering = nullptr;
};

class CharacterController {
 public:
  void begin(CrtEffect& crtEffect, const CharacterAssets& assets,
             uint16_t happyFrameIntervalMs);
  void update(Stream& serial);
  const char* emotion() const;

 private:
  static constexpr size_t MAX_LINE_BYTES = 160;
  void handleLine();
  bool extractString(const char* key, char* output, size_t outputSize) const;
  bool isSupportedEmotion(const char* value) const;
  const uint16_t* pixelsForEmotion(const char* value) const;
  void updateHappyAnimation();
  void showHappyFrame(bool frameA, bool fullRedraw);

  CrtEffect* crtEffect_ = nullptr;
  CharacterAssets assets_{};
  bool happyFrameAActive_ = true;
  uint32_t lastHappyFrameMs_ = 0;
  uint16_t happyFrameIntervalMs_ = 800;
  char line_[MAX_LINE_BYTES + 1]{};
  size_t lineLength_ = 0;
  bool discardingLine_ = false;
  char emotion_[12] = "neutral";
};

