#pragma once

#include <Arduino.h>

#include "CrtEffect.h"

class CharacterController {
 public:
  void begin(CrtEffect& crtEffect);
  void update(Stream& serial);
  const char* emotion() const;

 private:
  static constexpr size_t MAX_LINE_BYTES = 160;
  void handleLine();
  bool extractString(const char* key, char* output, size_t outputSize) const;
  bool isSupportedEmotion(const char* value) const;

  CrtEffect* crtEffect_ = nullptr;
  char line_[MAX_LINE_BYTES + 1]{};
  size_t lineLength_ = 0;
  bool discardingLine_ = false;
  char emotion_[12] = "neutral";
};

