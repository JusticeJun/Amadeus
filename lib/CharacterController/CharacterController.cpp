#include "CharacterController.h"

#include <cstring>
#include <esp_random.h>

namespace {
constexpr int16_t DIRTY_SIZE = 48;
constexpr int16_t HALF_DIRTY = DIRTY_SIZE / 2;
constexpr CrtUpdateRegion HAPPY_DIRTY_REGIONS[] = {
    {61 - HALF_DIRTY, 42 - HALF_DIRTY, DIRTY_SIZE, DIRTY_SIZE},
    {39 - HALF_DIRTY, 233 - HALF_DIRTY, DIRTY_SIZE, DIRTY_SIZE},
    {285 - HALF_DIRTY, 151 - HALF_DIRTY, DIRTY_SIZE, DIRTY_SIZE},
    {259 - HALF_DIRTY, 53 - HALF_DIRTY, DIRTY_SIZE, DIRTY_SIZE},
    {48 - HALF_DIRTY, 150 - HALF_DIRTY, DIRTY_SIZE, DIRTY_SIZE},
    {273 - HALF_DIRTY, 244 - HALF_DIRTY, DIRTY_SIZE, DIRTY_SIZE},
};
}  // namespace

void CharacterController::begin(CrtEffect& crtEffect, const CharacterAssets& assets,
                                uint16_t happyFrameIntervalMs) {
  crtEffect_ = &crtEffect;
  assets_ = assets;
  happyFrameIntervalMs_ = happyFrameIntervalMs;
}

void CharacterController::update(Stream& serial) {
  while (serial.available() > 0) {
    const char incoming = static_cast<char>(serial.read());
    if (incoming == '\r') continue;
    if (incoming == '\n') {
      if (!discardingLine_ && lineLength_ > 0) {
        line_[lineLength_] = '\0';
        handleLine();
      }
      lineLength_ = 0;
      discardingLine_ = false;
      continue;
    }
    if (discardingLine_) continue;
    if (lineLength_ >= MAX_LINE_BYTES) {
      lineLength_ = 0;
      discardingLine_ = true;
      Serial.println("[bridge] Oversized command ignored.");
      continue;
    }
    line_[lineLength_++] = incoming;
  }
  updateHappyAnimation();
}

const char* CharacterController::emotion() const {
  return emotion_;
}

void CharacterController::handleLine() {
  char type[12]{};
  char requestedEmotion[12]{};
  if (!extractString("type", type, sizeof(type)) || strcmp(type, "state") != 0 ||
      !extractString("emotion", requestedEmotion, sizeof(requestedEmotion))) {
    Serial.println("[bridge] Invalid command ignored.");
    return;
  }
  if (!isSupportedEmotion(requestedEmotion)) {
    strcpy(requestedEmotion, "neutral");
  }
  if (strcmp(emotion_, requestedEmotion) == 0) return;
  const bool becomesHappy = strcmp(requestedEmotion, "happy") == 0;
  strncpy(emotion_, requestedEmotion, sizeof(emotion_) - 1);
  emotion_[sizeof(emotion_) - 1] = '\0';
  Serial.printf("[bridge] Emotion: %s\n", emotion_);

  if (becomesHappy && assets_.happyFrameA != nullptr &&
      assets_.happyFrameB != nullptr) {
    happyFrameAActive_ = true;
    lastHappyFrameMs_ = millis();
    showHappyFrame(true, true);
    return;
  }

  const uint16_t* pixels = pixelsForEmotion(requestedEmotion);
  if (crtEffect_ != nullptr && pixels != nullptr) {
    crtEffect_->transitionToSource(pixels);
  }
}

const uint16_t* CharacterController::pixelsForEmotion(const char* value) const {
  if (strcmp(value, "neutral") == 0 || strcmp(value, "sleep") == 0) {
    return assets_.neutral;
  }
  if (strcmp(value, "listening") == 0) return assets_.listening;
  if (strcmp(value, "answering") == 0) {
    const uint16_t* variants[] = {
        assets_.answering1, assets_.answering2, assets_.answering3};
    return variants[esp_random() % 3];
  }
  if (strcmp(value, "shy") == 0) return assets_.shy;
  if (strcmp(value, "angry") == 0) return assets_.angry;
  if (strcmp(value, "sad") == 0) return assets_.sad;
  if (strcmp(value, "wondering") == 0) return assets_.wondering;
  return assets_.neutral;
}

void CharacterController::updateHappyAnimation() {
  if (strcmp(emotion_, "happy") != 0 || crtEffect_ == nullptr) return;
  const uint32_t now = millis();
  if (now - lastHappyFrameMs_ < happyFrameIntervalMs_) {
    return;
  }
  lastHappyFrameMs_ = now;
  happyFrameAActive_ = !happyFrameAActive_;
  showHappyFrame(happyFrameAActive_, false);
}

void CharacterController::showHappyFrame(bool frameA, bool fullRedraw) {
  const uint16_t* pixels = frameA ? assets_.happyFrameA : assets_.happyFrameB;
  if (crtEffect_ == nullptr || pixels == nullptr) return;
  if (fullRedraw) {
    crtEffect_->transitionToSource(pixels);
  } else {
    crtEffect_->setSourceRegions(
        pixels, HAPPY_DIRTY_REGIONS,
        sizeof(HAPPY_DIRTY_REGIONS) / sizeof(HAPPY_DIRTY_REGIONS[0]));
  }
}

bool CharacterController::extractString(const char* key, char* output,
                                        size_t outputSize) const {
  char needle[24]{};
  snprintf(needle, sizeof(needle), "\"%s\"", key);
  const char* position = strstr(line_, needle);
  if (position == nullptr) return false;
  position += strlen(needle);
  while (*position == ' ' || *position == '\t') ++position;
  if (*position++ != ':') return false;
  while (*position == ' ' || *position == '\t') ++position;
  if (*position++ != '\"') return false;
  size_t length = 0;
  while (*position != '\0' && *position != '\"') {
    if (*position == '\\' || length + 1 >= outputSize) return false;
    output[length++] = *position++;
  }
  if (*position != '\"') return false;
  output[length] = '\0';
  return true;
}

bool CharacterController::isSupportedEmotion(const char* value) const {
  static constexpr const char* SUPPORTED[] = {
      "neutral", "listening", "answering", "happy", "angry",
      "shy", "sad", "wondering", "sleep",
  };
  for (const char* supported : SUPPORTED) {
    if (strcmp(value, supported) == 0) return true;
  }
  return false;
}

