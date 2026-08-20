#include "CharacterController.h"

#include <cstring>

void CharacterController::begin(CrtEffect& crtEffect) {
  crtEffect_ = &crtEffect;
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
  strncpy(emotion_, requestedEmotion, sizeof(emotion_) - 1);
  emotion_[sizeof(emotion_) - 1] = '\0';
  Serial.printf("[bridge] Emotion: %s (neutral asset fallback)\n", emotion_);
  if (crtEffect_ != nullptr) crtEffect_->triggerTransition();
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
      "neutral", "happy", "shy", "pout", "surprised",
      "thinking", "sleep", "listening",
  };
  for (const char* supported : SUPPORTED) {
    if (strcmp(value, supported) == 0) return true;
  }
  return false;
}

