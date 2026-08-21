#include <Arduino.h>
#include <esp_chip_info.h>
#include <esp_flash.h>
#include <esp_heap_caps.h>

#include "AssetManager.h"
#include "CharacterController.h"
#include "CrtEffect.h"
#include "DisplayManager.h"
#include "ProjectConfig.h"

namespace {
DisplayManager display;
AssetManager assets;
CrtEffect crtEffect;
CharacterController character;
CharacterAssets characterAssets;

void printBytes(const char* label, uint64_t bytes) {
  Serial.printf("%-24s %llu bytes (%.2f MiB)\n", label,
                static_cast<unsigned long long>(bytes),
                static_cast<double>(bytes) / (1024.0 * 1024.0));
}

void printDiagnostics() {
  esp_chip_info_t chip{};
  esp_chip_info(&chip);

  uint32_t flashBytes = 0;
  const esp_err_t flashResult = esp_flash_get_size(nullptr, &flashBytes);

  Serial.println();
  Serial.println("=== Amadeus ESP32-S3 N16R8 diagnostic ===");
  Serial.printf("Chip model:              %s\n", ESP.getChipModel());
  Serial.printf("Chip revision:           %u\n", chip.revision);
  Serial.printf("CPU cores:               %u\n", chip.cores);
  Serial.printf("CPU frequency:           %u MHz\n", ESP.getCpuFreqMHz());
  if (flashResult == ESP_OK) {
    printBytes("Flash size:", flashBytes);
  } else {
    Serial.printf("Flash size read error:   %s\n", esp_err_to_name(flashResult));
  }

  const bool hasPsram = psramFound();
  Serial.printf("PSRAM detected:          %s\n", hasPsram ? "YES" : "NO");
  printBytes("PSRAM size:", ESP.getPsramSize());
  printBytes("Free PSRAM:", ESP.getFreePsram());
  printBytes("Internal heap free:",
             heap_caps_get_free_size(MALLOC_CAP_INTERNAL));
  printBytes("Internal heap largest:",
             heap_caps_get_largest_free_block(MALLOC_CAP_INTERNAL));
  Serial.println("=== diagnostic complete ===");
}

}  // namespace

void setup() {
  Serial.begin(ProjectConfig::SERIAL_BAUD);
  delay(ProjectConfig::STARTUP_SERIAL_WAIT_MS);
  printDiagnostics();
  Serial.println("Initializing ST7796 at 10 MHz SPI...");
  display.begin();
  Serial.printf("Display geometry:        %d x %d\n", display.width(),
                display.height());
  Serial.println("Mounting LittleFS...");
  if (!assets.begin()) {
    Serial.println("LittleFS mount failed. Upload the filesystem image separately.");
    return;
  }
  Serial.println("Drawing /neutral.rgb565...");
  if (assets.drawRgb565(display, "/neutral.rgb565")) {
    Serial.println("Character image draw complete.");
  } else {
    Serial.println("Character image draw failed.");
    return;
  }
  Serial.println("Loading character states into PSRAM...");
  characterAssets.neutral = assets.imagePixels();
  characterAssets.listening = assets.loadRgb565Asset("/listening.rgb565");
  characterAssets.answering1 = assets.loadRgb565Asset("/answering1.rgb565");
  characterAssets.answering2 = assets.loadRgb565Asset("/answering2.rgb565");
  characterAssets.answering3 = assets.loadRgb565Asset("/answering3.rgb565");
  characterAssets.happyFrameA = assets.loadRgb565Asset("/happy_hearts_a.rgb565");
  characterAssets.happyFrameB = assets.loadRgb565Asset("/happy_hearts_b.rgb565");
  characterAssets.shy = assets.loadRgb565Asset("/shy.rgb565");
  characterAssets.angry = assets.loadRgb565Asset("/angry.rgb565");
  characterAssets.sad = assets.loadRgb565Asset("/sad.rgb565");
  characterAssets.wondering = assets.loadRgb565Asset("/wondering.rgb565");
  if (characterAssets.listening == nullptr ||
      characterAssets.answering1 == nullptr ||
      characterAssets.answering2 == nullptr ||
      characterAssets.answering3 == nullptr ||
      characterAssets.happyFrameA == nullptr || characterAssets.happyFrameB == nullptr ||
      characterAssets.shy == nullptr || characterAssets.angry == nullptr ||
      characterAssets.sad == nullptr || characterAssets.wondering == nullptr) {
    Serial.println("Character state asset load failed.");
    return;
  }
  Serial.println("Character state assets loaded.");

  CrtEffectConfig crtConfig;
  crtConfig.enabled = ProjectConfig::Crt::ENABLED;
  crtConfig.rollingBandHeight = 0;
  crtConfig.refreshIntervalMs = ProjectConfig::Crt::REFRESH_INTERVAL_MS;
  crtConfig.glitchChancePerThousand =
      ProjectConfig::Crt::GLITCH_CHANCE_PER_THOUSAND;
  crtConfig.glitchMaxRegions = ProjectConfig::Crt::GLITCH_MAX_REGIONS;
  crtConfig.glitchMinWidth = ProjectConfig::Crt::GLITCH_MIN_WIDTH;
  crtConfig.glitchMaxWidth = ProjectConfig::Crt::GLITCH_MAX_WIDTH;
  crtConfig.glitchMaxHeight = ProjectConfig::Crt::GLITCH_MAX_HEIGHT;
  crtConfig.glitchMaxShift = ProjectConfig::Crt::GLITCH_MAX_SHIFT;
  crtConfig.staticRegionCount = 0;
  crtConfig.blockGlitchChancePerThousand = 0;
  crtConfig.scanlineDimPercent = ProjectConfig::Crt::SCANLINE_DIM_PERCENT;
  crtConfig.edgeDimPercent = ProjectConfig::Crt::EDGE_DIM_PERCENT;
  crtConfig.edgeWidthPixels = ProjectConfig::Crt::EDGE_WIDTH_PIXELS;
  crtConfig.syncTearChancePerThousand =
      ProjectConfig::Crt::SYNC_TEAR_CHANCE_PER_THOUSAND;
  crtConfig.syncTearMinHeight = ProjectConfig::Crt::SYNC_TEAR_MIN_HEIGHT;
  crtConfig.syncTearMaxHeight = ProjectConfig::Crt::SYNC_TEAR_MAX_HEIGHT;
  crtConfig.syncTearMinShift = ProjectConfig::Crt::SYNC_TEAR_MIN_SHIFT;
  crtConfig.syncTearMaxShift = ProjectConfig::Crt::SYNC_TEAR_MAX_SHIFT;
  if (crtEffect.begin(display, assets.imagePixels(), crtConfig)) {
    Serial.println("Ambient CRT effect started.");
    character.begin(crtEffect, characterAssets,
                    ProjectConfig::HappyHearts::FRAME_INTERVAL_MS);
    Serial.println("PC bridge protocol ready.");
  } else {
    Serial.println("Ambient CRT effect failed to start.");
  }
}

void loop() {
  character.update(Serial);
  crtEffect.update();
  delay(1);
}
