#include <Arduino.h>
#include <driver/i2s.h>
#include <esp_err.h>
#include <esp_heap_caps.h>

#include <algorithm>
#include <cstdint>
#include <limits>

#include "pins.h"

namespace {
constexpr i2s_port_t kI2sPort = I2S_NUM_0;
constexpr uint32_t kSerialBaud = 921600;
constexpr uint32_t kSampleRate = 16000;
constexpr uint8_t kPcmBits = 16;
constexpr uint32_t kMinCaptureSeconds = 1;
constexpr uint32_t kMaxCaptureSeconds = 30;
constexpr size_t kReadSamples = 256;

QueueHandle_t i2sEvents = nullptr;
char commandBuffer[48]{};
size_t commandLength = 0;

struct CaptureStats {
  uint32_t readErrors = 0;
  uint32_t overruns = 0;
  uint32_t nonzero = 0;
  int16_t minimum = std::numeric_limits<int16_t>::max();
  int16_t maximum = std::numeric_limits<int16_t>::min();
  uint32_t clipped = 0;
};

void printError(const char* operation, esp_err_t result) {
  Serial.printf("[voice] %s failed: %s (%d)\n", operation,
                esp_err_to_name(result), static_cast<int>(result));
}

void drainI2sEvents(CaptureStats& stats) {
  if (i2sEvents == nullptr) {
    return;
  }
  i2s_event_t event{};
  while (xQueueReceive(i2sEvents, &event, 0) == pdTRUE) {
    if (event.type == I2S_EVENT_RX_Q_OVF) {
      ++stats.overruns;
    } else if (event.type == I2S_EVENT_DMA_ERROR) {
      ++stats.readErrors;
    }
  }
}

constexpr int16_t pcm16FromInmp441(int32_t slotSample) {
  // INMP441 produces signed 24-bit two's-complement data in bits 31..8 of
  // a 32-bit Philips-I2S slot. Dropping the least-significant eight valid
  // bits yields conventional signed PCM16 without adding gain or clipping.
  return static_cast<int16_t>((slotSample >> 8) >> 8);
}

static_assert(pcm16FromInmp441(0x7FFF0000) == 32767);
static_assert(pcm16FromInmp441(static_cast<int32_t>(0x80000000U)) == -32768);

bool installI2s() {
  const i2s_config_t config = {
      .mode = static_cast<i2s_mode_t>(I2S_MODE_MASTER | I2S_MODE_RX),
      .sample_rate = kSampleRate,
      .bits_per_sample = I2S_BITS_PER_SAMPLE_32BIT,
      .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,
      .communication_format = I2S_COMM_FORMAT_STAND_I2S,
      .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
      .dma_buf_count = 8,
      .dma_buf_len = kReadSamples,
      .use_apll = false,
      .tx_desc_auto_clear = false,
      .fixed_mclk = 0,
      .mclk_multiple = I2S_MCLK_MULTIPLE_DEFAULT,
      .bits_per_chan = I2S_BITS_PER_CHAN_32BIT,
  };
  const i2s_pin_config_t pins = {
      .mck_io_num = I2S_PIN_NO_CHANGE,
      .bck_io_num = Pins::MIC_BCLK,
      .ws_io_num = Pins::MIC_LRCL,
      .data_out_num = I2S_PIN_NO_CHANGE,
      .data_in_num = Pins::MIC_DATA,
  };

  esp_err_t result = i2s_driver_install(kI2sPort, &config, 16, &i2sEvents);
  if (result != ESP_OK) {
    printError("i2s_driver_install", result);
    return false;
  }
  result = i2s_set_pin(kI2sPort, &pins);
  if (result != ESP_OK) {
    printError("i2s_set_pin", result);
    i2s_driver_uninstall(kI2sPort);
    return false;
  }
  return true;
}

void stopAndUninstallI2s(CaptureStats& stats) {
  esp_err_t result = i2s_stop(kI2sPort);
  if (result != ESP_OK) {
    ++stats.readErrors;
    printError("i2s_stop", result);
  }
  drainI2sEvents(stats);
  result = i2s_driver_uninstall(kI2sPort);
  if (result != ESP_OK) {
    ++stats.readErrors;
    printError("i2s_driver_uninstall", result);
  }
  i2sEvents = nullptr;
}

void capture(uint32_t seconds) {
  const size_t requestedSamples = static_cast<size_t>(kSampleRate) * seconds;
  auto* pcm = static_cast<int16_t*>(heap_caps_malloc(
      requestedSamples * sizeof(int16_t), MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT));
  if (pcm == nullptr) {
    Serial.printf("AUDIO_ERROR allocation_failed samples=%u bytes=%u\n",
                  static_cast<unsigned>(requestedSamples),
                  static_cast<unsigned>(requestedSamples * sizeof(int16_t)));
    return;
  }

  Serial.printf("[voice] capture_start seconds=%u requested_samples=%u\n",
                static_cast<unsigned>(seconds),
                static_cast<unsigned>(requestedSamples));
  CaptureStats stats;
  if (!installI2s()) {
    heap_caps_free(pcm);
    Serial.println("AUDIO_ERROR capture_start_failed");
    return;
  }

  size_t capturedSamples = 0;
  int32_t raw[kReadSamples]{};
  while (capturedSamples < requestedSamples) {
    const size_t remaining = requestedSamples - capturedSamples;
    const size_t requestSamples = std::min(remaining, kReadSamples);
    size_t bytesRead = 0;
    const esp_err_t result = i2s_read(kI2sPort, raw,
                                     requestSamples * sizeof(int32_t),
                                     &bytesRead, pdMS_TO_TICKS(1000));
    const size_t requestedBytes = requestSamples * sizeof(int32_t);
    if (result != ESP_OK || bytesRead != requestedBytes ||
        bytesRead % sizeof(int32_t) != 0) {
      ++stats.readErrors;
      Serial.printf(
          "[voice] i2s_read failed: %s (%d) requested_bytes=%u bytes_read=%u\n",
          esp_err_to_name(result), static_cast<int>(result),
          static_cast<unsigned>(requestedBytes),
          static_cast<unsigned>(bytesRead));
      break;
    }

    const size_t samplesRead = bytesRead / sizeof(int32_t);
    for (size_t index = 0; index < samplesRead; ++index) {
      const int16_t sample = pcm16FromInmp441(raw[index]);
      pcm[capturedSamples++] = sample;
      if (sample != 0) {
        ++stats.nonzero;
      }
      stats.minimum = std::min(stats.minimum, sample);
      stats.maximum = std::max(stats.maximum, sample);
      if (sample == std::numeric_limits<int16_t>::min() ||
          sample == std::numeric_limits<int16_t>::max()) {
        ++stats.clipped;
      }
    }
    drainI2sEvents(stats);
  }

  stopAndUninstallI2s(stats);

  if (capturedSamples == 0) {
    stats.minimum = 0;
    stats.maximum = 0;
  }
  const size_t pcmBytes = capturedSamples * sizeof(int16_t);
  Serial.printf(
      "AUDIO_BEGIN rate=%u channels=1 bits=%u samples=%u bytes=%u\n",
      static_cast<unsigned>(kSampleRate), static_cast<unsigned>(kPcmBits),
      static_cast<unsigned>(capturedSamples), static_cast<unsigned>(pcmBytes));
  Serial.write(reinterpret_cast<const uint8_t*>(pcm), pcmBytes);
  Serial.printf(
      "\nAUDIO_END samples=%u bytes=%u read_errors=%u overruns=%u nonzero=%u min=%d max=%d clipped=%u\n",
      static_cast<unsigned>(capturedSamples), static_cast<unsigned>(pcmBytes),
      static_cast<unsigned>(stats.readErrors), static_cast<unsigned>(stats.overruns),
      static_cast<unsigned>(stats.nonzero),
      static_cast<int>(stats.minimum), static_cast<int>(stats.maximum),
      static_cast<unsigned>(stats.clipped));
  Serial.flush();
  heap_caps_free(pcm);
}

void handleCommand(const char* command) {
  unsigned long seconds = 0;
  char trailing = '\0';
  if (sscanf(command, "CAPTURE %lu %c", &seconds, &trailing) != 1 ||
      seconds < kMinCaptureSeconds || seconds > kMaxCaptureSeconds) {
    Serial.printf("AUDIO_ERROR invalid_command expected=CAPTURE_1_to_30_seconds received=%s\n",
                  command);
    return;
  }
  capture(static_cast<uint32_t>(seconds));
}
}  // namespace

void setup() {
  Serial.begin(kSerialBaud);
  delay(1500);
  Serial.println("=== Amadeus INMP441 capture diagnostic ===");
  Serial.printf("[voice] pins bclk=%d ws=%d data=%d lr=left\n",
                Pins::MIC_BCLK, Pins::MIC_LRCL, Pins::MIC_DATA);
  Serial.printf("[voice] sample_rate=%u data_bits=24 slot_bits=32 pcm_bits=%u channel=left\n",
                static_cast<unsigned>(kSampleRate), static_cast<unsigned>(kPcmBits));
  if (!psramFound()) {
    Serial.println("AUDIO_ERROR psram_not_found");
    return;
  }
  if (!installI2s()) {
    Serial.println("AUDIO_ERROR i2s_initialization_failed");
    return;
  }
  const esp_err_t uninstallResult = i2s_driver_uninstall(kI2sPort);
  i2sEvents = nullptr;
  if (uninstallResult != ESP_OK) {
    printError("i2s_driver_uninstall", uninstallResult);
    Serial.println("AUDIO_ERROR i2s_initialization_failed");
    return;
  }
  Serial.println("INMP441_READY command=CAPTURE_seconds range=1..30");
}

void loop() {
  while (Serial.available() > 0) {
    const char value = static_cast<char>(Serial.read());
    if (value == '\n' || value == '\r') {
      if (commandLength > 0) {
        commandBuffer[commandLength] = '\0';
        handleCommand(commandBuffer);
        commandLength = 0;
      }
    } else if (commandLength + 1 < sizeof(commandBuffer)) {
      commandBuffer[commandLength++] = value;
    } else {
      commandLength = 0;
      Serial.println("AUDIO_ERROR command_too_long");
    }
  }
  delay(1);
}
