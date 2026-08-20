# PC bridge serial protocol

The PC bridge and ESP32 communicate over the CH343 UART at 115200 baud. Each command is one UTF-8/ASCII JSON object terminated by `\n`.

```json
{"type":"state","emotion":"thinking"}
```

Supported emotions are `neutral`, `happy`, `shy`, `pout`, `surprised`, `thinking`, `sleep`, and `listening`.

The ESP32 limits a line to 160 bytes, ignores unknown command types and malformed JSON, and maps unknown emotions to `neutral`. Until emotion-specific assets exist, every state safely keeps the approved neutral image and triggers the existing CRT transition glitch.

Only one Windows program can own COM3 at a time. Close the PlatformIO serial monitor before starting the PC bridge with serial enabled. Stop the bridge before reopening the monitor or uploading firmware.

