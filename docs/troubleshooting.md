# Troubleshooting

Change only one variable per diagnostic attempt and record the observed result.

- No serial port: verify the cable and use the USB-C port marked `COM`.
- Upload timeout: note the exact error before trying boot/reset controls.
- Wrong Flash or missing PSRAM: preserve the serial output, then inspect one memory setting at a time.
- Blank LCD: first re-check power and every wire; do not change several display settings together.
- Solid-color test order: red, green, blue, white, black; each color remains for two seconds.
- If solid colors are correct but RGB565 assets have distorted colors, verify the 16-bit byte order before changing RGB/BGR settings.
