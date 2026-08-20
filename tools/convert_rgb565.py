"""Convert an approved 320x480 PNG to little-endian RGB565 without resizing."""

from pathlib import Path
import struct
import sys

from PIL import Image

EXPECTED_SIZE = (320, 480)


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: convert_rgb565.py INPUT.png OUTPUT.rgb565", file=sys.stderr)
        return 2
    source = Path(sys.argv[1])
    destination = Path(sys.argv[2])
    with Image.open(source) as image:
        image.load()
        if image.size != EXPECTED_SIZE:
            raise ValueError(f"expected {EXPECTED_SIZE}, got {image.size}")
        rgb = image.convert("RGB")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as output:
        for red, green, blue in rgb.getdata():
            rgb565 = ((red & 0xF8) << 8) | ((green & 0xFC) << 3) | (blue >> 3)
            output.write(struct.pack("<H", rgb565))
    expected_bytes = EXPECTED_SIZE[0] * EXPECTED_SIZE[1] * 2
    actual_bytes = destination.stat().st_size
    if actual_bytes != expected_bytes:
        raise RuntimeError(f"expected {expected_bytes} bytes, wrote {actual_bytes}")
    print(f"created {destination} ({actual_bytes} bytes, RGB565 little-endian)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

