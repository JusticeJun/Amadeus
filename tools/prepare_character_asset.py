"""Create a 320x480 character derivative without altering the source image."""

from pathlib import Path
import sys

from PIL import Image


TARGET_SIZE = (320, 480)


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: prepare_character_asset.py INPUT OUTPUT", file=sys.stderr)
        return 2

    source = Path(sys.argv[1])
    destination = Path(sys.argv[2])
    with Image.open(source) as image:
        image.load()
        if image.width * TARGET_SIZE[1] != image.height * TARGET_SIZE[0]:
            raise ValueError(
                f"source aspect ratio {image.size} does not match {TARGET_SIZE}"
            )
        prepared = image.convert("RGB").resize(TARGET_SIZE, Image.Resampling.LANCZOS)

    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite {destination}")
    prepared.save(destination, optimize=True)
    print(f"created {destination} from {source} ({image.size} -> {TARGET_SIZE})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
