"""Build deterministic happy-heart overlay previews from a heart reference."""

from pathlib import Path

from PIL import Image, ImageChops, ImageEnhance, ImageFilter


ROOT = Path(__file__).resolve().parents[1]
HAPPY = ROOT / "assets" / "source" / "characters" / "happy_320x480.png"
HEART_REFERENCE = Path(
    r"C:\Users\JusticeJun\.codex\generated_images"
    r"\01a01df2-e284-7092-a2f7-405fc657c6c1"
    r"\exec-7040ea05-76f5-475d-b48c-f649270d883c.png"
)
OUTPUT = ROOT / "assets" / "previews" / "happy_hearts"
HEART_OUTPUT = ROOT / "assets" / "source" / "icons" / "happy_heart.png"

# Center X/Y, width/height, clockwise rotation. These are mapped from the
# user's two final 384x572 placement references onto the 320x480 LCD image.
FRAME_A = (
    (61, 42, 33, 34, -8),    # upper-left
    (39, 233, 33, 34, -10),  # lower-left
    (285, 151, 33, 34, 8),   # upper-right
)
FRAME_B = (
    (259, 53, 33, 34, 8),    # upper-right
    (48, 150, 32, 33, -10),  # upper-left
    (273, 244, 32, 33, 8),   # lower-right
)


def extract_heart() -> Image.Image:
    source = Image.open(HEART_REFERENCE).convert("RGB")
    red, green, blue = source.split()
    other = ImageChops.lighter(green, blue)
    redness = ImageChops.subtract(red, other, scale=1.0, offset=-8)
    alpha = ImageEnhance.Contrast(redness).enhance(4.0)
    alpha = alpha.point(lambda value: 0 if value < 18 else min(255, value * 3))
    alpha = alpha.filter(ImageFilter.GaussianBlur(0.7))
    bbox = alpha.getbbox()
    if bbox is None:
        raise RuntimeError("Heart extraction produced an empty mask")
    rgba = source.convert("RGBA")
    rgba.putalpha(alpha)
    return rgba.crop(bbox)


def place(base: Image.Image, heart: Image.Image, placement) -> None:
    center_x, center_y, width, height, angle = placement
    sprite = heart.resize((width, height), Image.Resampling.LANCZOS)
    sprite = sprite.rotate(angle, resample=Image.Resampling.BICUBIC, expand=True)
    x = round(center_x - sprite.width / 2)
    y = round(center_y - sprite.height / 2)
    base.alpha_composite(sprite, (x, y))


def make_frame(background: Image.Image, heart: Image.Image, placements):
    frame = background.copy()
    for placement in placements:
        place(frame, heart, placement)
    return frame


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    background = Image.open(HAPPY).convert("RGBA")
    heart = extract_heart()
    HEART_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    heart.save(HEART_OUTPUT, optimize=True)
    frame_a = make_frame(background, heart, FRAME_A)
    frame_b = make_frame(background, heart, FRAME_B)
    frame_a.save(OUTPUT / "frame_a.png", optimize=True)
    frame_b.save(OUTPUT / "frame_b.png", optimize=True)
    frame_a.save(
        OUTPUT / "happy_hearts_preview.gif",
        save_all=True,
        append_images=[frame_b],
        duration=500,
        loop=0,
        disposal=2,
    )


if __name__ == "__main__":
    main()
