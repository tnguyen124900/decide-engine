"""Generate the DECIDE app icon (white rounded square + logo) as
packaging/DECIDE.icns (macOS), DECIDE.ico (Windows) and DECIDE-icon.png.

    python packaging/make_icons.py        (needs Pillow)
"""
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parent.parent
LOGO = ROOT / "assets" / "decide-logo-full.png"
OUT = Path(__file__).resolve().parent
S = 1024  # canvas
BODY = 824  # macOS icon body inside the 1024 canvas
RADIUS = 185
D_MARK_BOTTOM = 620  # rows below this (wordmark, subtitle, tagline) are dropped


def build() -> Image.Image:
    canvas = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    off = (S - BODY) // 2
    # soft shadow
    shadow = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    ImageDraw.Draw(shadow).rounded_rectangle((off, off + 12, off + BODY, off + BODY + 12), RADIUS, fill=(0, 0, 0, 70))
    canvas.alpha_composite(shadow.filter(ImageFilter.GaussianBlur(14)))
    # white body
    ImageDraw.Draw(canvas).rounded_rectangle((off, off, off + BODY, off + BODY), RADIUS, fill=(255, 255, 255, 255))
    # logo, trimmed to its content and centred inside the body
    logo = Image.open(LOGO).convert("RGBA")
    # keep only the "D" mark: it is the top block of the full logo, above the wordmark
    logo = logo.crop((0, 0, logo.width, D_MARK_BOTTOM))
    bbox = logo.getchannel("A").point(lambda v: 255 if v > 8 else 0).getbbox()
    if bbox:
        logo = logo.crop(bbox)
    target = int(BODY * 0.66)
    scale = min(target / logo.width, target / logo.height)
    logo = logo.resize((max(1, int(logo.width * scale)), max(1, int(logo.height * scale))), Image.LANCZOS)
    canvas.alpha_composite(logo, ((S - logo.width) // 2, (S - logo.height) // 2))
    return canvas


if __name__ == "__main__":
    icon = build()
    icon.save(OUT / "DECIDE-icon.png")
    icon.save(OUT / "DECIDE.icns", sizes=[(16, 16), (32, 32), (64, 64), (128, 128), (256, 256), (512, 512), (1024, 1024)])
    icon.save(OUT / "DECIDE.ico", sizes=[(16, 16), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
    print("wrote", [p.name for p in OUT.glob("DECIDE*") if p.suffix in (".png", ".icns", ".ico")])
