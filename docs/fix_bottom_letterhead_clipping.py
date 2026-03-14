#!/usr/bin/env python3
"""
Fix clipped DIN in docs/letterhead/bottom.png by removing the baked-in "Director (DIN: ...)"
line from the image. add_letterhead_to_docx.py adds that line as text below the image, so
the DIN is never clipped.
"""
from pathlib import Path

from PIL import Image

DOCS = Path(__file__).resolve().parent
BOTTOM_PNG = DOCS / "letterhead" / "bottom.png"
# Crop this many pixels from top to remove the clipped Director/DIN line (image keeps rule + CIN + address)
CROP_TOP_PX = 90
# Small white margin above the rule after crop
PAD_TOP_PX = 16
PAD_LEFT_PX = 48
PAD_RIGHT_PX = 48
PAD_BOTTOM_PX = 24


def main() -> None:
    if not BOTTOM_PNG.exists():
        raise SystemExit(f"Not found: {BOTTOM_PNG}")

    im = Image.open(BOTTOM_PNG)
    im.load()
    if im.mode != "RGB":
        im = im.convert("RGB")

    w, h = im.size
    if CROP_TOP_PX >= h:
        raise SystemExit(f"CROP_TOP_PX ({CROP_TOP_PX}) >= image height ({h})")
    cropped = im.crop((0, CROP_TOP_PX, w, h))
    cw, ch = cropped.size
    new_w = cw + PAD_LEFT_PX + PAD_RIGHT_PX
    new_h = ch + PAD_TOP_PX + PAD_BOTTOM_PX
    out = Image.new("RGB", (new_w, new_h), (255, 255, 255))
    out.paste(cropped, (PAD_LEFT_PX, PAD_TOP_PX))
    out.save(BOTTOM_PNG, "PNG")
    print(f"Cropped {CROP_TOP_PX}px from top, added padding L{PAD_LEFT_PX} R{PAD_RIGHT_PX} T{PAD_TOP_PX} B{PAD_BOTTOM_PX} -> {BOTTOM_PNG} (size now {new_w}x{new_h})")


if __name__ == "__main__":
    main()
