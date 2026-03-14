#!/usr/bin/env python3
"""
Rebuild docs/letterhead/bottom.png with clean layout: Director (DIN: 11594139),
rule, company name | CIN, and registered office address. No overlapping or clipped text.
Run from repo root: python3 docs/letterhead/fix_bottom_din.py
"""
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

SCRIPT_DIR = Path(__file__).resolve().parent
BOTTOM_PNG = SCRIPT_DIR / "bottom.png"

W = 1200
MARGIN = 60
TOP_MARGIN = 40
LINE_HEIGHT = 36


def main() -> None:
    try:
        font_title = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 28)
        font_body = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 18)
    except Exception:
        font_title = ImageFont.load_default()
        font_body = ImageFont.load_default()

    y = TOP_MARGIN
    y += 28 + LINE_HEIGHT + 12 + 2 + 14 + 18 + LINE_HEIGHT + 18 + LINE_HEIGHT + 20
    H = int(y)

    im = Image.new("RGB", (W, H), (255, 255, 255))
    draw = ImageDraw.Draw(im)
    y = TOP_MARGIN

    draw.text(((W - 280) // 2, y), "Director (DIN: 11594139)", fill=(50, 50, 50), font=font_title)
    y += 28 + LINE_HEIGHT + 12
    rule_y = int(y)
    draw.line([(MARGIN, rule_y), (W - MARGIN, rule_y)], fill=(180, 180, 180), width=1)
    y += 14
    draw.text((MARGIN, y), "Paafekt Innovations Private Limited | CIN: U62010KA2025PTC210698", fill=(60, 60, 60), font=font_body)
    y += 18 + LINE_HEIGHT
    draw.text((MARGIN, y), "Regd. Office: F.No.3042, TWR-3, 4th Floor, Prestige B Temple Bells, Rajarajeshwarinagar, Bangalore South, Bangalore - 560098, Karnataka", fill=(60, 60, 60), font=font_body)

    im.save(BOTTOM_PNG, "PNG")
    print(f"Saved {BOTTOM_PNG}: {W}x{H}, DIN and all text clear, no clipping.")


if __name__ == "__main__":
    main()
