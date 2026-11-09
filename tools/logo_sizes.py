"""Pillow is needed only here, so it is not one of Yata's dependencies; run with:

    uv run --no-project --with pillow python tools/logo_sizes.py
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parent.parent
SIZES = {
    "logo": 36,  # .brand-mark's width and height in web/assets/style.css
    "favicon": 16,
}
SCALES = (1, 2, 3)


def main() -> None:
    with Image.open(ROOT / "logo.png") as source:
        # Premultiplied alpha, so the transparent edge does not turn dark as it is scaled down.
        art = source.convert("RGBa")
        for name, size in SIZES.items():
            for scale in SCALES:
                side = size * scale
                image = ImageOps.fit(art, (side, side), Image.Resampling.LANCZOS).convert("RGBA")
                target = ROOT / "web" / "assets" / f"{name}-{side}.png"
                image.save(target, optimize=True)
                print(f"{target.relative_to(ROOT)}: {side}x{side}, {target.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
