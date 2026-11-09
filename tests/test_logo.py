from __future__ import annotations

import re
import struct
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PAGE = ROOT / "web" / "index.html"


def png_size(path: Path) -> tuple[int, int]:
    header = path.read_bytes()[:24]
    assert header[:8] == b"\x89PNG\r\n\x1a\n", path
    return struct.unpack(">II", header[16:24])


class LogoTests(unittest.TestCase):
    def assert_logo_at(self, name: str, side: int) -> None:
        path = ROOT / "web" / "assets" / name
        self.assertEqual(png_size(path), (side, side), f"{name}: run tools/logo_sizes.py after changing logo.png")
        self.assertLess(path.stat().st_size, (ROOT / "logo.png").stat().st_size // 20)

    def test_each_image_the_page_asks_for_is_logo_png_at_its_scale(self):
        page = PAGE.read_text(encoding="utf-8")
        tag = re.search(r'<img [^>]*class="brand-mark"[^>]*>', page, re.DOTALL).group(0)
        size = int(re.search(r'width="(\d+)"', tag).group(1))
        css = (ROOT / "web" / "assets" / "style.css").read_text(encoding="utf-8")
        self.assertIn(f".brand-mark {{ width: {size}px; height: {size}px;", css, "the images are made for the mark's size")
        candidates = re.findall(r"/assets/(logo-\d+\.png) (\d)x", re.search(r'srcset="([^"]+)"', tag).group(1))
        self.assertEqual([scale for _, scale in candidates], ["1", "2", "3"])
        self.assertIn('src="/assets/logo-36.png"', tag)
        for name, scale in candidates:
            self.assert_logo_at(name, size * int(scale))

    def test_the_tab_icon_is_logo_png_at_16_px_for_each_scale(self):
        icons = re.findall(r'<link rel="icon" type="image/png" sizes="(\d+)x\1" href="/assets/(favicon-\d+\.png)">',
                           PAGE.read_text(encoding="utf-8"))
        self.assertEqual([int(side) for side, _ in icons], [16, 32, 48], "a tab's 16 px at 1x, 2x and 3x")
        for side, name in icons:
            self.assert_logo_at(name, int(side))


if __name__ == "__main__":
    unittest.main()
