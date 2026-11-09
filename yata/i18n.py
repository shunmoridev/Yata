from __future__ import annotations

import os
import sys
from contextvars import ContextVar, Token

SUPPORTED = ("ja", "en")
FALLBACK = "ja"

_current: ContextVar[str | None] = ContextVar("yata_locale", default=None)
_state: dict[str, str | None] = {"choice": None, "page": None}


def normalize(value: object) -> str | None:
    primary = str(value or "").strip().replace("_", "-").split("-")[0].split(".")[0].lower()
    return primary if primary in SUPPORTED else None


def negotiate(header: str | None) -> str | None:
    entries: list[tuple[float, int, str]] = []
    for index, part in enumerate(str(header or "").split(",")):
        tag, _, params = part.strip().partition(";")
        if not tag.strip():
            continue
        quality = 1.0
        for param in params.split(";"):
            name, _, value = param.strip().partition("=")
            if name.strip() == "q":
                try:
                    quality = float(value)
                except ValueError:
                    quality = 0.0
        if quality > 0:
            entries.append((-quality, index, tag.strip()))
    if not entries:
        return None
    return next((found for *_, tag in sorted(entries) if (found := normalize(tag))), FALLBACK)


def system_locale() -> str | None:
    if sys.platform == "win32":
        try:
            import ctypes

            primary = ctypes.windll.kernel32.GetUserDefaultUILanguage() & 0x3FF  # type: ignore[attr-defined]
            return {0x11: "ja", 0x09: "en"}.get(primary)
        except (AttributeError, OSError):
            return None
    for name in ("LC_ALL", "LC_MESSAGES", "LANG", "LANGUAGE"):
        value = os.environ.get(name, "").strip()
        if value and value not in ("C", "POSIX"):
            return normalize(value.split(":")[0])
    return None


def set_choice(value: str | None) -> None:
    _state["choice"] = normalize(value) if value else None


def remember_page(value: str | None) -> None:
    if value in SUPPORTED:
        _state["page"] = value


def background() -> str:
    return _state["choice"] or _state["page"] or system_locale() or FALLBACK


def for_request(choice: str | None, accept_language: str | None) -> str:
    return normalize(choice) or negotiate(accept_language) or background()


def locale() -> str:
    return _current.get() or background()


def use(value: str) -> Token[str | None]:
    return _current.set(value if value in SUPPORTED else FALLBACK)


def reset(token: Token[str | None] | None = None) -> None:
    if token is not None:
        _current.reset(token)
        return
    _state["choice"] = None
    _state["page"] = None


def capitalize(value: object) -> str:
    text = str(value)
    return text[:1].upper() + text[1:]


def t(ja: str, en: str) -> str:
    return en if locale() == "en" else ja


class Text:
    __slots__ = ("ja", "en")

    def __init__(self, ja: str, en: str) -> None:
        self.ja = ja
        self.en = en

    def __str__(self) -> str:
        return t(self.ja, self.en)

    def __repr__(self) -> str:
        return f"Text({self.ja!r}, {self.en!r})"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Text) and (self.ja, self.en) == (other.ja, other.en)

    def __hash__(self) -> int:
        return hash((self.ja, self.en))
