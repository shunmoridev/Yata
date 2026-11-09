from __future__ import annotations

import re
import unicodedata
from decimal import Decimal, InvalidOperation

_SCALE_WORDS = {
    # Single-letter "m"/"b" are left out: "5m" is more often metres than millions.
    "thousand": Decimal(10) ** 3, "k": Decimal(10) ** 3, "million": Decimal(10) ** 6,
    "mn": Decimal(10) ** 6, "billion": Decimal(10) ** 9, "bn": Decimal(10) ** 9,
    "trillion": Decimal(10) ** 12, "tn": Decimal(10) ** 12,
}
_KANJI_DIGITS = {"〇": 0, "零": 0, "一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
_KANJI_SMALL = {"十": 10, "百": 100, "千": 1000}
_KANJI_LARGE = {"万": 10 ** 4, "億": 10 ** 8, "兆": 10 ** 12}
# After digits only: Chinese (萬, 亿) and Korean (천, 백, 만, 억, 조) spellings of the same units, so "40亿" and
# "40억" read as 4 billion like "40億".
_CJK_UNITS = {**_KANJI_SMALL, **_KANJI_LARGE, "萬": 10 ** 4, "亿": 10 ** 8,
              "백": 100, "천": 1000, "만": 10 ** 4, "억": 10 ** 8, "조": 10 ** 12}
_UNITS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
    "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
    "seventeen": 17, "eighteen": 18, "nineteen": 19,
}
_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90}
_MONTHS = {
    name: index for index, names in enumerate([
        ("january", "jan"), ("february", "feb"), ("march", "mar"), ("april", "apr"), ("may",), ("june", "jun"),
        ("july", "jul"), ("august", "aug"), ("september", "sep", "sept"), ("october", "oct"), ("november", "nov"),
        ("december", "dec"),
    ], start=1) for name in names
}

_ARABIC = re.compile(
    r"(?<![A-Za-z0-9.])(\d+(?:,\d{3})+|\d+)(\.\d+)?"
    r"(?:\s?(thousand|million|billion|trillion|mn|bn|tn|k)(?![a-z]))?"
    r"([十百千]?[万萬億亿兆]|[十百千]|[천백]?[만억조]|[천백])?",
    re.IGNORECASE,
)
# Kanji numerals count only next to counters or standing alone, so 一時的 or 一部 are not numbers.
_KANJI = re.compile(
    r"(?:[〇零一二三四五六七八九十百千万億兆]+(?=円|人|個|回|日|月|年|時(?!的|停止)|分|秒|台|本|枚|件|名|歳|才|週間|か月|ヶ月|割|％|%|つ|倍|位|ドル|億|万))"
    r"|(?<![一-龯々])[〇零一二三四五六七八九十百千万億兆]{2,}(?![一-龯々])"
)
_WORD = re.compile(r"[a-z]+(?:-[a-z]+)*")
_NUMBER_WORDS = set(_UNITS) | set(_TENS) | {"hundred", "thousand", "million", "billion", "trillion", "and"}


def _kanji_value(text: str) -> int | None:
    total, section, digit = 0, 0, None
    for char in text:
        if char in _KANJI_DIGITS:
            digit = _KANJI_DIGITS[char] if digit is None else digit * 10 + _KANJI_DIGITS[char]
        elif char in _KANJI_SMALL:
            section += (digit if digit is not None else 1) * _KANJI_SMALL[char]
            digit = None
        elif char in _KANJI_LARGE:
            section += digit or 0
            total += (section or 1) * _KANJI_LARGE[char]
            section, digit = 0, None
    value = total + section + (digit or 0)
    return value if text else None


def _kanji_multiplier(suffix: str) -> int:
    multiplier = 1
    for char in suffix:
        multiplier *= _CJK_UNITS.get(char, 1)
    return multiplier


def _english_values(text: str) -> set[Decimal]:
    values: set[Decimal] = set()
    words = _WORD.findall(text)
    run: list[str] = []
    for word in words + [""]:
        parts = word.split("-")
        if word and all(part in _NUMBER_WORDS for part in parts) and word != "and":
            run.extend(parts)
            continue
        if word == "and" and run:
            continue
        if run:
            total, current = 0, 0
            for part in run:
                if part in _UNITS:
                    current += _UNITS[part]
                elif part in _TENS:
                    current += _TENS[part]
                elif part == "hundred":
                    current = (current or 1) * 100
                elif part in {"thousand", "million", "billion", "trillion"}:
                    total += (current or 1) * int(_SCALE_WORDS[part])
                    current = 0
            values.add(Decimal(total + current))
            run = []
        if word in _MONTHS:
            values.add(Decimal(_MONTHS[word]))
    return values


def _normalize(text: str) -> str:
    return unicodedata.normalize("NFKC", text).replace("，", ",").replace("．", ".")


def _digit_matches(normalized: str) -> list[tuple[Decimal, tuple[int, int]]]:
    found = []
    for match in _ARABIC.finditer(normalized):
        whole, fraction, scale, kanji = match.groups()
        try:
            value = Decimal(whole.replace(",", "") + (fraction or ""))
        except InvalidOperation:
            continue
        if scale:
            value *= _SCALE_WORDS[scale.lower()]
        if kanji:
            value *= _kanji_multiplier(kanji)
        found.append((value.normalize(), match.span()))
    return found


def digit_values(text: str) -> set[Decimal]:
    return {value for value, _ in _digit_matches(_normalize(text))}


def numeric_values(text: str) -> set[Decimal]:
    normalized = _normalize(text)
    matches = _digit_matches(normalized)
    values = {value for value, _ in matches}
    consumed = [span for _, span in matches]
    remainder = list(normalized)
    for start, end in consumed:
        remainder[start:end] = [" "] * (end - start)
    rest = "".join(remainder)
    for match in _KANJI.finditer(rest):
        value = _kanji_value(match.group(0))
        if value is not None:
            values.add(Decimal(value))
    values |= {value.normalize() for value in _english_values(rest.casefold())}
    return values


def numbers_supported(statement: str, sources: list[str]) -> bool:
    claimed = numeric_values(statement)
    if not claimed:
        return True
    available: set[Decimal] = set()
    for source in sources:
        available |= numeric_values(source)
    return claimed <= available


def contains_verbatim(value: str, source: str) -> bool:
    def squash(text: str) -> str:
        return re.sub(r"\s+", "", unicodedata.normalize("NFKC", text)).casefold()
    needle = squash(value)
    return bool(needle) and needle in squash(source)
