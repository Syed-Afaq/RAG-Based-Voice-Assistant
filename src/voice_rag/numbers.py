"""Extract and normalise numeric figures from text (used by the calculator and output guard)."""

from __future__ import annotations

import re

_UNITS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
    "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19,
}  # fmt: skip
_TENS = {
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70,
    "eighty": 80, "ninety": 90,
}  # fmt: skip

# Digits not glued to a letter, so "L1" and "S1" are product codes rather than the number 1.
_DIGITS = re.compile(r"(?<![A-Za-z0-9])\d[\d,]*(?:\.\d+)?(?![A-Za-z0-9])")
_WORDS = re.compile(
    r"\b("
    + "|".join(_TENS)
    + r")(?:[-\s]("
    + "|".join(_UNITS)
    + r"))?\b|\b("
    + "|".join(_UNITS)
    + r")\b",
    re.IGNORECASE,
)


def parse_number(token: str) -> float:
    """Turn ``"2,700"`` or ``"2.5"`` into a float."""
    return float(token.replace(",", ""))


def extract_numbers(text: str) -> set[float]:
    """All numeric values in ``text``, written as digits or as English number words."""
    found = {parse_number(m.group()) for m in _DIGITS.finditer(text)}
    for tens, unit, lone in _WORDS.findall(text):
        if lone:
            found.add(float(_UNITS[lone.lower()]))
        else:
            found.add(float(_TENS[tens.lower()] + (_UNITS[unit.lower()] if unit else 0)))
    return found


def format_amount(value: float) -> str:
    """Render ``2700.0`` as ``"2,700"`` and ``2.5`` as ``"2.5"``."""
    return f"{value:,.0f}" if value == int(value) else f"{value:,.2f}".rstrip("0")
