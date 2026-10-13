"""Rewrite an answer so a TTS engine reads it naturally.

Kept separate from the Kokoro wrapper so it can be unit-tested without importing torch.
"""

from __future__ import annotations

import re

# Currency codes are a vocabulary, not a store fact; the KB decides which one appears.
_CURRENCY_NAMES = {
    "PKR": "Pakistani rupees",
    "USD": "US dollars",
    "EUR": "euros",
    "GBP": "pounds",
    "INR": "Indian rupees",
    "AED": "dirhams",
}
_CURRENCY = re.compile(r"\b([A-Z]{3})\s?(\d[\d,]*(?:\.\d+)?)\b")
_RANGE = re.compile(r"(?<=\d)\s?[–—-]\s?(?=\d)")
_MODEL_CODE = re.compile(r"\b([A-Z])(\d+)\b")
_MARKDOWN = re.compile(r"[*_`#>]+|^\s*[-•]\s+", re.MULTILINE)
_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)


def normalize_for_speech(text: str) -> str:
    """Expand currencies and ranges, space out model codes, strip markdown and think tags."""
    text = _THINK.sub("", text)
    text = _MARKDOWN.sub("", text)
    text = _CURRENCY.sub(
        lambda m: f"{m.group(2)} {_CURRENCY_NAMES.get(m.group(1), m.group(1))}", text
    )
    text = _RANGE.sub(" to ", text)
    text = _MODEL_CODE.sub(r"\1 \2", text)
    text = text.replace("USB-C", "USB C")
    return re.sub(r"\s+", " ", text).strip()
