"""Derive the store's product catalogue from the knowledge base instead of hardcoding it.

Assumption (documented in the README): products are introduced in the KB as a model code made of a
capital letter and digits followed by a short lowercase name, e.g. "L1 desk lamp" or
"S1 Bluetooth speaker". The name ends at the first verb-like word ("costs", "has", ...) or at
punctuation. If the KB names products differently, this parser needs adjusting.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from voice_rag.types import Chunk

_CODE = re.compile(r"\b([A-Z]\d+)\b((?:\s+[A-Za-z][\w-]*){0,4})")
# Words that cannot be part of a product name; they mark where the name stops.
_STOP_WORDS = frozenset({"costs", "cost", "is", "has", "have", "uses", "comes", "and", "or", "the"})
MAX_NAME_WORDS = 3


@dataclass(frozen=True)
class Product:
    """A product mentioned in the KB, e.g. code ``L1`` with name ``desk lamp``."""

    code: str
    name: str

    @property
    def label(self) -> str:
        """Display form such as ``L1 desk lamp``."""
        return f"{self.code} {self.name}".strip()

    @property
    def aliases(self) -> frozenset[str]:
        """Lowercase tokens that identify this product in a question (code and head noun)."""
        words = self.name.lower().split()
        aliases = {self.code.lower()}
        if words:
            aliases.add(words[-1])  # head noun: "lamp", "speaker"
        return frozenset(aliases)


def _name_after_code(tail: str) -> str:
    words: list[str] = []
    for word in tail.split():
        if word.lower() in _STOP_WORDS or len(words) >= MAX_NAME_WORDS:
            break
        words.append(word)
    return " ".join(words)


def derive_products(chunks: list[Chunk]) -> list[Product]:
    """Scan all chunks for product codes and return unique products in order of appearance."""
    seen: dict[str, Product] = {}
    for chunk in chunks:
        for match in _CODE.finditer(chunk.text):
            code, tail = match.group(1), match.group(2)
            if code not in seen:
                seen[code] = Product(code=code, name=_name_after_code(tail))
    return list(seen.values())


def mentioned_products(question: str, products: list[Product]) -> list[Product]:
    """Products whose code or head noun appears as a whole word in the question."""
    tokens = set(re.findall(r"[a-z0-9]+", question.lower()))
    return [p for p in products if p.aliases & tokens]
