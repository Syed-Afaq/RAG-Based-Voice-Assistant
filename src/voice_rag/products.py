"""Derive the store's product catalogue from the knowledge base instead of hardcoding it.

Assumption (documented in the README): products are introduced in the KB as a model code made of a
capital letter and digits followed by a short lowercase name, e.g. "L1 desk lamp" or
"S1 Bluetooth speaker". The name ends at the first verb-like word ("costs", "has", ...) or at
punctuation. If the KB names products differently, this parser needs adjusting.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from voice_rag.lexical import stem, tokenize
from voice_rag.types import Chunk

_CODE = re.compile(r"\b([A-Z]\d+)\b((?:\s+[A-Za-z][\w-]*){0,4})")
# Words that cannot be part of a product name; they mark where the name stops.
_PRONOUN = re.compile(r"\b(its|it|this one|that one|this|that|one)\b", re.IGNORECASE)
_ALL_PRODUCTS = re.compile(r"\b(both|all|every|each)\b.{0,12}\b(product|item)s?\b", re.IGNORECASE)
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
        """Stemmed lowercase tokens that identify this product in a question (code, head noun)."""
        words = self.name.lower().split()
        aliases = {self.code.lower()}
        if words:
            aliases.add(stem(words[-1]))  # head noun: "lamp", "speaker"
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
    """Products whose code or head noun (singular or plural) appears in the question.

    "Both products" / "all items" counts as naming every product.
    """
    if _ALL_PRODUCTS.search(question):
        return list(products)
    tokens = set(tokenize(question))
    return [p for p in products if p.aliases & tokens]


def product_vocabulary(chunks: list[Chunk], products: list[Product]) -> frozenset[str]:
    """Stemmed words that occur only in passages naming a product ("battery", "usb", "power").

    Used to recognise a pronoun question about a product attribute ("Does it have a battery?")
    without hardcoding attribute names: words that also appear in non-product passages (such as
    "cost" or "hours") are excluded so they do not trigger a clarification.
    """
    codes = {p.code for p in products}
    product_words: set[str] = set()
    other_words: set[str] = set()
    for chunk in chunks:
        words = set(tokenize(chunk.text, drop_stop_words=True))
        if any(re.search(rf"\b{re.escape(code)}\b", chunk.text) for code in codes):
            product_words |= words
        else:
            other_words |= words
    return frozenset(product_words - other_words - {c.lower() for c in codes})


def resolve_pronoun(question: str, product: Product) -> str:
    """Rewrite "How much does it cost?" as "How much does the L1 desk lamp cost?".

    The first pronoun is replaced; "its" becomes a possessive. A question with no pronoun gets the
    product appended in brackets so the model still knows the subject.
    """
    match = _PRONOUN.search(question)
    if match is None:
        return f"{question.rstrip()} (about the {product.label})"
    word = match.group(1).lower()
    replacement = f"the {product.label}'s" if word == "its" else f"the {product.label}"
    return question[: match.start()] + replacement + question[match.end() :]
