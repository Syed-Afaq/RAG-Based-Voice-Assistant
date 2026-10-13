"""Compute order totals in code from figures found in the retrieved passages.

Small language models are unreliable at arithmetic, so when a question asks what a product costs
including delivery we parse the currency amounts out of the retrieved text, add them up, and hand
the model a ready-made ``COMPUTED:`` line to phrase. Nothing here knows any price in advance; if the
needed figures are not in the retrieved passages, no total is produced.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from voice_rag.numbers import format_amount, parse_number
from voice_rag.products import Product, mentioned_products
from voice_rag.types import Retrieved

# "PKR 2,500", "USD 10.50": a three-letter currency code followed by an amount.
_AMOUNT = re.compile(r"\b([A-Z]{3})\s?(\d[\d,]*(?:\.\d+)?)\b")
_PRICE_WORDS = re.compile(r"\b(cost|costs|price|pay|total|how much|charge|fee)\b", re.IGNORECASE)
_DELIVERY_WORDS = re.compile(r"\b(deliver\w*|shipping|ship|shipped|postage)\b", re.IGNORECASE)


@dataclass(frozen=True)
class Computed:
    """A total worked out in code, plus the line that is passed to the LLM."""

    product: Product
    price: float
    delivery: float
    currency: str

    @property
    def total(self) -> float:
        """Product price plus delivery fee."""
        return self.price + self.delivery

    @property
    def line(self) -> str:
        """Explicit arithmetic for the prompt, e.g. ``COMPUTED: L1 desk lamp PKR 2,500 + ...``."""
        return (
            f"COMPUTED: {self.product.label} {self.currency} {format_amount(self.price)}"
            f" + delivery {self.currency} {format_amount(self.delivery)}"
            f" = {self.currency} {format_amount(self.total)}."
        )


def wants_total_with_delivery(question: str) -> bool:
    """True when the question asks about price and mentions delivery or shipping."""
    return bool(_PRICE_WORDS.search(question) and _DELIVERY_WORDS.search(question))


def amounts_in(text: str) -> list[tuple[str, float]]:
    """Currency amounts in ``text`` as ``(currency, value)`` pairs, in order of appearance."""
    return [(m.group(1), parse_number(m.group(2))) for m in _AMOUNT.finditer(text)]


def _product_price(product: Product, retrieved: list[Retrieved]) -> tuple[str, float] | None:
    for hit in retrieved:
        if re.search(rf"\b{re.escape(product.code)}\b", hit.chunk.text):
            amounts = amounts_in(hit.chunk.text)
            if amounts:
                return amounts[0]
    return None


def _delivery_fee(retrieved: list[Retrieved]) -> tuple[str, float] | None:
    for hit in retrieved:
        if _DELIVERY_WORDS.search(hit.chunk.text):
            amounts = amounts_in(hit.chunk.text)
            if amounts:
                return amounts[0]
    return None


def compute_totals(
    question: str, products: list[Product], retrieved: list[Retrieved]
) -> list[Computed]:
    """Return one :class:`Computed` per mentioned product with a known price and delivery fee."""
    if not wants_total_with_delivery(question):
        return []
    fee = _delivery_fee(retrieved)
    if fee is None:
        return []
    results: list[Computed] = []
    for product in mentioned_products(question, products):
        price = _product_price(product, retrieved)
        if price is None or price[0] != fee[0]:
            continue  # missing figure or mismatched currency: do not guess
        results.append(Computed(product=product, price=price[1], delivery=fee[1], currency=fee[0]))
    return results
