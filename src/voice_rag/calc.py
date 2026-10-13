"""Parse figures out of the retrieved passages and do any arithmetic in code.

Small language models are unreliable at arithmetic, so the pipeline never asks the model to add.
Instead:

* every currency amount in the retrieved text is labelled by the product named in its sentence,
  or as the delivery fee when the sentence talks about delivery;
* durations ("2–3 working days", "48 hours") are listed with the clause they came from;
* when the question combines products, quantities or delivery, the total is computed here and
  handed to the model as an explicit ``COMPUTED:`` line to phrase;
* otherwise the labelled figures go to the model as a ``FACTS:`` block so it reasons from clean
  numbers rather than re-parsing prose.

Nothing in this module knows a price or a city in advance; every figure comes from the passages.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from voice_rag.lexical import tokenize
from voice_rag.numbers import NUMBER_WORDS, extract_amounts, format_amount
from voice_rag.products import Product, mentioned_products
from voice_rag.types import Retrieved

PRICE_WORDS = re.compile(
    r"\b(cost|costs|price|prices|pay|total|how much|charge|charges|fee|fees)\b", re.IGNORECASE
)
DELIVERY_WORDS = re.compile(r"\b(deliver\w*|shipping|ship|shipped|postage)\b", re.IGNORECASE)
_PER_UNIT = re.compile(r"\bper\s+(\w+)\b", re.IGNORECASE)
_NUMBER_WORD = "|".join(sorted(NUMBER_WORDS, key=len, reverse=True))
_DURATION = re.compile(
    rf"\b(\d+(?:\s?[–—-]\s?\d+)?|(?:{_NUMBER_WORD})(?:[-\s](?:{_NUMBER_WORD}))?)\s+"
    r"(working\s+days?|days?|hours?|weeks?|months?|years?)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Amount:
    """One currency figure and what it belongs to."""

    label: str  # "L1 desk lamp", "delivery" or a short snippet of the sentence
    currency: str
    value: float
    per: str | None = None  # e.g. "order" from "per order"
    product: Product | None = None

    @property
    def is_delivery(self) -> bool:
        """True when this is the delivery fee."""
        return self.product is None and self.label == "delivery"

    def describe(self) -> str:
        """``L1 desk lamp: PKR 2,500`` or ``delivery: PKR 200 per order``."""
        text = f"{self.label}: {self.currency} {format_amount(self.value)}"
        return f"{text} per {self.per}" if self.per else text


@dataclass(frozen=True)
class Duration:
    """A time span and the clause it was stated in."""

    text: str  # "2–3 working days"
    clause: str  # "Delivery to Islamabad takes 2–3 working days"

    def describe(self) -> str:
        """``2–3 working days (Delivery to Islamabad takes 2–3 working days)``."""
        return f"{self.text} ({self.clause})"


@dataclass(frozen=True)
class Facts:
    """Everything numeric that was parsed from the retrieved passages."""

    amounts: list[Amount]
    durations: list[Duration]

    def block(self) -> str:
        """The ``FACTS:`` block for the prompt, or an empty string when nothing was parsed."""
        lines = [a.describe() for a in self.amounts] + [d.describe() for d in self.durations]
        if not lines:
            return ""
        return "FACTS (parsed from the passages):\n" + "\n".join(f"- {line}" for line in lines)


@dataclass(frozen=True)
class LineItem:
    """``quantity`` units of ``product`` at ``unit_price`` each."""

    product: Product
    quantity: int
    unit_price: float

    @property
    def subtotal(self) -> float:
        """Quantity times unit price."""
        return self.quantity * self.unit_price


@dataclass(frozen=True)
class Computed:
    """A total worked out in code, plus the line that is passed to the LLM."""

    items: list[LineItem]
    delivery: float | None
    currency: str

    @property
    def total(self) -> float:
        """Sum of line items plus the delivery fee (charged once per order) when it applies."""
        return sum(i.subtotal for i in self.items) + (self.delivery or 0.0)

    @property
    def line(self) -> str:
        """Explicit arithmetic, e.g. ``COMPUTED: 2 x L1 desk lamp PKR 2,500 + delivery ...``."""
        parts = []
        for item in self.items:
            qty = f"{item.quantity} x " if item.quantity != 1 else ""
            parts.append(
                f"{qty}{item.product.label} {self.currency} {format_amount(item.unit_price)}"
            )
        if self.delivery is not None:
            parts.append(f"delivery {self.currency} {format_amount(self.delivery)}")
        return f"COMPUTED: {' + '.join(parts)} = {self.currency} {format_amount(self.total)}."


def _clauses(sentence: str) -> list[str]:
    return [c.strip(" .") for c in sentence.split(";") if c.strip(" .")]


def _product_in(text: str, products: list[Product]) -> Product | None:
    for product in products:
        if re.search(rf"\b{re.escape(product.code)}\b", text):
            return product
    return None


def parse_facts(retrieved: list[Retrieved], products: list[Product]) -> Facts:
    """Label every amount and duration found in the retrieved passages."""
    amounts: list[Amount] = []
    durations: list[Duration] = []
    for hit in retrieved:
        sentence = hit.chunk.text
        product = _product_in(sentence, products)
        for currency, value in extract_amounts(sentence):
            per = _PER_UNIT.search(sentence)
            if product is not None:
                label = product.label
            elif DELIVERY_WORDS.search(sentence):
                label = "delivery"
            else:
                label = sentence[:40].rstrip(" .")
            amounts.append(Amount(label, currency, value, per.group(1) if per else None, product))
        for clause in _clauses(sentence):
            for match in _DURATION.finditer(clause):
                durations.append(Duration(text=match.group(0), clause=clause))
    return Facts(amounts=amounts, durations=durations)


def requested_quantities(question: str, products: list[Product]) -> dict[str, int]:
    """Quantity per mentioned product code, e.g. "two lamps and a speaker" -> {L1: 2, S1: 1}."""
    tokens = tokenize(question)
    quantities: dict[str, int] = {}
    for product in mentioned_products(question, products):
        quantities[product.code] = 1
        for i, token in enumerate(tokens):
            if token not in product.aliases:
                continue
            for previous in tokens[max(0, i - 2) : i]:  # "two L1 lamps", "2 lamps"
                if previous.isdigit():
                    quantities[product.code] = int(previous)
                elif previous in NUMBER_WORDS:
                    quantities[product.code] = NUMBER_WORDS[previous]
            break
    return quantities


def compute_total(question: str, products: list[Product], facts: Facts) -> Computed | None:
    """Total for the mentioned products, quantities and delivery, when arithmetic is needed."""
    if not PRICE_WORDS.search(question):
        return None
    quantities = requested_quantities(question, products)
    wants_delivery = bool(DELIVERY_WORDS.search(question))
    if not quantities:
        return None
    needs_arithmetic = (
        wants_delivery or len(quantities) > 1 or any(q > 1 for q in quantities.values())
    )
    if not needs_arithmetic:
        return None  # a single unit price is already a plain fact in the context

    prices = {a.product.code: a for a in facts.amounts if a.product is not None}
    items: list[LineItem] = []
    for code, quantity in quantities.items():
        amount = prices.get(code)
        if amount is None:
            return None  # a price is missing: do not guess
        product = amount.product
        assert product is not None
        items.append(LineItem(product, quantity, amount.value))

    delivery: Amount | None = next((a for a in facts.amounts if a.is_delivery), None)
    if wants_delivery and delivery is None:
        return None
    currencies = {i.currency for i in facts.amounts if i.product is not None} | (
        {delivery.currency} if wants_delivery and delivery else set()
    )
    if len(currencies) != 1:
        return None  # mixed currencies: do not sum
    return Computed(
        items=items,
        delivery=delivery.value if wants_delivery and delivery else None,
        currency=currencies.pop(),
    )
