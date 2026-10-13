"""Deterministic checks around the LLM call. Each one is plain code so it can be unit-tested.

Input side:
    * ``detect_injection``: flags requests to ignore or override policy. The request still runs,
      but the prompt gets a note that the documents take precedence.
    * ``needs_clarification``: a price question with no product and no other subject gets a
      clarifying reply without calling the model.
    * ``is_off_topic``: a coarse filter on the best retrieval score. It cannot detect a missing
      detail about a known product (the product sentence still scores high); that job belongs to
      the prompt rules and the output guard.

Output side:
    * ``unsupported_figures``: every number in the answer must appear in the retrieved passages or
      in a COMPUTED line, otherwise the answer is not spoken.
"""

from __future__ import annotations

import re

from voice_rag.numbers import extract_numbers
from voice_rag.products import Product, mentioned_products
from voice_rag.types import GuardResult, Retrieved

_INJECTION_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (
        "asks to ignore or override policy/instructions",
        re.compile(
            r"\b(ignore|disregard|forget|override|bypass|skip)\b.{0,40}"
            r"\b(polic\w*|instruction\w*|rule\w*|context|document\w*|guideline\w*)\b",
            re.IGNORECASE | re.DOTALL,
        ),
    ),
    ("asks to confirm a claim", re.compile(r"\bconfirm that\b", re.IGNORECASE)),
    (
        "asks the assistant to pretend",
        re.compile(r"\b(pretend|act as if|imagine)\b", re.IGNORECASE),
    ),
    ("tries to reassign the assistant's role", re.compile(r"\byou are now\b", re.IGNORECASE)),
    ("claims a new system prompt", re.compile(r"\bsystem prompt\b", re.IGNORECASE)),
]

_PRICE_QUESTION = re.compile(
    r"\b(cost|costs|price|prices|pricing|how much|charge|charges|fee|fees)\b", re.IGNORECASE
)
# Function words, price vocabulary and contraction fragments ("what's" -> "what", "s");
# whatever remains after removing these is the subject of the question.
_FILLER = frozenset(
    """
    a an the this that these those it its one ones they them there here
    i you we me my your our us
    is are was were be been do does did will would can could should shall may might
    have has had
    how much many what which who whom whose when where why
    cost costs price prices pricing charge charges fee fees pay paying total
    of for to in on at with and or but if so please tell say
    s t re ve ll d m
    """.split()
)


def detect_injection(question: str) -> list[str]:
    """Descriptions of every override pattern found in the question (empty when clean)."""
    return [label for label, pattern in _INJECTION_PATTERNS if pattern.search(question)]


def needs_clarification(question: str, products: list[Product]) -> bool:
    """True for a price question whose only subject is a pronoun, e.g. "How much does it cost?"."""
    if not _PRICE_QUESTION.search(question) or mentioned_products(question, products):
        return False
    tokens = re.findall(r"[a-z0-9]+", question.lower())
    return not [t for t in tokens if t not in _FILLER]


def clarification_reply(products: list[Product]) -> str:
    """The deterministic reply used instead of calling the model."""
    if not products:
        return "Which product do you mean?"
    names = [p.label for p in products]
    listed = names[0] if len(names) == 1 else ", ".join(names[:-1]) + f" or {names[-1]}"
    return f"Which product do you mean: the {listed}?"


def is_off_topic(retrieved: list[Retrieved], min_similarity: float) -> bool:
    """True when even the best passage scores below the configured floor."""
    return not retrieved or retrieved[0].score < min_similarity


def unsupported_figures(answer: str, supporting_texts: list[str]) -> list[str]:
    """Numbers in ``answer`` that occur in none of ``supporting_texts`` (as digits or words)."""
    allowed: set[float] = set()
    for text in supporting_texts:
        allowed |= extract_numbers(text)
    unsupported = sorted(extract_numbers(answer) - allowed)
    return [f"{n:g}" for n in unsupported]


def check_output(answer: str, supporting_texts: list[str]) -> GuardResult:
    """Wrap :func:`unsupported_figures` in a :class:`GuardResult` with a terminal message."""
    bad = unsupported_figures(answer, supporting_texts)
    if not bad:
        return GuardResult(ok=True)
    return GuardResult(
        ok=False,
        messages=(f"answer contains figures not found in the documents: {', '.join(bad)}",),
    )
