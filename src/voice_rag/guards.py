"""Deterministic checks around the LLM call. Each one is plain code so it can be unit-tested.

Input side:
    * ``detect_injection``: flags requests to ignore or override policy. The request still runs,
      but the prompt gets a note that the documents take precedence.
    * ``needs_clarification``: a price question with no product and no other subject gets a
      clarifying reply without calling the model.

Output side:
    * ``unsupported_figures``: every number in the answer must be supported by the retrieved
      passages or the COMPUTED line, otherwise the answer is not spoken. "Supported" means it
      appears literally, or it is a sum/difference of up to three currency amounts from the
      context, optionally multiplied by a small quantity stated in the question. That lets a
      correct total the model worked out pass, while an invented "one year warranty" still fails
      (durations are never combined, so 1 cannot be derived from "2–3 days"). Figures the user
      typed in the question are also accepted, so "10 days ago" can be answered.

Off-topic questions are not rejected here: retrieval falls back to sending every passage when
confidence is low, and the prompt rules tell the model to say when the documents do not answer.
"""

from __future__ import annotations

import re
from itertools import combinations
from itertools import product as cartesian

from voice_rag.numbers import extract_amounts, extract_numbers
from voice_rag.products import Product, mentioned_products
from voice_rag.types import GuardResult

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
    (
        "asks for the instructions",
        re.compile(r"\b(system prompt|your (instructions|rules|prompt))\b", re.IGNORECASE),
    ),
]
LEAK_LABEL = "asks for the instructions"

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


MAX_QUANTITY = 20
MAX_TERMS = 3


def _question_quantities(question: str) -> set[int]:
    """Small whole numbers stated in the question, e.g. "two lamps" -> {2}."""
    return {int(n) for n in extract_numbers(question) if n == int(n) and 1 < n <= MAX_QUANTITY}


def derivable_amounts(supporting_texts: list[str], question: str = "") -> set[float]:
    """Totals a correct answer might contain: sums/differences of up to three context amounts."""
    base = {value for text in supporting_texts for _, value in extract_amounts(text)}
    scaled = base | {value * q for value in base for q in _question_quantities(question)}
    derived: set[float] = set(scaled)
    for n in range(2, MAX_TERMS + 1):
        for terms in combinations(sorted(scaled), n):
            for signs in cartesian((1, -1), repeat=n - 1):
                total = abs(terms[0] + sum(s * t for s, t in zip(signs, terms[1:], strict=True)))
                if total > 0:
                    derived.add(total)
    return derived


def unsupported_figures(answer: str, supporting_texts: list[str], question: str = "") -> list[str]:
    """Numbers in ``answer`` that are neither in ``supporting_texts`` nor derivable from them."""
    allowed: set[float] = set()
    for text in supporting_texts:
        allowed |= extract_numbers(text)
    allowed |= derivable_amounts(supporting_texts, question)
    allowed |= extract_numbers(question)  # echoing the user's own figure is not an invention
    unsupported = sorted(extract_numbers(answer) - allowed)
    return [f"{n:g}" for n in unsupported]


def check_output(answer: str, supporting_texts: list[str], question: str = "") -> GuardResult:
    """Wrap :func:`unsupported_figures` in a :class:`GuardResult` with a terminal message."""
    bad = unsupported_figures(answer, supporting_texts, question)
    if not bad:
        return GuardResult(ok=True)
    return GuardResult(
        ok=False,
        messages=(f"answer contains figures not found in the documents: {', '.join(bad)}",),
    )
