from __future__ import annotations

import numpy as np

from voice_rag.lexical import BM25, singular, stem, tokenize


def test_singular_strips_common_plurals() -> None:
    assert singular("batteries") == "battery"
    assert singular("lamps") == "lamp"
    assert singular("days") == "day"
    assert singular("glass") == "glass"
    assert singular("bus") == "bus"


def test_stem_joins_related_forms() -> None:
    assert stem("delivery") == stem("delivered") == stem("deliver") == "deliver"
    assert stem("shipping") == stem("shipped") == "ship"
    assert stem("batteries") == stem("battery") == "batter"
    assert stem("day") == "day" and stem("days") == "day"  # never below three letters
    assert stem("l1") == "l1"


def test_tokenize_lowercases_and_drops_punctuation() -> None:
    assert tokenize("Does the L1 lamp have batteries?") == [
        "doe", "the", "l1", "lamp", "have", "batter",
    ]  # fmt: skip


def test_bm25_prefers_exact_rare_tokens() -> None:
    docs = [
        "The L1 desk lamp costs PKR 2,500.",
        "The S1 Bluetooth speaker costs PKR 4,000.",
        "Delivery costs PKR 200 per order within Pakistan.",
    ]
    scores = BM25(docs).scores("S1 speaker")
    assert int(np.argmax(scores)) == 1
    assert scores[2] == 0.0  # no overlapping tokens


def test_bm25_ignores_function_words() -> None:
    scores = BM25(["The L1 desk lamp costs PKR 2,500.", "It uses USB-C power."]).scores(
        "What is the weather?"
    )
    assert not scores.any()
    assert tokenize("the lamp is here", drop_stop_words=True) == ["lamp"]


def test_bm25_plural_query_matches_singular_document() -> None:
    scores = BM25(["It has no built-in battery.", "Delivery costs PKR 200."]).scores("batteries?")
    assert scores[0] > 0 and scores[1] == 0
