from __future__ import annotations

from pathlib import Path

from voice_rag.index import chunk_file
from voice_rag.products import (
    Product,
    derive_products,
    mentioned_products,
    product_vocabulary,
    resolve_pronoun,
)
from voice_rag.types import Chunk


def test_derives_products_from_real_kb(kb_dir: Path) -> None:
    chunks = [c for p in sorted(kb_dir.glob("*.txt")) for c in chunk_file(p)]
    products = derive_products(chunks)
    assert products == [
        Product(code="L1", name="desk lamp"),
        Product(code="S1", name="Bluetooth speaker"),
    ]


def test_derivation_follows_kb_edits() -> None:
    chunks = [Chunk("The K9 kettle costs PKR 1,000.", "products.txt", 0)]
    assert derive_products(chunks) == [Product(code="K9", name="kettle")]


def test_name_stops_at_verb_or_limit() -> None:
    chunks = [Chunk("The Z2 ultra wide curved gaming monitor has HDR.", "p.txt", 0)]
    assert derive_products(chunks)[0].name == "ultra wide curved"


def test_mentioned_products_by_code_or_head_noun() -> None:
    products = [Product("L1", "desk lamp"), Product("S1", "Bluetooth speaker")]
    assert mentioned_products("Does the l1 lamp have a battery?", products) == [products[0]]
    assert mentioned_products("does the lamp need batteries", products) == [products[0]]
    assert mentioned_products("S1 warranty?", products) == [products[1]]
    assert mentioned_products("How much does it cost?", products) == []
    assert len(mentioned_products("lamp or speaker?", products)) == 2
    assert mentioned_products("total for both products with delivery?", products) == products
    assert mentioned_products("do all items ship?", products) == products


def test_label_and_aliases() -> None:
    p = Product("S1", "Bluetooth speaker")
    assert p.label == "S1 Bluetooth speaker"
    assert p.aliases == frozenset({"s1", "speaker"})


def test_product_vocabulary_is_specific_to_product_passages(kb_dir: Path) -> None:
    chunks = [c for p in sorted(kb_dir.glob("*.txt")) for c in chunk_file(p)]
    vocab = product_vocabulary(chunks, derive_products(chunks))
    assert {"batter", "usb", "power", "bluetooth"} <= vocab
    assert "cost" not in vocab and "hour" not in vocab and "pkr" not in vocab  # shared words
    assert "l1" not in vocab  # codes are handled by mentioned_products


def test_resolve_pronoun() -> None:
    lamp = Product("L1", "desk lamp")
    assert resolve_pronoun("How much does it cost?", lamp) == "How much does the L1 desk lamp cost?"
    assert resolve_pronoun("What is its price?", lamp) == "What is the L1 desk lamp's price?"
    assert (
        resolve_pronoun("What's the price?", lamp) == "What's the price? (about the L1 desk lamp)"
    )
