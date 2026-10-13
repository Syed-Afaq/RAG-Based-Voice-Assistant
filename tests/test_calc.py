from __future__ import annotations

from voice_rag.calc import amounts_in, compute_totals, wants_total_with_delivery
from voice_rag.products import Product
from voice_rag.types import Chunk, Retrieved

PRODUCTS = [Product("L1", "desk lamp"), Product("S1", "Bluetooth speaker")]


def hits(*texts: str) -> list[Retrieved]:
    return [Retrieved(Chunk(t, "x.txt", i), 0.9 - i * 0.1) for i, t in enumerate(texts)]


def test_wants_total_with_delivery() -> None:
    assert wants_total_with_delivery("What will the L1 lamp cost with delivery to Islamabad?")
    assert wants_total_with_delivery("How much is the speaker including shipping?")
    assert not wants_total_with_delivery("Does the L1 lamp have a battery?")
    assert not wants_total_with_delivery("How much does the L1 cost?")


def test_amounts_in_parses_currency_and_commas() -> None:
    assert amounts_in("The L1 desk lamp costs PKR 2,500. Fee USD 10.50.") == [
        ("PKR", 2500.0),
        ("USD", 10.5),
    ]


def test_total_is_computed_from_retrieved_text() -> None:
    retrieved = hits(
        "The L1 desk lamp costs PKR 2,500.",
        "Delivery costs PKR 200 per order within Pakistan.",
        "Delivery to Islamabad takes 2–3 working days; Lahore takes 3–5 working days.",
    )
    [computed] = compute_totals("What will the L1 lamp cost with delivery?", PRODUCTS, retrieved)
    assert computed.total == 2700
    assert computed.line == "COMPUTED: L1 desk lamp PKR 2,500 + delivery PKR 200 = PKR 2,700."


def test_changed_price_changes_total() -> None:
    """Guards against hardcoded facts: the total follows whatever the passages say."""
    retrieved = hits("The L1 desk lamp costs PKR 3,100.", "Delivery costs PKR 350 per order.")
    [computed] = compute_totals("L1 price with delivery?", PRODUCTS, retrieved)
    assert computed.total == 3450
    assert "3,450" in computed.line


def test_no_total_when_a_figure_is_missing() -> None:
    assert compute_totals("L1 price with delivery?", PRODUCTS, hits("The L1 desk lamp.")) == []
    only_fee = hits("Delivery costs PKR 200 per order.")
    assert compute_totals("L1 price with delivery?", PRODUCTS, only_fee) == []


def test_no_total_when_no_product_mentioned() -> None:
    retrieved = hits("The L1 desk lamp costs PKR 2,500.", "Delivery costs PKR 200 per order.")
    assert compute_totals("How much is delivery?", PRODUCTS, retrieved) == []


def test_currency_mismatch_is_not_summed() -> None:
    retrieved = hits("The L1 desk lamp costs USD 25.", "Delivery costs PKR 200 per order.")
    assert compute_totals("L1 with delivery?", PRODUCTS, retrieved) == []


def test_one_total_per_mentioned_product() -> None:
    retrieved = hits(
        "The L1 desk lamp costs PKR 2,500.",
        "The S1 Bluetooth speaker costs PKR 4,000 and has a battery life of up to six hours.",
        "Delivery costs PKR 200 per order within Pakistan.",
    )
    totals = compute_totals("Lamp and speaker price with delivery?", PRODUCTS, retrieved)
    assert [c.total for c in totals] == [2700, 4200]
