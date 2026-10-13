from __future__ import annotations

from voice_rag.calc import compute_total, parse_facts, requested_quantities
from voice_rag.products import Product
from voice_rag.types import Chunk, Retrieved

PRODUCTS = [Product("L1", "desk lamp"), Product("S1", "Bluetooth speaker")]
KB = [
    "The L1 desk lamp costs PKR 2,500.",
    "It uses USB-C power and has no built-in battery.",
    "The S1 Bluetooth speaker costs PKR 4,000 and has a battery life of up to six hours.",
    "Delivery costs PKR 200 per order within Pakistan.",
    "Delivery to Islamabad takes 2–3 working days; Lahore takes 3–5 working days.",
    "Return requests are accepted within seven days of delivery for unused items.",
    "Damaged items must be reported within 48 hours.",
]


def hits(*texts: str) -> list[Retrieved]:
    return [Retrieved(Chunk(t, "x.txt", i), 0.9 - i * 0.05) for i, t in enumerate(texts)]


def test_parse_facts_labels_amounts_and_durations() -> None:
    facts = parse_facts(hits(*KB), PRODUCTS)
    assert [a.describe() for a in facts.amounts] == [
        "L1 desk lamp: PKR 2,500",
        "S1 Bluetooth speaker: PKR 4,000",
        "delivery: PKR 200 per order",
    ]
    assert [d.text for d in facts.durations] == [
        "six hours",
        "2–3 working days",
        "3–5 working days",
        "seven days",
        "48 hours",
    ]
    assert facts.durations[2].clause == "Lahore takes 3–5 working days"
    assert facts.block().startswith("FACTS (parsed from the passages):\n- L1 desk lamp: PKR 2,500")


def test_facts_block_empty_without_figures() -> None:
    assert parse_facts(hits("International delivery is unavailable."), PRODUCTS).block() == ""


def test_requested_quantities() -> None:
    assert requested_quantities("two lamps and a speaker", PRODUCTS) == {"L1": 2, "S1": 1}
    assert requested_quantities("3 L1 lamps with delivery", PRODUCTS) == {"L1": 3}
    assert requested_quantities("Does the lamp have a battery?", PRODUCTS) == {"L1": 1}
    assert requested_quantities("How much is delivery?", PRODUCTS) == {}


def facts() -> object:
    return parse_facts(hits(*KB), PRODUCTS)


def test_product_plus_delivery() -> None:
    computed = compute_total(
        "What will the L1 lamp cost with delivery to Islamabad?", PRODUCTS, facts()
    )  # type: ignore[arg-type]
    assert computed is not None
    assert computed.total == 2700
    assert computed.line == "COMPUTED: L1 desk lamp PKR 2,500 + delivery PKR 200 = PKR 2,700."


def test_quantity_and_two_products() -> None:
    computed = compute_total(
        "How much for two lamps and a speaker with shipping?", PRODUCTS, facts()
    )  # type: ignore[arg-type]
    assert computed is not None
    assert computed.total == 2 * 2500 + 4000 + 200
    assert computed.line == (
        "COMPUTED: 2 x L1 desk lamp PKR 2,500 + S1 Bluetooth speaker PKR 4,000"
        " + delivery PKR 200 = PKR 9,200."
    )


def test_quantity_without_delivery() -> None:
    computed = compute_total("What do three speakers cost?", PRODUCTS, facts())  # type: ignore[arg-type]
    assert computed is not None
    assert computed.total == 12000
    assert "delivery" not in computed.line


def test_single_unit_price_needs_no_arithmetic() -> None:
    assert compute_total("How much does the L1 cost?", PRODUCTS, facts()) is None  # type: ignore[arg-type]
    assert compute_total("Does the lamp have a battery?", PRODUCTS, facts()) is None  # type: ignore[arg-type]
    assert compute_total("How much is delivery?", PRODUCTS, facts()) is None  # type: ignore[arg-type]


def test_changed_prices_change_total() -> None:
    """Guards against hardcoded facts: the total follows whatever the passages say."""
    changed = parse_facts(
        hits("The L1 desk lamp costs PKR 3,100.", "Delivery costs PKR 350 per order."), PRODUCTS
    )
    computed = compute_total("L1 price with delivery?", PRODUCTS, changed)
    assert computed is not None and computed.total == 3450
    assert "3,450" in computed.line


def test_missing_figures_mean_no_total() -> None:
    no_price = parse_facts(hits("The L1 desk lamp.", "Delivery costs PKR 200."), PRODUCTS)
    assert compute_total("L1 price with delivery?", PRODUCTS, no_price) is None
    no_fee = parse_facts(hits("The L1 desk lamp costs PKR 2,500."), PRODUCTS)
    assert compute_total("L1 price with delivery?", PRODUCTS, no_fee) is None


def test_currency_mismatch_is_not_summed() -> None:
    mixed = parse_facts(
        hits("The L1 desk lamp costs USD 25.", "Delivery costs PKR 200 per order."), PRODUCTS
    )
    assert compute_total("L1 with delivery?", PRODUCTS, mixed) is None
