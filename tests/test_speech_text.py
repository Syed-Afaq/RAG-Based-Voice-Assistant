from __future__ import annotations

from voice_rag.speech_text import normalize_for_speech


def test_currency_and_ranges() -> None:
    text = "The L1 lamp costs PKR 2,700 with delivery, in 2–3 working days."
    assert normalize_for_speech(text) == (
        "The L 1 lamp costs 2,700 Pakistani rupees with delivery, in 2 to 3 working days."
    )


def test_model_codes_usb_and_markdown() -> None:
    text = "**Answer:** The S1 speaker uses *USB-C* power.\n- item"
    assert normalize_for_speech(text) == "Answer: The S 1 speaker uses USB C power. item"


def test_think_tags_and_whitespace_removed() -> None:
    assert normalize_for_speech("<think>reasoning</think>\n\nNo   battery.") == "No battery."


def test_hyphen_range_and_unknown_currency() -> None:
    assert normalize_for_speech("3-5 days, XYZ 10") == "3 to 5 days, 10 XYZ"
