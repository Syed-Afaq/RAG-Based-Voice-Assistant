from __future__ import annotations

from voice_rag.numbers import extract_numbers, format_amount, parse_number


def test_parse_number_handles_commas_and_decimals() -> None:
    assert parse_number("2,500") == 2500
    assert parse_number("2.5") == 2.5


def test_extract_digits_words_and_ranges() -> None:
    text = "PKR 2,700 within 2–3 working days, six hours, forty-eight hours, seven days"
    assert extract_numbers(text) == {2700, 2, 3, 6, 48, 7}


def test_model_codes_are_not_numbers() -> None:
    assert extract_numbers("The L1 lamp and the S1 speaker use USB-C.") == set()


def test_format_amount() -> None:
    assert format_amount(2700) == "2,700"
    assert format_amount(10.5) == "10.5"
