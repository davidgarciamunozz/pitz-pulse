import json
from pathlib import Path

import pytest

from app.masking import mask_sensitive_data


@pytest.mark.parametrize(
    ("value", "placeholder"),
    [
        ("11.222.333/0001-81", "[CNPJ]"),
        ("11222333000181", "[CNPJ]"),
        ("GODE561231GR8", "[RFC]"),
        ("gode561231gr8", "[RFC]"),
        ("ABC-010203-1A2", "[RFC]"),
        ("ana.silva+work@example.com", "[EMAIL]"),
        ("+55 (11) 91234-5678", "[PHONE]"),
        ("+55 11 9 1234-5678", "[PHONE]"),
        ("+52 55 1234 5678", "[PHONE]"),
        ("+52 222 123 4567", "[PHONE]"),
        ("+52 (442) 123 4567", "[PHONE]"),
        ("+52 1 55 1234 5678", "[PHONE]"),
        ("(11) 91234-5678", "[PHONE]"),
        ("11 912345678", "[PHONE]"),
        ("55 1234 5678", "[PHONE]"),
        ("Tel: 222-123-4567", "Tel: [PHONE]"),
        ("222 123 4567", "[PHONE]"),
        ("(442) 123-4567", "[PHONE]"),
        ("11912345678", "[PHONE]"),
    ],
)
def test_required_identifier_variants_are_masked(value, placeholder):
    prefix = "Please contact " if value != "11912345678" else "Telefone: "
    assert mask_sensitive_data(f"{prefix}{value} about the request.") == (
        f"{prefix}{placeholder} about the request."
    )


def test_multiple_identifiers_and_surrounding_text_are_preserved():
    message = (
        "Seller 11.222.333/0001-81, RFC GODE561231GR8, email ana@example.com, "
        "phone +52 55 1234 5678; order ORD-12345 remains open."
    )
    assert mask_sensitive_data(message) == (
        "Seller [CNPJ], RFC [RFC], email [EMAIL], phone [PHONE]; order ORD-12345 remains open."
    )


@pytest.mark.parametrize(
    "message",
    [
        "Order ORD-1234567890 is delayed.",
        "Order ORD-11912345678 is delayed.",
        "Reference 12345678901234 is not a valid CNPJ.",
        "Reference 12345678901 is not a mobile number.",
        "Order ORD-55-1234-5678 is pending.",
        "The date is 2026-09-26 and the time is 12:00.",
        "Price is $1,234.56 and quantity is 42.",
        "The campaign code ABC123XYZ is ordinary text.",
    ],
)
def test_unrelated_numbers_dates_prices_and_references_are_preserved(message):
    assert mask_sensitive_data(message) == message


def test_formatted_cnpj_is_masked_even_if_its_check_digits_are_mistyped():
    assert mask_sensitive_data("CNPJ 11.222.333/0001-80") == "CNPJ [CNPJ]"


def test_explicitly_labelled_bare_cnpj_is_masked_even_with_invalid_check_digits():
    assert mask_sensitive_data("CNPJ: 12345678901234") == "CNPJ: [CNPJ]"


def test_bare_phone_requires_context_to_avoid_masking_an_order_id():
    assert mask_sensitive_data("Telefone: 11912345678; pedido 11912345678") == (
        "Telefone: [PHONE]; pedido 11912345678"
    )


def test_reference_context_does_not_match_the_suffix_of_madrid():
    assert mask_sensitive_data("Madrid 55 1234 5678") == "Madrid [PHONE]"


def test_phone_context_does_not_match_the_suffix_of_an_ordinary_word():
    assert mask_sensitive_data("microphone 5512345678") == "microphone 5512345678"


def test_all_annex_a_messages_remain_unchanged():
    messages_path = Path(__file__).resolve().parents[1] / "evaluation/messages.json"
    messages = json.loads(messages_path.read_text(encoding="utf-8"))

    assert len(messages) == 12
    for item in messages:
        assert mask_sensitive_data(item["message"]) == item["message"], item["id"]
