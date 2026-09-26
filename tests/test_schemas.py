import pytest
from pydantic import ValidationError

from app.schemas import Classification


@pytest.fixture
def valid_payload():
    return {
        "id": "TEST-01",
        "categoria": "bug",
        "prioridad": "alta",
        "area_sugerida": "backend",
        "idioma": "es",
        "resumen": "El servicio de pagos devuelve un error.",
        "requiere_info": True,
        "pregunta_seguimiento": "¿Qué pedido está afectado?",
        "confianza": 0.85,
        "version_prompt": "test-v1",
    }


def test_valid_classification_preserves_external_contract(valid_payload):
    classification = Classification.model_validate(valid_payload)
    assert classification.model_dump() == valid_payload
    assert Classification.model_validate_json(classification.model_dump_json()) == classification


@pytest.mark.parametrize("field", ["categoria", "prioridad", "area_sugerida", "idioma"])
def test_invalid_enum_is_rejected(valid_payload, field):
    valid_payload[field] = "unsupported"
    with pytest.raises(ValidationError) as error:
        Classification.model_validate(valid_payload)
    assert error.value.errors()[0]["loc"] == (field,)


@pytest.mark.parametrize("field", list(Classification.model_fields))
def test_every_field_is_required(valid_payload, field):
    del valid_payload[field]
    with pytest.raises(ValidationError) as error:
        Classification.model_validate(valid_payload)
    assert error.value.errors()[0]["type"] == "missing"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("id", 123),
        ("categoria", 1),
        ("prioridad", []),
        ("area_sugerida", {}),
        ("idioma", True),
        ("resumen", 123),
        ("requiere_info", "true"),
        ("requiere_info", 1),
        ("pregunta_seguimiento", 123),
        ("confianza", "0.85"),
        ("confianza", True),
        ("version_prompt", 1),
    ],
)
def test_incorrect_types_are_not_coerced(valid_payload, field, value):
    valid_payload[field] = value
    with pytest.raises(ValidationError):
        Classification.model_validate(valid_payload)


@pytest.mark.parametrize("confidence", [-0.01, 1.01, float("nan"), float("inf"), -float("inf")])
def test_invalid_confidence_is_rejected(valid_payload, confidence):
    valid_payload["confianza"] = confidence
    with pytest.raises(ValidationError):
        Classification.model_validate(valid_payload)


@pytest.mark.parametrize("confidence", [0, 1, 0.0, 1.0])
def test_confidence_boundaries_are_valid_numbers(valid_payload, confidence):
    valid_payload["confianza"] = confidence
    assert Classification.model_validate(valid_payload).confianza == confidence


def test_summary_longer_than_twenty_words_is_rejected(valid_payload):
    valid_payload["resumen"] = " ".join(["palabra"] * 21)
    with pytest.raises(ValidationError, match="between 1 and 20 words"):
        Classification.model_validate(valid_payload)


def test_twenty_word_summary_preserves_whitespace(valid_payload):
    summary = " \n" + "\t  ".join(["palabra"] * 20) + " "
    valid_payload["resumen"] = summary
    assert Classification.model_validate(valid_payload).resumen == summary


@pytest.mark.parametrize("field", ["id", "version_prompt", "resumen"])
@pytest.mark.parametrize("value", ["", " \n\t"])
def test_blank_required_text_is_rejected(valid_payload, field, value):
    valid_payload[field] = value
    with pytest.raises(ValidationError):
        Classification.model_validate(valid_payload)


@pytest.mark.parametrize(
    ("requires_info", "question"),
    [(True, None), (True, ""), (True, " \n\t"), (False, "¿Qué pasó?"), (False, "")],
)
def test_inconsistent_follow_up_is_rejected(valid_payload, requires_info, question):
    valid_payload["requiere_info"] = requires_info
    valid_payload["pregunta_seguimiento"] = question
    with pytest.raises(ValidationError, match="follow-up|Follow-up"):
        Classification.model_validate(valid_payload)


def test_no_missing_information_requires_explicit_null(valid_payload):
    valid_payload["requiere_info"] = False
    valid_payload["pregunta_seguimiento"] = None
    assert Classification.model_validate(valid_payload).pregunta_seguimiento is None


def test_unknown_fields_are_rejected(valid_payload):
    valid_payload["justification"] = "Reference-only metadata must stay outside the contract."
    with pytest.raises(ValidationError) as error:
        Classification.model_validate(valid_payload)
    assert error.value.errors()[0]["type"] == "extra_forbidden"
