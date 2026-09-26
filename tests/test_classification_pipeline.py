import asyncio

import pytest
from pydantic import ValidationError

from app.classification.pipeline import ClassificationPipeline
from tests.fakes import RecordingFakeClassifier


@pytest.fixture
def provider_response():
    return {
        "categoria": "bug",
        "prioridad": "alta",
        "area_sugerida": "backend",
        "idioma": "es",
        "resumen": "Un cliente reporta un problema con la solicitud.",
        "requiere_info": True,
        "pregunta_seguimiento": "¿Qué operación está afectada?",
        "confianza": 0.8,
    }


def test_raw_identifiers_never_reach_the_recording_classifier(provider_response):
    fake = RecordingFakeClassifier(provider_response)
    raw_message = (
        "Cliente 11.222.333/0001-81 / GODE561231GR8: ana@example.com, +55 (11) 91234-5678."
    )

    result = asyncio.run(
        ClassificationPipeline(fake).classify(
            message_id="MSG-PRIVACY", raw_message=raw_message, prompt_version="v1"
        )
    )

    assert fake.calls == [("Cliente [CNPJ] / [RFC]: [EMAIL], [PHONE].", "v1")]
    assert result.id == "MSG-PRIVACY"
    assert result.version_prompt == "v1"
    assert result.categoria == "bug"
    for sensitive_value in (
        "11.222.333/0001-81",
        "GODE561231GR8",
        "ana@example.com",
        "+55 (11) 91234-5678",
    ):
        assert sensitive_value not in fake.calls[0][0]


def test_fake_returns_deterministic_classification_without_a_network_client(provider_response):
    fake = RecordingFakeClassifier(provider_response)
    pipeline = ClassificationPipeline(fake)
    first = asyncio.run(
        pipeline.classify(message_id="MSG-1", raw_message="Request A", prompt_version="v1")
    )
    second = asyncio.run(
        pipeline.classify(message_id="MSG-2", raw_message="Request B", prompt_version="v1")
    )

    assert first.categoria == second.categoria == "bug"
    assert first.id == "MSG-1"
    assert second.id == "MSG-2"
    assert fake.calls == [("Request A", "v1"), ("Request B", "v1")]


def test_invalid_provider_result_is_rejected_after_masking(provider_response):
    provider_response["categoria"] = "invented"
    fake = RecordingFakeClassifier(provider_response)
    with pytest.raises(ValidationError):
        asyncio.run(
            ClassificationPipeline(fake).classify(
                message_id="MSG-1", raw_message="ana@example.com", prompt_version="v1"
            )
        )
    assert fake.calls == [("[EMAIL]", "v1")]


def test_provider_failure_propagates_without_exposing_raw_message(provider_response):
    fake = RecordingFakeClassifier(provider_response, failure=RuntimeError("provider unavailable"))
    with pytest.raises(RuntimeError, match="provider unavailable"):
        asyncio.run(
            ClassificationPipeline(fake).classify(
                message_id="MSG-1", raw_message="ana@example.com", prompt_version="v1"
            )
        )
    assert fake.calls == [("[EMAIL]", "v1")]
