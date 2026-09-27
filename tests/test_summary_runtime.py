"""Summary acceptance uses the existing provider retry and persistence paths."""

import asyncio
import json
import logging
from copy import deepcopy
from decimal import Decimal

import pytest
from pydantic import ValidationError

from app.classification.pipeline import ClassificationPipeline
from app.classification.provider import ErrorKind, ProviderError, ProviderResult, TokenUsage
from app.config import Settings
from app.observability import LOGGER_NAME
from app.persistence.database import initialize_database
from app.persistence.models import RequestState
from app.persistence.repository import RequestRepository
from app.services.requests import RequestService

PORTUGUESE = "O checkout está cobrando duas vezes alguns clientes."
SPANISH = "El checkout está cobrando dos veces a algunos clientes."


class ScriptedClassifier:
    model = "test-model"

    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.messages = []

    async def classify(self, *, masked_message, prompt):
        self.messages.append(masked_message)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def provider_result(content):
    return ProviderResult(
        classification=content,
        model="test-model",
        latency_ms=13.5,
        usage=TokenUsage(input_tokens=100, output_tokens=20, cached_input_tokens=10),
    )


def output_with_summary(valid_model_output, summary, *, language="es"):
    return {**deepcopy(valid_model_output), "resumen": summary, "idioma": language}


def run_pipeline(fake, *, attempts=3, raw_message="ana@example.com"):
    delays = []

    async def no_wait(delay):
        delays.append(delay)

    pipeline = ClassificationPipeline(
        fake,
        settings=Settings(
            model="test-model",
            max_attempts=attempts,
            input_price_per_million=Decimal("1"),
            output_price_per_million=Decimal("2"),
            cached_input_price_per_million=Decimal("0.5"),
        ),
        sleep=no_wait,
    )
    return pipeline, delays, raw_message


def test_non_spanish_attempt_retries_with_same_masked_text_and_preserves_usage(
    valid_model_output, caplog
):
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)
    rejected = output_with_summary(valid_model_output, PORTUGUESE, language="pt")
    accepted = output_with_summary(valid_model_output, SPANISH, language="pt")
    fake = ScriptedClassifier([provider_result(rejected), provider_result(accepted)])
    pipeline, delays, raw = run_pipeline(fake, raw_message="ana@example.com +52 55 1234 5678")

    logger = logging.getLogger(LOGGER_NAME)
    logger.addHandler(caplog.handler)
    try:
        result = asyncio.run(
            pipeline.classify_with_metadata(message_id="language-1", raw_message=raw)
        )
    finally:
        logger.removeHandler(caplog.handler)

    assert result.classification.resumen == SPANISH
    assert result.classification.idioma == "pt"
    assert result.metadata.attempt == 2
    assert result.metadata.input_tokens == 100
    assert result.metadata.cached_input_tokens == 10
    assert result.metadata.latency_ms == 13.5
    assert result.metadata.estimated_cost_usd == Decimal("0.000135")
    assert fake.messages == ["[EMAIL] [PHONE]"] * 2
    assert delays == [0.5]
    events = [json.loads(row.message) for row in caplog.records if row.name == LOGGER_NAME]
    assert [(row["attempt"], row["success"]) for row in events] == [(1, False), (2, True)]
    assert events[0]["error_type"] == "invalid_summary_language"
    assert all(row["model"] == "test-model" for row in events)
    assert all(row["latency_ms"] == 13.5 for row in events)
    assert all(row["input_tokens"] == 100 for row in events)
    assert all(row["output_tokens"] == 20 for row in events)
    assert all(row["cached_input_tokens"] == 10 for row in events)
    assert all(row["estimated_cost_usd"] == 0.000135 for row in events)
    assert PORTUGUESE not in caplog.text
    assert raw not in caplog.text
    assert "ana@example.com" not in caplog.text
    assert "[EMAIL]" not in caplog.text


def test_non_spanish_attempts_exhaust_configured_budget(valid_model_output):
    rejected = provider_result(output_with_summary(valid_model_output, PORTUGUESE))
    fake = ScriptedClassifier([rejected] * 4)
    pipeline, delays, _ = run_pipeline(fake, attempts=3)
    with pytest.raises(ProviderError) as failure:
        asyncio.run(pipeline.classify(message_id="language-2", raw_message="ana@example.com"))
    assert failure.value.kind is ErrorKind.INVALID_SUMMARY_LANGUAGE
    assert failure.value.result is rejected
    assert fake.messages == ["[EMAIL]"] * 3
    assert delays == [0.5, 1.0]


def test_transport_and_language_failures_share_one_budget(valid_model_output):
    rejected = provider_result(output_with_summary(valid_model_output, PORTUGUESE))
    fake = ScriptedClassifier(
        [ProviderError(ErrorKind.CONNECTION, retryable=True), rejected, rejected]
    )
    pipeline, delays, _ = run_pipeline(fake, attempts=3)
    with pytest.raises(ProviderError) as failure:
        asyncio.run(pipeline.classify(message_id="language-3", raw_message="ana@example.com"))
    assert failure.value.kind is ErrorKind.INVALID_SUMMARY_LANGUAGE
    assert fake.messages == ["[EMAIL]"] * 3
    assert delays == [0.5, 1.0]


def test_structural_word_limit_precedes_language_policy(valid_model_output):
    invalid = output_with_summary(valid_model_output, " ".join(["palabra"] * 21))
    fake = ScriptedClassifier([provider_result(invalid)])
    pipeline, _, _ = run_pipeline(fake, attempts=1)
    with pytest.raises(ValidationError):
        asyncio.run(pipeline.classify(message_id="language-4", raw_message="Request"))
    assert fake.messages == ["Request"]


def test_language_exhaustion_persists_failed_state_without_reclassification(
    tmp_path, valid_model_output
):
    async def scenario():
        path = tmp_path / "summary-failure.sqlite3"
        await initialize_database(path)
        rejected = provider_result(output_with_summary(valid_model_output, PORTUGUESE))
        fake = ScriptedClassifier([rejected] * 3)
        pipeline, _, _ = run_pipeline(fake)
        service = RequestService(RequestRepository(path), pipeline)
        first = await service.submit(message_id="language-5", raw_message="ana@example.com")
        second = await service.submit(message_id="language-5", raw_message="ana@example.com")
        assert first == second
        assert first.state is RequestState.FAILED
        assert first.failure_code == "invalid_summary_language"
        assert fake.messages == ["[EMAIL]"] * 3

    asyncio.run(scenario())
