import asyncio
import json
import logging
from copy import deepcopy

import pytest
from pydantic import ValidationError

from app.classification.pipeline import ClassificationPipeline
from app.classification.provider import ErrorKind, ProviderError, ProviderResult, TokenUsage
from app.config import Settings
from app.observability import LOGGER_NAME
from app.schemas import ProviderClassification


class ScriptedClassifier:
    model = "test-model"

    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = []

    async def classify(self, *, masked_message, prompt):
        self.calls.append((masked_message, prompt))
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def result(content):
    return ProviderResult(content, "test-model", 12.5, TokenUsage(100, 20))


def execute(classifier, *, settings=None, message="ana@example.com"):
    delays = []

    async def sleep(delay):
        delays.append(delay)

    pipeline = ClassificationPipeline(classifier, settings=settings, sleep=sleep)
    return asyncio.run(pipeline.classify(message_id="test-1", raw_message=message)), delays


@pytest.mark.parametrize(
    "field,value",
    [
        ("id", "model-owned-id"),
        ("version_prompt", "model-owned-version"),
        ("categoria", "invented"),
        ("confianza", -0.1),
        ("confianza", 1.1),
        ("confianza", float("nan")),
        ("resumen", " ".join(["palabra"] * 21)),
        ("pregunta_seguimiento", None),
    ],
)
def test_entire_provider_contract_is_validated_and_retried(valid_model_output, field, value):
    invalid = {**valid_model_output, field: value}
    with pytest.raises(ValidationError):
        ProviderClassification.model_validate(invalid)
    fake = ScriptedClassifier([result(invalid), result(valid_model_output)])
    classification, delays = execute(fake)
    assert classification.categoria == "bug"
    assert delays == [0.5]
    assert len(fake.calls) == 2
    assert all(call[0] == "[EMAIL]" for call in fake.calls)


@pytest.mark.parametrize(
    "failure",
    [
        ProviderError(ErrorKind.CONNECTION, retryable=True),
        ProviderError(ErrorKind.SERVER, retryable=True),
        ProviderError(ErrorKind.RATE_LIMIT, retryable=True),
        ProviderError(ErrorKind.TIMEOUT, retryable=True),
        TimeoutError("transport timed out"),
    ],
)
def test_transient_failures_retry_and_success_returns(valid_model_output, failure):
    fake = ScriptedClassifier([failure, result(valid_model_output)])
    classification, delays = execute(fake)
    assert classification.id == "test-1"
    assert delays == [0.5]
    assert len(fake.calls) == 2


def test_invalid_json_retries(valid_model_output):
    fake = ScriptedClassifier([result("{broken"), result(json.dumps(valid_model_output))])
    classification, delays = execute(fake)
    assert classification.confianza == 0.85
    assert delays == [0.5]


def test_retry_budget_and_capped_exponential_backoff():
    fake = ScriptedClassifier([ProviderError(ErrorKind.RATE_LIMIT, retryable=True)] * 4)
    delays = []

    async def sleep(delay):
        delays.append(delay)

    pipeline = ClassificationPipeline(
        fake,
        settings=Settings(max_attempts=4, backoff_base_seconds=2, backoff_max_seconds=3),
        sleep=sleep,
    )
    with pytest.raises(ProviderError):
        asyncio.run(pipeline.classify(message_id="test", raw_message="Request"))
    assert len(fake.calls) == 4
    assert delays == [2, 3, 3]


def test_exhausted_validation_preserves_failure_without_extra_calls():
    fake = ScriptedClassifier([result({})] * 3)

    async def no_wait(delay):
        pass

    with pytest.raises(ValidationError):
        asyncio.run(
            ClassificationPipeline(fake, sleep=no_wait).classify(message_id="x", raw_message="x")
        )
    assert len(fake.calls) == 3


@pytest.mark.parametrize(
    "kind", [ErrorKind.AUTHENTICATION, ErrorKind.CONFIGURATION, ErrorKind.REFUSAL]
)
def test_permanent_errors_are_not_retried(kind):
    fake = ScriptedClassifier([ProviderError(kind, retryable=False)])
    with pytest.raises(ProviderError):
        execute(fake)
    assert len(fake.calls) == 1


def test_usage_latency_cost_and_invalid_attempts_are_logged_privately(valid_model_output, caplog):
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)
    raw = "ana@example.com +52 55 1234 5678"
    invalid = deepcopy(valid_model_output)
    invalid["resumen"] = " ".join(["invalid"] * 21)
    fake = ScriptedClassifier([result(invalid), result(valid_model_output)])
    settings = Settings(
        model="test-model", input_price_per_million="1", output_price_per_million="2"
    )
    execute(fake, settings=settings, message=raw)
    events = [json.loads(record.message) for record in caplog.records if record.name == LOGGER_NAME]
    assert len(events) == 2
    assert [event["success"] for event in events] == [False, True]
    assert [event["attempt"] for event in events] == [1, 2]
    for event in events:
        assert event["message_id"] == "test-1"
        assert event["latency_ms"] == 12.5
        assert event["input_tokens"] == 100 and event["output_tokens"] == 20
        assert event["estimated_cost_usd"] == 0.00014
        assert event["prompt_version"] == "v1"
    assert events[0]["error_type"] == "invalid_classification"
    assert raw not in caplog.text and "ana@example.com" not in caplog.text
    assert "+52 55 1234 5678" not in caplog.text
    assert "invalid invalid" not in caplog.text


def test_unknown_pricing_and_usage_are_logged_as_null(valid_model_output, caplog):
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)
    execute(ScriptedClassifier([ProviderResult(valid_model_output, "unknown", 1)]))
    event = json.loads(caplog.records[-1].message)
    assert event["input_tokens"] is None and event["estimated_cost_usd"] is None


def test_exception_text_is_not_logged(caplog):
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)
    fake = ScriptedClassifier([RuntimeError("ana@example.com test-secret")])
    with pytest.raises(RuntimeError):
        execute(fake)
    assert "ana@example.com" not in caplog.text and "test-secret" not in caplog.text
    assert json.loads(caplog.records[-1].message)["error_type"] == "unexpected_error"


def test_concurrency_limit_with_event_gates(valid_model_output):
    async def scenario():
        gate = asyncio.Event()
        all_started = asyncio.Event()
        started = 0

        class GatedClassifier:
            model = "test-model"
            active = 0
            maximum = 0
            entered = 0

            async def classify(self, **kwargs):
                self.entered += 1
                self.active += 1
                self.maximum = max(self.maximum, self.active)
                try:
                    await gate.wait()
                    return result(valid_model_output)
                finally:
                    self.active -= 1

        fake = GatedClassifier()
        pipeline = ClassificationPipeline(fake, settings=Settings(concurrency_limit=2))

        async def classify(index):
            nonlocal started
            started += 1
            if started == 6:
                all_started.set()
            return await pipeline.classify(message_id=str(index), raw_message="Request")

        tasks = [asyncio.create_task(classify(index)) for index in range(6)]
        await all_started.wait()
        assert fake.entered == fake.active == 2
        gate.set()
        classifications = await asyncio.gather(*tasks)
        assert len(classifications) == fake.entered == 6
        assert fake.maximum == 2 and fake.active == 0

    asyncio.run(scenario())


def test_cancelling_active_classification_releases_permit(valid_model_output, caplog):
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)

    async def scenario():
        entered = asyncio.Event()

        class BlockingClassifier:
            model = "test-model"
            calls = 0

            async def classify(self, **kwargs):
                self.calls += 1
                if self.calls == 1:
                    entered.set()
                    await asyncio.Event().wait()
                return result(valid_model_output)

        fake = BlockingClassifier()
        pipeline = ClassificationPipeline(fake, settings=Settings(concurrency_limit=1))
        first = asyncio.create_task(
            pipeline.classify(message_id="cancelled-request", raw_message="ana@example.com")
        )
        await entered.wait()
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        second = await asyncio.wait_for(
            pipeline.classify(message_id="next-request", raw_message="second request"),
            timeout=1,
        )
        assert second.id == "next-request"
        assert fake.calls == 2

    asyncio.run(scenario())
    events = [json.loads(record.message) for record in caplog.records if record.name == LOGGER_NAME]
    assert [
        (event["message_id"], event["error_type"] if not event["success"] else None)
        for event in events
    ] == [("cancelled-request", "cancelled"), ("next-request", None)]
    assert "ana@example.com" not in caplog.text


def test_pipeline_deadline_cancels_attempt_then_retries(valid_model_output, monkeypatch):
    original_timeout = asyncio.timeout
    contexts = []

    def expire_first_attempt(seconds):
        contexts.append(seconds)
        return original_timeout(0 if len(contexts) == 1 else seconds)

    monkeypatch.setattr(asyncio, "timeout", expire_first_attempt)

    class HangingOnce:
        model = "test-model"
        calls = 0
        cancelled = False

        async def classify(self, **kwargs):
            self.calls += 1
            if self.calls == 1:
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    self.cancelled = True
                    raise
            return result(valid_model_output)

    fake = HangingOnce()
    classification, delays = execute(fake, settings=Settings(concurrency_limit=1))
    assert classification.id == "test-1"
    assert fake.cancelled and fake.calls == 2
    assert contexts == [15, 15] and delays == [0.5]
