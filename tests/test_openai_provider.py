import asyncio
import json
import socket

import httpx
import pytest
from openai import AsyncOpenAI

from app.classification.openai_provider import OpenAIClassifier
from app.classification.pipeline import ClassificationPipeline
from app.classification.prompts import load_prompt
from app.classification.provider import ErrorKind, ProviderError, TokenUsage
from app.config import Settings


def test_network_connections_are_blocked_by_suite_fixture():
    with socket.socket() as connection:
        with pytest.raises(pytest.fail.Exception, match="Network access is forbidden"):
            connection.connect(("127.0.0.1", 1))


def response_body(content, *, status="completed", refusal=False, usage=True):
    part = (
        {"type": "refusal", "refusal": "Request refused."}
        if refusal
        else {"type": "output_text", "text": json.dumps(content), "annotations": []}
    )
    return {
        "id": "response-test",
        "object": "response",
        "created_at": 0,
        "model": "test-model",
        "status": status,
        "output": [
            {
                "type": "message",
                "id": "message-test",
                "role": "assistant",
                "status": "completed",
                "content": [part],
            }
        ],
        "usage": {
            "input_tokens": 100,
            "output_tokens": 20,
            "total_tokens": 120,
            "input_tokens_details": {"cached_tokens": 10},
            "output_tokens_details": {"reasoning_tokens": 0},
        }
        if usage
        else None,
    }


def run_with_transport(handler, operation):
    async def scenario():
        async with AsyncOpenAI(
            api_key="test-key-not-real",
            max_retries=5,
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        ) as client:
            return await operation(OpenAIClassifier(Settings(model="test-model"), client=client))

    return asyncio.run(scenario())


def test_sdk_request_uses_schema_temperature_prompt_and_masked_input(valid_model_output):
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(200, json=response_body(valid_model_output))

    async def operation(provider):
        return await ClassificationPipeline(provider).classify(
            message_id="APPLICATION-ID", raw_message="Email ana@example.com; phone +52 55 1234 5678"
        )

    classification = run_with_transport(handler, operation)
    assert classification.id == "APPLICATION-ID" and classification.version_prompt == "v3"
    assert len(requests) == 1
    request = requests[0]
    body = json.loads(request.content)
    assert request.url.path == "/v1/responses"
    assert body["model"] == "test-model" and body["temperature"] == 0
    assert body["store"] is False
    assert body["instructions"] == load_prompt().content
    assert body["input"] == [{"role": "user", "content": "Email [EMAIL]; phone [PHONE]"}]
    output_format = body["text"]["format"]
    assert output_format["type"] == "json_schema" and output_format["strict"] is True
    schema = output_format["schema"]
    assert schema["additionalProperties"] is False
    assert set(schema["properties"]) == set(valid_model_output)
    assert set(schema["required"]) == set(valid_model_output)
    assert "APPLICATION-ID" not in request.content.decode()
    assert "ana@example.com" not in request.content.decode()
    assert request.extensions["timeout"]["read"] == 15


def test_adapter_preserves_usage_model_and_latency(valid_model_output, monkeypatch):
    times = iter([10, 10.0125])
    monkeypatch.setattr("app.classification.openai_provider.perf_counter", lambda: next(times))

    async def operation(provider):
        return await provider.classify(masked_message="[EMAIL]", prompt=load_prompt())

    result = run_with_transport(
        lambda request: httpx.Response(200, json=response_body(valid_model_output)), operation
    )
    assert result.model == "test-model"
    assert result.usage == TokenUsage(100, 20, 10)
    assert result.latency_ms == pytest.approx(12.5)
    assert json.loads(result.classification) == valid_model_output


def test_usage_can_be_unavailable(valid_model_output):
    async def operation(provider):
        return await provider.classify(masked_message="request", prompt=load_prompt())

    result = run_with_transport(
        lambda request: httpx.Response(200, json=response_body(valid_model_output, usage=False)),
        operation,
    )
    assert result.usage is None


@pytest.mark.parametrize(
    "status,kind,retryable",
    [
        (400, ErrorKind.CONFIGURATION, False),
        (401, ErrorKind.AUTHENTICATION, False),
        (403, ErrorKind.AUTHENTICATION, False),
        (404, ErrorKind.CONFIGURATION, False),
        (422, ErrorKind.CONFIGURATION, False),
        (408, ErrorKind.SERVER, True),
        (409, ErrorKind.SERVER, True),
        (429, ErrorKind.RATE_LIMIT, True),
        (500, ErrorKind.SERVER, True),
        (503, ErrorKind.SERVER, True),
    ],
)
def test_sdk_failures_are_sanitized_and_sdk_retries_are_disabled(status, kind, retryable):
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(status, json={"error": {"message": "sensitive error payload"}})

    async def operation(provider):
        return await provider.classify(masked_message="[EMAIL]", prompt=load_prompt())

    with pytest.raises(ProviderError) as failure:
        run_with_transport(handler, operation)
    assert failure.value.kind == kind and failure.value.retryable == retryable
    assert "sensitive error payload" not in str(failure.value)
    assert len(requests) == 1


@pytest.mark.parametrize(
    "error,kind",
    [(httpx.ReadTimeout, ErrorKind.TIMEOUT), (httpx.ConnectError, ErrorKind.CONNECTION)],
)
def test_sdk_transport_failures_are_retryable(error, kind):
    def handler(request):
        raise error("transport failed", request=request)

    async def operation(provider):
        return await provider.classify(masked_message="[EMAIL]", prompt=load_prompt())

    with pytest.raises(ProviderError) as failure:
        run_with_transport(handler, operation)
    assert failure.value.kind == kind and failure.value.retryable


@pytest.mark.parametrize(
    "status,refusal,kind,retryable",
    [
        ("completed", True, ErrorKind.REFUSAL, False),
        ("incomplete", False, ErrorKind.INVALID_RESPONSE, True),
    ],
)
def test_unsuccessful_responses_retain_usage(valid_model_output, status, refusal, kind, retryable):
    async def operation(provider):
        return await provider.classify(masked_message="[EMAIL]", prompt=load_prompt())

    with pytest.raises(ProviderError) as failure:
        run_with_transport(
            lambda request: httpx.Response(
                200, json=response_body(valid_model_output, status=status, refusal=refusal)
            ),
            operation,
        )
    assert failure.value.kind == kind and failure.value.retryable == retryable
    assert failure.value.result.usage == TokenUsage(100, 20, 10)


def test_incomplete_response_without_output_is_retryable(valid_model_output):
    body = response_body(valid_model_output, status="incomplete")
    body["output"] = []

    async def operation(provider):
        return await provider.classify(masked_message="[EMAIL]", prompt=load_prompt())

    with pytest.raises(ProviderError) as failure:
        run_with_transport(lambda request: httpx.Response(200, json=body), operation)
    assert failure.value.kind == ErrorKind.INVALID_RESPONSE
    assert failure.value.retryable
    assert failure.value.result.classification == ""
    assert failure.value.result.usage == TokenUsage(100, 20, 10)


def test_missing_key_only_fails_when_real_provider_is_used():
    provider = OpenAIClassifier(Settings.from_env({}))
    with pytest.raises(ProviderError) as failure:
        asyncio.run(provider.classify(masked_message="[EMAIL]", prompt=load_prompt()))
    assert failure.value.kind == ErrorKind.CONFIGURATION
    assert not failure.value.retryable


def test_environment_factory_loads_prompt_without_creating_a_client(monkeypatch):
    monkeypatch.setenv("OPENAI_MODEL", "configured-model")
    monkeypatch.setattr("app.classification.pipeline.configure_json_logging", lambda: None)
    pipeline = ClassificationPipeline.from_env()
    assert pipeline._classifier.model == "configured-model"
    assert pipeline._classifier._client is None
    asyncio.run(pipeline.aclose())


def test_owned_client_is_created_lazily_and_closed(valid_model_output, monkeypatch):
    clients = []

    def create_client(**kwargs):
        assert kwargs["max_retries"] == 0
        assert kwargs["timeout"] == 15
        client = AsyncOpenAI(
            **kwargs,
            http_client=httpx.AsyncClient(
                transport=httpx.MockTransport(
                    lambda request: httpx.Response(200, json=response_body(valid_model_output))
                )
            ),
        )
        clients.append(client)
        return client

    monkeypatch.setattr("app.classification.openai_provider.AsyncOpenAI", create_client)

    async def scenario():
        settings = Settings(api_key="test-key-not-real", model="test-model")
        provider = OpenAIClassifier(settings)
        pipeline = ClassificationPipeline(provider, settings=settings)
        assert not clients
        try:
            await pipeline.classify(message_id="request-1", raw_message="Un error")
            await pipeline.classify(message_id="request-2", raw_message="Otro error")
            assert len(clients) == 1
            assert not clients[0].is_closed()
        finally:
            await pipeline.aclose()
        assert clients[0].is_closed()
        assert provider._client is None

    asyncio.run(scenario())


def test_injected_client_lifecycle_remains_caller_owned():
    async def operation(provider):
        await provider.aclose()
        assert not provider._client.is_closed()

    run_with_transport(lambda request: pytest.fail("No request expected"), operation)
