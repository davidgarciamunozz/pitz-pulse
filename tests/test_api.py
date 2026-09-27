"""HTTP behavior uses a fake classifier and file-backed temporary databases."""

import asyncio
import json
import logging
from contextlib import asynccontextmanager

import pytest
from httpx import ASGITransport, AsyncClient
from pydantic import SecretStr

from app.api.main import create_app
from app.classification.provider import ErrorKind, ProviderError
from app.config import Settings
from app.observability import LOGGER_NAME
from app.persistence.models import RequestState
from app.persistence.repository import RequestRepository
from tests.fakes import RecordingFakeClassifier

AUTH = {"X-API-Key": "test-only-api-key"}


@pytest.fixture(autouse=True)
def restore_model_call_logger():
    logger = logging.getLogger(LOGGER_NAME)
    previous_handlers = logger.handlers[:]
    previous_level = logger.level
    previous_propagate = logger.propagate
    yield
    logger.handlers = previous_handlers
    logger.setLevel(previous_level)
    logger.propagate = previous_propagate


@asynccontextmanager
async def api_client(tmp_path, model_output, *, fake=None, api_key=AUTH["X-API-Key"]):
    classifier = fake or RecordingFakeClassifier(model_output)
    settings = Settings(
        service_api_key=SecretStr(api_key),
        database_path=tmp_path / "api.sqlite3",
        max_attempts=1,
    )
    app = create_app(settings=settings, classifier=classifier)
    async with app.router.lifespan_context(app):
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            yield client, app, classifier, RequestRepository(settings.database_path)


def post(client, message_id="R-1", message="Error en checkout", **kwargs):
    return client.post(
        "/solicitudes", json={"id": message_id, "mensaje": message}, headers=AUTH, **kwargs
    )


def test_health_is_static_and_does_not_require_authentication(tmp_path, valid_model_output):
    async def scenario():
        async with api_client(tmp_path, valid_model_output) as (client, _, fake, _):
            response = await client.get("/health")
            assert response.status_code == 200
            assert response.json() == {"status": "ok"}
            assert fake.calls == []

    asyncio.run(scenario())


def test_post_new_duplicate_and_exact_message_conflict(tmp_path, valid_model_output):
    async def scenario():
        async with api_client(tmp_path, valid_model_output) as (client, _, fake, repository):
            first = await post(client, message="ana@example.com")
            assert first.status_code == 201
            body = first.json()
            assert body["id"] == "R-1" and body["state"] == "completed"
            assert body["ai_classification"]["id"] == "R-1"
            assert body["ai_classification"]["version_prompt"] == "v1"
            assert body["provider_metadata"]["model"] == "test-model"
            assert "mensaje" not in body
            assert "ana@example.com" not in first.text
            assert fake.calls == [("[EMAIL]", "v1")]

            duplicate = await post(client, message="ana@example.com")
            assert duplicate.status_code == 200 and duplicate.json() == body
            conflict = await post(client, message="bea@example.com")
            assert conflict.status_code == 409
            assert "bea@example.com" not in conflict.text
            assert len(fake.calls) == 1
            assert len(await repository.list_completed()) == 1

    asyncio.run(scenario())


def test_post_processing_and_failed_are_explicit_and_never_reclassify(tmp_path, valid_model_output):
    async def scenario():
        async with api_client(tmp_path, valid_model_output) as (client, _, fake, repository):
            await repository.reserve("pending", "Same")
            pending = await post(client, message_id="pending", message="Same")
            assert pending.status_code == 202
            assert pending.json()["state"] == "processing"
            assert pending.headers["Retry-After"] == "1"
            await repository.reserve("failed", "Same")
            await repository.mark_failed("failed", "provider_error")
            failed = await post(client, message_id="failed", message="Same")
            assert failed.status_code == 503
            assert failed.json() == {
                "detail": "Classification unavailable.",
                "id": "failed",
                "state": "failed",
            }
            assert fake.calls == []

    asyncio.run(scenario())


def test_collection_filters_combination_and_pagination(tmp_path, valid_model_output):
    async def scenario():
        async with api_client(tmp_path, valid_model_output) as (client, _, _, repository):
            for message_id in ("A", "B", "C", "D"):
                assert (await post(client, message_id, f"Message {message_id}")).status_code == 201
            await repository.correct(
                "B", {"categoria": "datos", "prioridad": "media", "area_sugerida": "data"}
            )
            await repository.correct(
                "C", {"categoria": "datos", "prioridad": "media", "area_sugerida": "backend"}
            )
            await repository.reserve("processing", "Not completed")
            cases = [
                ({}, ["D", "C", "B", "A"]),
                ({"categoria": "datos"}, ["C", "B"]),
                ({"prioridad": "media"}, ["C", "B"]),
                ({"area": "data"}, ["B"]),
                ({"categoria": "datos", "prioridad": "media", "area": "backend"}, ["C"]),
                ({"limit": 2, "offset": 1}, ["C", "B"]),
            ]
            for query, expected in cases:
                response = await client.get("/solicitudes", params=query, headers=AUTH)
                assert response.status_code == 200
                assert [row["id"] for row in response.json()["items"]] == expected
                assert response.json()["limit"] == query.get("limit", 20)
                assert response.json()["offset"] == query.get("offset", 0)
                assert "raw_message" not in response.text
                assert "mensaje" not in response.text
            assert (await client.get("/solicitudes", params={"offset": 30}, headers=AUTH)).json()[
                "items"
            ] == []

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "query",
    [
        {"categoria": "invalid"},
        {"prioridad": "urgent"},
        {"area": "sales"},
        {"limit": 0},
        {"limit": 101},
        {"offset": -1},
    ],
)
def test_collection_rejects_invalid_filters_and_pagination(tmp_path, valid_model_output, query):
    async def scenario():
        async with api_client(tmp_path, valid_model_output) as (client, _, _, _):
            response = await client.get("/solicitudes", params=query, headers=AUTH)
            assert response.status_code == 422
            assert response.json() == {"detail": "Invalid request."}

    asyncio.run(scenario())


def test_detail_exposes_original_only_to_authenticated_caller(tmp_path, valid_model_output):
    async def scenario():
        async with api_client(tmp_path, valid_model_output) as (client, _, _, _):
            raw = "Mi correo es ana@example.com"
            await post(client, message=raw)
            detail = await client.get("/solicitudes/R-1", headers=AUTH)
            assert detail.status_code == 200
            assert detail.json()["mensaje"] == raw
            assert detail.json()["ai_classification"]["categoria"] == "bug"
            missing = await client.get("/solicitudes/missing", headers=AUTH)
            assert missing.status_code == 404
            assert raw not in missing.text

    asyncio.run(scenario())


def test_patch_merges_partial_corrections_without_changing_original(tmp_path, valid_model_output):
    async def scenario():
        async with api_client(tmp_path, valid_model_output) as (client, _, _, repository):
            await post(client)
            first = await client.patch(
                "/solicitudes/R-1", json={"categoria": "datos", "prioridad": "media"}, headers=AUTH
            )
            assert first.status_code == 200
            second = await client.patch(
                "/solicitudes/R-1", json={"area_sugerida": "data"}, headers=AUTH
            )
            assert second.status_code == 200
            body = second.json()
            assert body["ai_classification"]["categoria"] == "bug"
            assert body["ai_classification"]["confianza"] == 0.85
            assert body["human_correction"]["categoria"] == "datos"
            assert body["effective_classification"]["prioridad"] == "media"
            assert body["effective_classification"]["area_sugerida"] == "data"
            assert body["provider_metadata"] == first.json()["provider_metadata"]
            assert body["corrected_at"] is not None
            assert (await repository.get("R-1")).effective_area == "data"

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "correction",
    [
        {"categoria": "invalid"},
        {"id": "changed"},
        {"confianza": 0.1},
        {"version_prompt": "v2"},
        {"provider_metadata": {"model": "other"}},
        {"requiere_info": False},
        {},
    ],
)
def test_patch_invalid_or_forbidden_is_sanitized_and_atomic(
    tmp_path, valid_model_output, correction
):
    async def scenario():
        async with api_client(tmp_path, valid_model_output) as (client, _, _, repository):
            await post(client)
            before = await repository.get("R-1")
            response = await client.patch("/solicitudes/R-1", json=correction, headers=AUTH)
            assert response.status_code == 422
            assert response.json()["detail"] in {"Invalid request.", "Invalid correction."}
            assert (await repository.get("R-1")) == before

    asyncio.run(scenario())


def test_patch_missing_and_uncompleted(tmp_path, valid_model_output):
    async def scenario():
        async with api_client(tmp_path, valid_model_output) as (client, _, _, repository):
            missing = await client.patch(
                "/solicitudes/absent", json={"categoria": "datos"}, headers=AUTH
            )
            assert missing.status_code == 404
            await repository.reserve("pending", "Message")
            pending = await client.patch(
                "/solicitudes/pending", json={"categoria": "datos"}, headers=AUTH
            )
            assert pending.status_code == 409

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "method,path",
    [
        ("POST", "/solicitudes"),
        ("GET", "/solicitudes"),
        ("GET", "/solicitudes/R-1"),
        ("PATCH", "/solicitudes/R-1"),
    ],
)
def test_authentication_applies_to_every_endpoint(tmp_path, valid_model_output, method, path):
    async def scenario():
        async with api_client(tmp_path, valid_model_output) as (client, _, fake, _):
            for headers in ({}, {"X-API-Key": "wrong"}):
                response = await client.request(
                    method, path, json={} if method != "GET" else None, headers=headers
                )
                assert response.status_code == 401
                assert response.json() == {"detail": "Invalid API key."}
                assert AUTH["X-API-Key"] not in response.text
            assert fake.calls == []

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("configured_key", "header_value", "expected_status"),
    [
        ("ascii-key", b"ascii-key", 200),
        ("ascii-key", b"wrong-key", 401),
        ("clé-privée", bytes("clé-privée", "utf-8"), 200),
        ("clé-privée", bytes("clé-incorrecte", "utf-8"), 401),
        ("ascii-key", bytes("clé-privée", "utf-8"), 401),
        ("ascii-key", b"\xff", 401),
    ],
)
def test_api_key_comparison_accepts_utf8_and_rejects_invalid_values(
    tmp_path, valid_model_output, configured_key, header_value, expected_status, caplog
):
    async def scenario():
        async with api_client(tmp_path, valid_model_output, api_key=configured_key) as (
            client,
            _,
            fake,
            _,
        ):
            response = await client.get("/solicitudes", headers={b"X-API-Key": header_value})
            assert response.status_code == expected_status
            if expected_status == 401:
                assert response.json() == {"detail": "Invalid API key."}
            assert configured_key not in response.text
            assert header_value not in response.content
            assert configured_key not in caplog.text
            assert fake.calls == []

    asyncio.run(scenario())


def test_provider_and_internal_errors_are_sanitized(tmp_path, valid_model_output, caplog):
    async def scenario():
        fake = RecordingFakeClassifier(
            valid_model_output, failure=ProviderError(ErrorKind.SERVER, retryable=False)
        )
        async with api_client(tmp_path, valid_model_output, fake=fake) as (
            client,
            app,
            _,
            repository,
        ):
            provider_failure = await post(client, message="ana@example.com")
            assert provider_failure.status_code == 503
            assert "ana@example.com" not in provider_failure.text
            assert len(fake.calls) == 1
            assert (await repository.get("R-1")).state is RequestState.FAILED
            assert (await post(client, message="ana@example.com")).status_code == 503
            assert len(fake.calls) == 1

            class FailingService:
                async def list_completed(self, **_kwargs):
                    raise RuntimeError("private database detail ana@example.com")

            app.state.service = FailingService()
            internal_failure = await client.get("/solicitudes", headers=AUTH)
            assert internal_failure.status_code == 503
            assert internal_failure.json() == {"detail": "Service unavailable."}
            assert "ana@example.com" not in internal_failure.text
            assert "ana@example.com" not in caplog.text
            assert AUTH["X-API-Key"] not in caplog.text

    asyncio.run(scenario())


def test_invalid_post_body_does_not_echo_sensitive_input(tmp_path, valid_model_output):
    async def scenario():
        async with api_client(tmp_path, valid_model_output) as (client, _, fake, _):
            response = await client.post(
                "/solicitudes",
                json={"id": "R-1", "mensaje": "ana@example.com", "confianza": 0.9},
                headers=AUTH,
            )
            assert response.status_code == 422
            assert response.json() == {"detail": "Invalid request."}
            assert fake.calls == []

    asyncio.run(scenario())


def test_lifespan_initializes_migrations_and_reuses_and_closes_pipeline(
    tmp_path, valid_model_output
):
    class ClosableFake(RecordingFakeClassifier):
        closed = False

        async def aclose(self):
            self.closed = True

    async def scenario():
        fake = ClosableFake(valid_model_output)
        async with api_client(tmp_path, valid_model_output, fake=fake) as (
            client,
            app,
            _,
            repository,
        ):
            assert repository.database_path.exists()
            service = app.state.service
            pipeline = app.state.pipeline
            await post(client, message_id="A")
            await post(client, message_id="B")
            assert app.state.service is service
            assert app.state.pipeline is pipeline
            assert len(fake.calls) == 2
            assert not fake.closed
        assert fake.closed
        assert app.state.service is None and app.state.pipeline is None

    asyncio.run(scenario())


def test_startup_without_service_api_key_fails_closed(tmp_path, valid_model_output):
    async def scenario():
        settings = Settings(database_path=tmp_path / "no-key.sqlite3")
        app = create_app(settings=settings, classifier=RecordingFakeClassifier(valid_model_output))
        with pytest.raises(RuntimeError, match="PITZ_API_KEY"):
            async with app.router.lifespan_context(app):
                pass
        assert not settings.database_path.exists()

    asyncio.run(scenario())


def test_api_startup_enables_content_free_model_call_info_events(
    tmp_path, valid_model_output, monkeypatch, capsys
):
    logger = logging.getLogger(LOGGER_NAME)
    monkeypatch.setattr(logger, "handlers", [])
    monkeypatch.setattr(logger, "level", logging.WARNING)
    monkeypatch.setattr(logger, "propagate", True)
    logger.setLevel(logging.WARNING)
    assert not logger.isEnabledFor(logging.INFO)

    output = {**valid_model_output, "resumen": "Resumen privado para prueba."}

    async def scenario():
        fake = RecordingFakeClassifier(output)
        async with api_client(tmp_path, output, fake=fake) as (client, _, _, _):
            assert logger.isEnabledFor(logging.INFO)
            assert len(logger.handlers) == 1
            assert not logger.propagate
            response = await post(client, message_id="LOG-1", message="Correo ana@example.com")
            assert response.status_code == 201
            assert fake.calls == [("Correo [EMAIL]", "v1")]

    asyncio.run(scenario())

    events = [json.loads(line) for line in capsys.readouterr().err.splitlines()]
    assert len(events) == 1
    assert events[0] == {
        "event": "model_call",
        "message_id": "LOG-1",
        "model": "test-model",
        "prompt_version": "v1",
        "attempt": 1,
        "latency_ms": 0,
        "input_tokens": None,
        "output_tokens": None,
        "cached_input_tokens": None,
        "estimated_cost_usd": None,
        "success": True,
    }
    serialized = json.dumps(events)
    for sensitive in (
        "ana@example.com",
        "Correo [EMAIL]",
        "Resumen privado para prueba.",
        AUTH["X-API-Key"],
    ):
        assert sensitive not in serialized


def test_repeated_api_lifespans_reuse_one_model_call_handler(
    tmp_path, valid_model_output, monkeypatch
):
    logger = logging.getLogger(LOGGER_NAME)
    monkeypatch.setattr(logger, "handlers", [])
    monkeypatch.setattr(logger, "level", logging.WARNING)
    monkeypatch.setattr(logger, "propagate", True)
    logger.setLevel(logging.WARNING)

    async def scenario():
        async with api_client(tmp_path, valid_model_output):
            first_handler = logger.handlers[0]
            assert logger.isEnabledFor(logging.INFO)
            assert len(logger.handlers) == 1
        async with api_client(tmp_path, valid_model_output):
            assert logger.handlers == [first_handler]
            assert logger.isEnabledFor(logging.INFO)

    asyncio.run(scenario())
