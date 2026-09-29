import asyncio
from decimal import Decimal

import pytest
from pydantic import ValidationError

from app.classification.pipeline import ClassificationPipeline
from app.classification.provider import ProviderResult, TokenUsage
from app.classification.result import ClassificationResult
from app.config import Settings
from app.persistence.database import connect_database, initialize_database
from app.persistence.models import (
    IdempotencyConflict,
    InvalidCorrection,
    RequestNotCompleted,
    RequestNotFound,
    RequestState,
)
from app.persistence.repository import RequestRepository
from app.services.requests import RequestService
from tests.fakes import RecordingFakeClassifier


async def prepared_service(path, output):
    await initialize_database(path)
    fake = RecordingFakeClassifier(output)
    repository = RequestRepository(path)
    return RequestService(repository, ClassificationPipeline(fake)), repository, fake


def test_round_trip_and_new_repository_instance(tmp_path, valid_model_output):
    async def scenario():
        path = tmp_path / "roundtrip.sqlite3"
        service, _, fake = await prepared_service(path, valid_model_output)
        original = "Solicitante ana@example.com informa un error."
        created = await service.submit(message_id="R-1", raw_message=original)
        assert created.state is RequestState.COMPLETED
        assert created.raw_message == original
        assert original not in repr(created)
        assert created.ai_classification.id == "R-1"
        assert created.ai_classification.confianza == 0.85
        assert created.provider_metadata.model == "test-model"
        assert created.provider_metadata.attempt == 1
        assert created.provider_metadata.prompt_version == "v3"
        assert created.provider_metadata.input_tokens is None
        assert created.provider_metadata.estimated_cost_usd is None
        assert created.effective_category == "bug"
        assert created.classified_at is not None
        assert created.created_at <= created.classified_at <= created.updated_at
        assert fake.calls == [("Solicitante [EMAIL] informa un error.", "v3")]

        reopened = RequestRepository(path)
        persisted = await reopened.get("R-1")
        assert persisted == created
        assert persisted.effective_classification.categoria == "bug"
        assert await reopened.get("missing") is None

    asyncio.run(scenario())


def test_repository_enforces_exact_raw_message_identity(tmp_path, valid_model_output):
    async def scenario():
        path = tmp_path / "identity.sqlite3"
        service, repository, fake = await prepared_service(path, valid_model_output)
        first = await service.submit(message_id="R-1", raw_message="ana@example.com")
        duplicate = await service.submit(message_id="R-1", raw_message="ana@example.com")
        assert duplicate == first
        assert len(fake.calls) == 1
        with pytest.raises(IdempotencyConflict):
            await service.submit(message_id="R-1", raw_message="bea@example.com")
        assert len(fake.calls) == 1
        assert await repository.get("R-1") == first
        assert len(await repository.list_completed()) == 1

    asyncio.run(scenario())


def test_successful_usage_and_cost_survive_reopening(tmp_path, valid_model_output):
    async def scenario():
        path = tmp_path / "usage.sqlite3"
        await initialize_database(path)

        class MeteredClassifier:
            model = "priced-model"

            async def classify(self, *, masked_message, prompt):
                assert masked_message == "[EMAIL]"
                return ProviderResult(
                    valid_model_output,
                    self.model,
                    27.5,
                    TokenUsage(input_tokens=100, output_tokens=20, cached_input_tokens=10),
                )

        pipeline = ClassificationPipeline(
            MeteredClassifier(),
            settings=Settings(
                model="priced-model",
                input_price_per_million=Decimal("1"),
                output_price_per_million=Decimal("2"),
                cached_input_price_per_million=Decimal("0.5"),
            ),
        )
        service = RequestService(RequestRepository(path), pipeline)
        created = await service.submit(message_id="metered", raw_message="ana@example.com")
        stored = await RequestRepository(path).get("metered")
        assert stored == created
        assert stored.provider_metadata.model == "priced-model"
        assert stored.provider_metadata.attempt == 1
        assert stored.provider_metadata.input_tokens == 100
        assert stored.provider_metadata.output_tokens == 20
        assert stored.provider_metadata.cached_input_tokens == 10
        assert stored.provider_metadata.latency_ms == 27.5
        assert stored.provider_metadata.estimated_cost_usd == Decimal("0.000135")
        assert stored.provider_metadata.prompt_version == stored.ai_classification.version_prompt

    asyncio.run(scenario())


def test_correction_merges_and_preserves_original_and_provider_metadata(
    tmp_path, valid_model_output
):
    async def scenario():
        path = tmp_path / "corrections.sqlite3"
        service, repository, _ = await prepared_service(path, valid_model_output)
        original = await service.submit(message_id="R-1", raw_message="Error en checkout")
        first = await service.correct("R-1", {"categoria": "datos", "prioridad": "media"})
        assert first.human_correction.categoria == "datos"
        assert first.effective_category == "datos" and first.effective_priority == "media"
        assert first.corrected_at is not None
        second = await service.correct("R-1", {"area_sugerida": "data"})
        assert second.effective_classification.categoria == "datos"
        assert second.effective_classification.prioridad == "media"
        assert second.effective_area == "data"
        assert second.ai_classification == original.ai_classification
        assert second.ai_classification.categoria == "bug"
        assert second.ai_classification.confianza == original.ai_classification.confianza
        assert second.provider_metadata == original.provider_metadata
        assert second.created_at == original.created_at
        assert second.classified_at == original.classified_at
        assert second.corrected_at >= first.corrected_at
        assert await repository.get("R-1") == second
        assert await service.submit(message_id="R-1", raw_message="Error en checkout") == second
        with pytest.raises(RequestNotCompleted):
            await repository.complete(
                "R-1",
                ClassificationResult(
                    classification=original.ai_classification,
                    metadata=original.provider_metadata,
                ),
            )

        for forbidden in ("confianza", "id", "version_prompt", "provider_metadata"):
            with pytest.raises(InvalidCorrection):
                await service.correct("R-1", {forbidden: "replacement"})
        with pytest.raises(InvalidCorrection):
            await service.correct("R-1", {})
        with pytest.raises(ValidationError):
            await service.correct("R-1", {"requiere_info": False})
        with pytest.raises(ValidationError):
            await service.correct("R-1", {"prioridad": "urgent"})
        assert await repository.get("R-1") == second

        cleared = await service.correct(
            "R-1", {"requiere_info": False, "pregunta_seguimiento": None}
        )
        assert cleared.human_correction.pregunta_seguimiento is None
        assert cleared.ai_classification.requiere_info is True

    asyncio.run(scenario())


def test_correction_requires_completed_existing_request(tmp_path, valid_model_output):
    async def scenario():
        path = tmp_path / "states.sqlite3"
        service, repository, _ = await prepared_service(path, valid_model_output)
        with pytest.raises(RequestNotFound):
            await service.correct("missing", {"categoria": "datos"})
        await repository.reserve("pending", "Un mensaje")
        with pytest.raises(RequestNotCompleted):
            await service.correct("pending", {"categoria": "datos"})
        assert (await repository.get("pending")).human_correction is None

    asyncio.run(scenario())


def test_effective_filters_combination_and_stable_pagination(tmp_path, valid_model_output):
    async def scenario():
        path = tmp_path / "listing.sqlite3"
        service, repository, _ = await prepared_service(path, valid_model_output)
        for message_id in ("A", "B", "C", "D"):
            await service.submit(message_id=message_id, raw_message=f"Request {message_id}")
        await service.correct(
            "B", {"categoria": "datos", "prioridad": "media", "area_sugerida": "data"}
        )
        await service.correct(
            "C", {"categoria": "datos", "prioridad": "media", "area_sugerida": "backend"}
        )
        async with connect_database(path) as connection:
            await connection.execute(
                "UPDATE requests SET created_at = ? WHERE id IN ('A', 'B', 'C', 'D')",
                ("2026-09-26T00:00:00.000000+00:00",),
            )
            await connection.commit()
        await repository.reserve("pending", "Still processing")
        assert [row.id for row in await service.list_completed()] == ["D", "C", "B", "A"]
        assert [row.id for row in await service.list_completed(category="datos")] == ["C", "B"]
        assert [row.id for row in await service.list_completed(priority="media")] == ["C", "B"]
        assert [row.id for row in await service.list_completed(area="data")] == ["B"]
        assert [
            row.id
            for row in await service.list_completed(
                category="datos", priority="media", area="backend"
            )
        ] == ["C"]
        assert [row.id for row in await service.list_completed(limit=2, offset=1)] == ["C", "B"]
        assert await service.list_completed(offset=20) == []
        for limit in (0, 101, -1, True):
            with pytest.raises(ValueError):
                await service.list_completed(limit=limit)
        for offset in (-1, True):
            with pytest.raises(ValueError):
                await service.list_completed(offset=offset)
        with pytest.raises(ValueError):
            await service.list_completed(category="invalid")

    asyncio.run(scenario())
