"""Idempotent request coordination outside long-lived database transactions."""

import asyncio
from collections.abc import Mapping

from pydantic import ValidationError

from app.classification.pipeline import ClassificationPipeline
from app.classification.provider import ProviderError
from app.persistence.models import InProgress, RequestRecord, RequestState
from app.persistence.repository import RequestRepository


class RequestService:
    def __init__(self, repository: RequestRepository, pipeline: ClassificationPipeline) -> None:
        self._repository = repository
        self._pipeline = pipeline

    async def submit(self, *, message_id: str, raw_message: str) -> RequestRecord | InProgress:
        if not isinstance(message_id, str) or not message_id.strip():
            raise ValueError("Message ID must be a non-empty string.")
        if not isinstance(raw_message, str) or not raw_message.strip():
            raise ValueError("Message must be a non-empty string.")
        reservation = await self._repository.reserve(message_id, raw_message)
        if not reservation.won:
            if reservation.record.state is RequestState.PROCESSING:
                return InProgress(id=message_id, created_at=reservation.record.created_at)
            return reservation.record

        try:
            result = await self._pipeline.classify_with_metadata(
                message_id=message_id, raw_message=raw_message
            )
        except asyncio.CancelledError:
            await self._repository.mark_failed(message_id, "cancelled")
            raise
        except Exception as error:
            if isinstance(error, ProviderError):
                code = error.kind.value
            elif isinstance(error, ValidationError):
                code = "invalid_classification"
            elif isinstance(error, TimeoutError):
                code = "timeout"
            else:
                code = "unexpected_error"
            return await self._repository.mark_failed(message_id, code)
        try:
            return await self._repository.complete(message_id, result)
        except Exception:
            try:
                await self._repository.mark_failed(message_id, "persistence_error")
            except BaseException:
                # Preserve the finalization error even if best-effort cleanup fails.
                pass
            raise

    async def get(self, message_id: str) -> RequestRecord | None:
        return await self._repository.get(message_id)

    async def correct(self, message_id: str, changes: Mapping[str, object]) -> RequestRecord:
        return await self._repository.correct(message_id, dict(changes))

    async def list_completed(
        self,
        *,
        category: str | None = None,
        priority: str | None = None,
        area: str | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> list[RequestRecord]:
        return await self._repository.list_completed(
            category=category, priority=priority, area=area, limit=limit, offset=offset
        )
