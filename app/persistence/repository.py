"""Explicit SQLite queries for reservations, results, corrections, and listings."""

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.classification.result import ClassificationResult, SuccessfulCallMetadata
from app.persistence.database import connect_database
from app.persistence.models import (
    IdempotencyConflict,
    InvalidCorrection,
    RequestNotCompleted,
    RequestNotFound,
    RequestRecord,
    RequestState,
)
from app.schemas import Classification, ClassificationContent

CATEGORIES = {"bug", "datos", "acceso", "automatizacion", "consulta", "otro"}
PRIORITIES = {"alta", "media", "baja"}
AREAS = {"backend", "frontend", "data", "devops", "producto", "digital_transformation"}


def utc_timestamp() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds")


def decode_record(row: Any) -> RequestRecord:
    return RequestRecord(
        id=row["id"],
        raw_message=row["raw_message"],
        state=RequestState(row["state"]),
        ai_classification=Classification.model_validate_json(row["ai_classification_json"])
        if row["ai_classification_json"] is not None
        else None,
        human_correction=ClassificationContent.model_validate_json(row["human_correction_json"])
        if row["human_correction_json"] is not None
        else None,
        provider_metadata=SuccessfulCallMetadata.model_validate_json(row["provider_metadata_json"])
        if row["provider_metadata_json"] is not None
        else None,
        effective_category=row["effective_category"],
        effective_priority=row["effective_priority"],
        effective_area=row["effective_area"],
        failure_code=row["failure_code"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        classified_at=row["classified_at"],
        corrected_at=row["corrected_at"],
    )


@dataclass(frozen=True)
class Reservation:
    won: bool
    record: RequestRecord


class RequestRepository:
    def __init__(self, database_path: Path) -> None:
        self.database_path = Path(database_path)

    async def reserve(self, message_id: str, raw_message: str) -> Reservation:
        """A committed primary-key reservation decides who may classify."""
        timestamp = utc_timestamp()
        async with connect_database(self.database_path) as connection:
            await connection.execute("BEGIN IMMEDIATE")
            try:
                cursor = await connection.execute(
                    "INSERT INTO requests (id, raw_message, state, created_at, updated_at) "
                    "VALUES (?, ?, 'processing', ?, ?) ON CONFLICT(id) DO NOTHING",
                    (message_id, raw_message, timestamp, timestamp),
                )
                won = cursor.rowcount == 1
                await cursor.close()
                async with connection.execute(
                    "SELECT * FROM requests WHERE id = ?", (message_id,)
                ) as read_cursor:
                    row = await read_cursor.fetchone()
                await connection.commit()
            except BaseException:
                await connection.rollback()
                raise
        record = decode_record(row)
        if record.raw_message != raw_message:
            raise IdempotencyConflict()
        return Reservation(won=won, record=record)

    async def complete(self, message_id: str, result: ClassificationResult) -> RequestRecord:
        if result.classification.id != message_id:
            raise ValueError("Classification ID differs from the reserved ID.")
        classification = result.classification
        timestamp = utc_timestamp()
        async with connect_database(self.database_path) as connection:
            await connection.execute("BEGIN IMMEDIATE")
            try:
                cursor = await connection.execute(
                    "UPDATE requests SET state = 'completed', ai_classification_json = ?, "
                    "provider_metadata_json = ?, effective_category = ?, effective_priority = ?, "
                    "effective_area = ?, updated_at = ?, classified_at = ? "
                    "WHERE id = ? AND state = 'processing'",
                    (
                        classification.model_dump_json(),
                        result.metadata.model_dump_json(),
                        classification.categoria,
                        classification.prioridad,
                        classification.area_sugerida,
                        timestamp,
                        timestamp,
                        message_id,
                    ),
                )
                updated = cursor.rowcount
                await cursor.close()
                if updated != 1:
                    raise RequestNotCompleted()
                async with connection.execute(
                    "SELECT * FROM requests WHERE id = ?", (message_id,)
                ) as read_cursor:
                    row = await read_cursor.fetchone()
                await connection.commit()
            except BaseException:
                await connection.rollback()
                raise
        return decode_record(row)

    async def mark_failed(self, message_id: str, failure_code: str) -> RequestRecord:
        if not failure_code or any(
            character not in "abcdefghijklmnopqrstuvwxyz_" for character in failure_code
        ):
            raise ValueError("Failure code must be sanitized.")
        timestamp = utc_timestamp()
        async with connect_database(self.database_path) as connection:
            await connection.execute("BEGIN IMMEDIATE")
            try:
                cursor = await connection.execute(
                    "UPDATE requests SET state = 'failed', failure_code = ?, updated_at = ? "
                    "WHERE id = ? AND state = 'processing'",
                    (failure_code, timestamp, message_id),
                )
                updated = cursor.rowcount
                await cursor.close()
                if updated != 1:
                    raise RequestNotCompleted()
                async with connection.execute(
                    "SELECT * FROM requests WHERE id = ?", (message_id,)
                ) as read_cursor:
                    row = await read_cursor.fetchone()
                await connection.commit()
            except BaseException:
                await connection.rollback()
                raise
        return decode_record(row)

    async def get(self, message_id: str) -> RequestRecord | None:
        async with connect_database(self.database_path) as connection:
            async with connection.execute(
                "SELECT * FROM requests WHERE id = ?", (message_id,)
            ) as cursor:
                row = await cursor.fetchone()
        return decode_record(row) if row is not None else None

    async def correct(self, message_id: str, changes: dict[str, object]) -> RequestRecord:
        if not changes or set(changes) - set(ClassificationContent.model_fields):
            raise InvalidCorrection()
        async with connect_database(self.database_path) as connection:
            await connection.execute("BEGIN IMMEDIATE")
            try:
                async with connection.execute(
                    "SELECT * FROM requests WHERE id = ?", (message_id,)
                ) as cursor:
                    row = await cursor.fetchone()
                if row is None:
                    raise RequestNotFound()
                original = decode_record(row)
                if original.state is not RequestState.COMPLETED:
                    raise RequestNotCompleted()
                merged = original.effective_classification.model_dump()
                merged.update(changes)
                corrected = ClassificationContent.model_validate(merged)
                timestamp = utc_timestamp()
                await connection.execute(
                    "UPDATE requests SET human_correction_json = ?, effective_category = ?, "
                    "effective_priority = ?, effective_area = ?, corrected_at = ?, updated_at = ? "
                    "WHERE id = ? AND state = 'completed'",
                    (
                        corrected.model_dump_json(),
                        corrected.categoria,
                        corrected.prioridad,
                        corrected.area_sugerida,
                        timestamp,
                        timestamp,
                        message_id,
                    ),
                )
                async with connection.execute(
                    "SELECT * FROM requests WHERE id = ?", (message_id,)
                ) as cursor:
                    updated_row = await cursor.fetchone()
                await connection.commit()
            except BaseException:
                await connection.rollback()
                raise
        return decode_record(updated_row)

    async def list_completed(
        self,
        *,
        category: str | None = None,
        priority: str | None = None,
        area: str | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> list[RequestRecord]:
        if type(limit) is not int or not 1 <= limit <= 100:
            raise ValueError("Limit must be between 1 and 100.")
        if type(offset) is not int or offset < 0:
            raise ValueError("Offset must be a non-negative integer.")
        for value, allowed in ((category, CATEGORIES), (priority, PRIORITIES), (area, AREAS)):
            if value is not None and value not in allowed:
                raise ValueError("Invalid classification filter.")
        clauses = ["state = 'completed'"]
        parameters: list[object] = []
        for column, value in (
            ("effective_category", category),
            ("effective_priority", priority),
            ("effective_area", area),
        ):
            if value is not None:
                clauses.append(f"{column} = ?")
                parameters.append(value)
        query = (
            "SELECT * FROM requests WHERE "
            + " AND ".join(clauses)
            + " ORDER BY created_at DESC, id DESC LIMIT ? OFFSET ?"
        )
        parameters.extend((limit, offset))
        async with connect_database(self.database_path) as connection:
            async with connection.execute(query, parameters) as cursor:
                rows = await cursor.fetchall()
        return [decode_record(row) for row in rows]
