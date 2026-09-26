"""Explicit connections and transactional numbered SQL migrations."""

import sqlite3
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path

import aiosqlite

MIGRATIONS_DIR = Path(__file__).resolve().parents[2] / "migrations"


@asynccontextmanager
async def connect_database(path: Path) -> AsyncIterator[aiosqlite.Connection]:
    async with aiosqlite.connect(path) as connection:
        connection.row_factory = aiosqlite.Row
        await connection.execute("PRAGMA busy_timeout = 5000")
        await connection.execute("PRAGMA foreign_keys = ON")
        yield connection


def migration_statements(source: str) -> list[str]:
    statements = []
    pending = ""
    for line in source.splitlines(keepends=True):
        pending += line
        if sqlite3.complete_statement(pending):
            statements.append(pending.strip())
            pending = ""
    if pending.strip():
        raise ValueError("Migration ends with an incomplete SQL statement.")
    return statements


async def initialize_database(path: Path, *, migrations_dir: Path = MIGRATIONS_DIR) -> None:
    """Apply pending migrations explicitly; schema and ledger commit together."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    files = sorted(migrations_dir.glob("[0-9][0-9][0-9]_*.sql"))
    versions = [int(file.name[:3]) for file in files]
    if not files or len(set(versions)) != len(files):
        raise ValueError("Migrations must have unique numbered SQL files.")

    async with connect_database(path) as connection:
        await connection.execute("PRAGMA journal_mode = WAL")
        await connection.execute("BEGIN IMMEDIATE")
        try:
            await connection.execute(
                "CREATE TABLE IF NOT EXISTS schema_migrations "
                "(version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL)"
            )
            async with connection.execute("SELECT version FROM schema_migrations") as cursor:
                applied = {row[0] for row in await cursor.fetchall()}
            if applied - set(versions):
                raise ValueError("A database migration is missing from the repository.")
            for version, file in zip(versions, files, strict=True):
                if version in applied:
                    continue
                for statement in migration_statements(file.read_text(encoding="utf-8")):
                    await connection.execute(statement)
                await connection.execute(
                    "INSERT INTO schema_migrations (version, applied_at) VALUES (?, ?)",
                    (version, datetime.now(UTC).isoformat(timespec="microseconds")),
                )
            await connection.commit()
        except BaseException:
            await connection.rollback()
            raise
