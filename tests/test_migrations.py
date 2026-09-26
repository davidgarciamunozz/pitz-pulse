import asyncio

import aiosqlite
import pytest

from app.persistence.database import initialize_database
from app.persistence.repository import RequestRepository


def test_migrations_initialize_and_are_repeatable(tmp_path):
    async def scenario():
        path = tmp_path / "migrations.sqlite3"
        await initialize_database(path)
        await RequestRepository(path).reserve("R-1", "Original")
        await initialize_database(path)
        assert (await RequestRepository(path).get("R-1")).raw_message == "Original"
        async with aiosqlite.connect(path) as connection:
            async with connection.execute("SELECT version FROM schema_migrations") as cursor:
                assert await cursor.fetchall() == [(1,)]

    asyncio.run(scenario())


def test_failed_migration_rolls_back_schema_and_version(tmp_path):
    async def scenario():
        path = tmp_path / "broken.sqlite3"
        migrations = tmp_path / "migrations"
        migrations.mkdir()
        (migrations / "001_broken.sql").write_text(
            "CREATE TABLE partial_table (id TEXT);\nINSERT INTO missing_table (id) VALUES ('x');\n",
            encoding="utf-8",
        )
        with pytest.raises(aiosqlite.OperationalError):
            await initialize_database(path, migrations_dir=migrations)
        async with aiosqlite.connect(path) as connection:
            async with connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ) as cursor:
                assert await cursor.fetchall() == []

    asyncio.run(scenario())
