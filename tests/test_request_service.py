import asyncio

import aiosqlite
import pytest

from app.classification.pipeline import ClassificationPipeline
from app.persistence.database import initialize_database
from app.persistence.models import IdempotencyConflict, InProgress, RequestState
from app.persistence.repository import RequestRepository
from app.services.requests import RequestService
from tests.fakes import RecordingFakeClassifier


def test_simultaneous_identical_submissions_use_one_database_reservation(
    tmp_path, valid_model_output
):
    async def scenario():
        path = tmp_path / "simultaneous.sqlite3"
        await initialize_database(path)
        participant_count = 12
        ready = 0
        returned = 0
        all_ready = asyncio.Event()
        start = asyncio.Event()
        all_non_winners_returned = asyncio.Event()
        classifier_entered = asyncio.Event()
        release_classifier = asyncio.Event()

        class GatedClassifier(RecordingFakeClassifier):
            call_count = 0

            async def classify(self, *, masked_message, prompt):
                self.call_count += 1
                classifier_entered.set()
                await release_classifier.wait()
                return await super().classify(masked_message=masked_message, prompt=prompt)

        fake = GatedClassifier(valid_model_output)
        services = [
            RequestService(RequestRepository(path), ClassificationPipeline(fake))
            for _ in range(participant_count)
        ]

        async def submit(service):
            nonlocal ready, returned
            ready += 1
            if ready == participant_count:
                all_ready.set()
            await start.wait()
            outcome = await service.submit(message_id="shared", raw_message="Same request")
            returned += 1
            if returned == participant_count - 1:
                all_non_winners_returned.set()
            return outcome

        tasks = [asyncio.create_task(submit(service)) for service in services]
        await all_ready.wait()
        start.set()
        try:
            await asyncio.wait_for(classifier_entered.wait(), timeout=3)
            await asyncio.wait_for(all_non_winners_returned.wait(), timeout=3)
            assert fake.call_count == 1
        finally:
            release_classifier.set()
        outcomes = await asyncio.gather(*tasks)
        completed = [
            outcome
            for outcome in outcomes
            if not isinstance(outcome, InProgress) and outcome.state is RequestState.COMPLETED
        ]
        assert len(completed) == 1
        assert all(
            isinstance(outcome, InProgress) or outcome == completed[0] for outcome in outcomes
        )
        assert fake.call_count == len(fake.calls) == 1
        async with aiosqlite.connect(path) as connection:
            async with connection.execute("SELECT COUNT(*) FROM requests") as cursor:
                assert (await cursor.fetchone())[0] == 1

    asyncio.run(scenario())


def test_concurrent_identical_requests_get_in_progress_then_stored_result(
    tmp_path, valid_model_output
):
    async def scenario():
        path = tmp_path / "concurrent.sqlite3"
        await initialize_database(path)
        entered = asyncio.Event()
        release = asyncio.Event()

        class GatedClassifier(RecordingFakeClassifier):
            async def classify(self, *, masked_message, prompt):
                entered.set()
                await release.wait()
                return await super().classify(masked_message=masked_message, prompt=prompt)

        fake = GatedClassifier(valid_model_output)
        first_service = RequestService(RequestRepository(path), ClassificationPipeline(fake))
        second_service = RequestService(RequestRepository(path), ClassificationPipeline(fake))
        first = asyncio.create_task(first_service.submit(message_id="same", raw_message="Request"))
        await entered.wait()
        duplicate = await asyncio.wait_for(
            second_service.submit(message_id="same", raw_message="Request"), timeout=1
        )
        assert isinstance(duplicate, InProgress)
        assert duplicate.id == "same"
        with pytest.raises(IdempotencyConflict):
            await second_service.submit(message_id="same", raw_message="Different")
        assert fake.calls == []
        release.set()
        completed = await first
        assert completed.state is RequestState.COMPLETED
        assert len(fake.calls) == 1
        assert await second_service.submit(message_id="same", raw_message="Request") == completed
        assert len(await second_service.list_completed()) == 1

    asyncio.run(scenario())


@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_finalization_failure_preserves_original_error_and_never_reclassifies(
    tmp_path, valid_model_output, cleanup_fails
):
    async def scenario():
        path = tmp_path / "finalization.sqlite3"
        await initialize_database(path)
        original_error = OSError("synthetic finalization failure")

        class FailingCompleteRepository(RequestRepository):
            failure_codes = []

            async def complete(self, message_id, result):
                raise original_error

            async def mark_failed(self, message_id, failure_code):
                self.failure_codes.append(failure_code)
                if cleanup_fails:
                    raise RuntimeError("synthetic cleanup failure")
                return await super().mark_failed(message_id, failure_code)

        repository = FailingCompleteRepository(path)
        fake = RecordingFakeClassifier(valid_model_output)
        service = RequestService(repository, ClassificationPipeline(fake))
        with pytest.raises(OSError) as failure:
            await service.submit(message_id="finalization", raw_message="Original")
        assert failure.value is original_error
        assert repository.failure_codes == ["persistence_error"]
        assert len(fake.calls) == 1

        stored = await RequestRepository(path).get("finalization")
        assert stored.state is (RequestState.PROCESSING if cleanup_fails else RequestState.FAILED)
        if cleanup_fails:
            assert isinstance(
                await service.submit(message_id="finalization", raw_message="Original"),
                InProgress,
            )
        else:
            assert stored.failure_code == "persistence_error"
            assert await service.submit(message_id="finalization", raw_message="Original") == stored
        assert len(fake.calls) == 1

    asyncio.run(scenario())


def test_slow_classifier_does_not_hold_database_write_transaction(tmp_path, valid_model_output):
    async def scenario():
        path = tmp_path / "unlocked.sqlite3"
        await initialize_database(path)
        entered = asyncio.Event()
        release = asyncio.Event()

        class GatedClassifier(RecordingFakeClassifier):
            async def classify(self, *, masked_message, prompt):
                entered.set()
                await release.wait()
                return await super().classify(masked_message=masked_message, prompt=prompt)

        fake = GatedClassifier(valid_model_output)
        repository = RequestRepository(path)
        service = RequestService(repository, ClassificationPipeline(fake))
        first = asyncio.create_task(service.submit(message_id="slow", raw_message="Slow request"))
        await entered.wait()
        other = await asyncio.wait_for(repository.reserve("other", "Another request"), timeout=1)
        assert other.won and other.record.state is RequestState.PROCESSING
        await repository.mark_failed("other", "test_failure")
        release.set()
        assert (await first).state is RequestState.COMPLETED

    asyncio.run(scenario())


def test_failed_reservation_is_terminal_for_identical_submission(tmp_path, valid_model_output):
    async def scenario():
        path = tmp_path / "failed.sqlite3"
        await initialize_database(path)
        fake = RecordingFakeClassifier(valid_model_output, failure=RuntimeError("internal details"))
        service = RequestService(RequestRepository(path), ClassificationPipeline(fake))
        first = await service.submit(message_id="failed", raw_message="Request")
        assert first.state is RequestState.FAILED
        assert first.failure_code == "unexpected_error"
        assert first.ai_classification is None
        assert len(fake.calls) == 1
        assert await service.submit(message_id="failed", raw_message="Request") == first
        assert len(fake.calls) == 1
        with pytest.raises(IdempotencyConflict):
            await service.submit(message_id="failed", raw_message="Different")

    asyncio.run(scenario())


def test_abandoned_processing_reservation_never_classifies_automatically(
    tmp_path, valid_model_output
):
    async def scenario():
        path = tmp_path / "abandoned.sqlite3"
        await initialize_database(path)
        repository = RequestRepository(path)
        await repository.reserve("abandoned", "Original")
        fake = RecordingFakeClassifier(valid_model_output)
        service = RequestService(repository, ClassificationPipeline(fake))
        outcome = await service.submit(message_id="abandoned", raw_message="Original")
        assert isinstance(outcome, InProgress)
        assert fake.calls == []
        assert (await repository.get("abandoned")).state is RequestState.PROCESSING

    asyncio.run(scenario())


def test_cancelled_classification_is_not_restarted(tmp_path, valid_model_output):
    async def scenario():
        path = tmp_path / "cancelled.sqlite3"
        await initialize_database(path)
        entered = asyncio.Event()

        class BlockingClassifier(RecordingFakeClassifier):
            call_count = 0

            async def classify(self, *, masked_message, prompt):
                self.call_count += 1
                entered.set()
                await asyncio.Event().wait()

        fake = BlockingClassifier(valid_model_output)
        repository = RequestRepository(path)
        service = RequestService(repository, ClassificationPipeline(fake))
        task = asyncio.create_task(service.submit(message_id="cancelled", raw_message="Request"))
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        stored = await repository.get("cancelled")
        assert stored.state is RequestState.FAILED
        assert stored.failure_code == "cancelled"
        assert await service.submit(message_id="cancelled", raw_message="Request") == stored
        assert fake.call_count == 1

    asyncio.run(scenario())
